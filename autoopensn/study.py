"""Running a whole spec: render every point, run it, cache it, tabulate it.

This is the deterministic middle, end to end, and it is deliberately small.
Every hard decision is made somewhere else: what to run is the spec's, what the
script says is the template's, whether to execute is the runner's, what happened
is the parser's. What is left here is order, caching, and retries.
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import pandas as pd

from autoopensn.parse.table import results_table
from autoopensn.runner.base import RunRequest, RunResult, Runner, RunnerError, run_directory
from autoopensn.runner.fake import FixtureMissing
from autoopensn.runner.reference import UnsupportedScenario
from autoopensn.spec.model import Spec, SweepPoint
from autoopensn.store.db import RunStore, cache_key
from autoopensn.templates import Template, render_run_script

Progress = Callable[[str], None]


@dataclass
class RenderedPoint:
    """One sweep point with its script, before anything has run."""

    point: SweepPoint
    script: str

    @property
    def case_id(self) -> str:
        return self.point.case_id


@dataclass
class StudyResult:
    """Everything a study produced."""

    spec: Spec
    points: list[SweepPoint]
    results: list[RunResult]
    table: pd.DataFrame
    cache_hits: list[str] = field(default_factory=list)
    run_root: Optional[Path] = None

    @property
    def failures(self) -> list[RunResult]:
        return [result for result in self.results if not result.ok]

    def summary(self) -> str:
        total = len(self.points)
        failed = len(self.failures)
        cached = len(self.cache_hits)
        parts = [f"{total} case{'s' if total != 1 else ''}"]
        if cached:
            parts.append(f"{cached} from cache")
        if failed:
            parts.append(f"{failed} failed")
        return ", ".join(parts)


def render_points(
    spec: Spec, template: Optional[Template] = None
) -> list[RenderedPoint]:
    """Render every sweep point. Pure, and safe to inspect before running."""
    template = template or spec.validate_against()
    rendered = []
    for point in spec.expand(template):
        script = render_run_script(
            template,
            point.parameters,
            pack_commit=spec.pack_commit,
            spec_name=spec.name,
            case_id=point.case_id,
            varied=point.varied,
        )
        rendered.append(RenderedPoint(point=point, script=script))
    return rendered


def write_scripts(spec: Spec, directory: Path, template: Optional[Template] = None) -> list[Path]:
    """Render every point and write the scripts out, without running them."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for rendered in render_points(spec, template):
        path = directory / f"{rendered.case_id}.py"
        path.write_text(rendered.script)
        written.append(path)
    (directory / "spec.yaml").write_text(spec.to_yaml())
    return written


TUTORIALS_IN_SOURCE = "doc/source/tutorials"


def data_files_for(template: Template) -> tuple[str, ...]:
    """A template's data files, as paths inside the OpenSn source tree.

    Each tutorial keeps its meshes and cross sections beside its notebook, and
    the template records which notebook it came from, so the path is the
    notebook's directory plus the file name.
    """
    if not template.data_files or not template.generated_from:
        return ()
    folder = Path(TUTORIALS_IN_SOURCE) / Path(template.generated_from).parent
    paths = []
    for name in template.data_files:
        # A helper reached as "../../../../../tools/x.py" from the notebook is
        # really "tools/x.py" in the source tree. Normalised here so the runner
        # sees a plain path, and refused if it climbs out of the tree.
        path = posixpath.normpath(str(folder / name))
        if path.startswith(".."):
            raise ValueError(f"data file {name!r} of {template.name} is outside the OpenSn source")
        paths.append(path)
    return tuple(paths)


def workdir_for(template: Template) -> str:
    """The notebook's folder inside the source tree, or empty when there is none."""
    if not template.generated_from:
        return ""
    return str(Path(TUTORIALS_IN_SOURCE) / Path(template.generated_from).parent)


def run_study(
    spec: Spec,
    runner: Runner,
    *,
    template: Optional[Template] = None,
    run_root: Path = Path("runs"),
    run_id: Optional[str] = None,
    store: Optional[RunStore] = None,
    use_cache: bool = True,
    progress: Optional[Progress] = None,
    tests_json: Optional[Path] = None,
) -> StudyResult:
    """Render, run, cache, and tabulate every point of a spec."""
    template = template or spec.validate_against()
    run_id = run_id or spec.name
    say: Progress = progress or (lambda _message: None)

    rendered_points = render_points(spec, template)
    spec_yaml = spec.to_yaml()
    runner_config = runner.config()

    results: list[RunResult] = []
    cache_hits: list[str] = []

    for index, rendered in enumerate(rendered_points, start=1):
        label = f"[{index}/{len(rendered_points)}] {rendered.case_id}"
        key = cache_key(rendered.script, spec.pack_commit, runner_config)

        if use_cache and store is not None:
            cached = store.get(key)
            if cached is not None:
                say(f"{label}: cached")
                cache_hits.append(rendered.case_id)
                results.append(cached)
                continue

        directory = run_directory(run_root, run_id, rendered.case_id)
        request = RunRequest(
            case_id=rendered.case_id,
            script=rendered.script,
            directory=directory,
            template=template.name,
            num_procs=spec.num_procs,
            timeout_seconds=spec.timeout_seconds,
            parameters=rendered.point.parameters,
            data_files=data_files_for(template),
            workdir=workdir_for(template),
        )
        request.write_inputs(spec_yaml=spec_yaml)

        result = _run_with_retries(runner, request, spec.retry_budget, say, label)
        results.append(result)

        if store is not None:
            store.put(
                key,
                result,
                script=rendered.script,
                pack_commit=spec.pack_commit,
                runner_config=runner_config,
                study=spec.name,
            )

    points = [rendered.point for rendered in rendered_points]
    table = results_table(results, points, template, tests_json=tests_json)

    return StudyResult(
        spec=spec,
        points=points,
        results=results,
        table=table,
        cache_hits=cache_hits,
        run_root=Path(run_root) / run_id,
    )


def _run_with_retries(
    runner: Runner,
    request: RunRequest,
    budget: int,
    say: Progress,
    label: str,
) -> RunResult:
    """Run one point, retrying a failure up to ``budget`` times.

    Retries exist for the failures that are not about the physics: a launcher
    that lost a rank, a filesystem that was briefly busy. A case that diverges
    will diverge identically every time, so the budget is small by default and
    zero is a reasonable setting.
    """
    attempt = 0
    while True:
        try:
            result = runner.run(request)
        except (FixtureMissing, UnsupportedScenario) as missing:
            # A study generated from a request will usually have no recording,
            # because recordings exist only for cases someone has run. That is
            # a missing measurement, not a broken study: the point gets a row
            # saying so, and the other points still run. Crashing here would
            # throw away the rendered scripts, which are the useful output when
            # OpenSn is not installed.
            say(f"{label}: not available ({type(missing).__name__})")
            return RunResult(
                case_id=request.case_id,
                exit_code=127,
                stdout="",
                stderr=str(missing) + "\n",
                wall_time=0.0,
                directory=Path(request.directory),
                runner=getattr(runner, "name", "unknown"),
                command=(),
                replayed=False,
            )
        if result.ok or attempt >= budget:
            if result.ok:
                say(f"{label}: ok ({result.wall_time:.1f}s)")
            else:
                say(f"{label}: FAILED ({result.failure_reason()})")
            return result
        attempt += 1
        say(f"{label}: {result.failure_reason()}; retry {attempt}/{budget}")
