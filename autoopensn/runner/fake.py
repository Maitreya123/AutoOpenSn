"""Replaying recorded runs.

A fixture is a directory named after a case id:

    tests/fixtures/<study>/<case_id>/
        stdout.txt
        stderr.txt        optional
        meta.json         exit_code, wall_time, and provenance of the recording
        <outputs>         optional; copied into the run directory

``FakeRunner`` looks up the case id and replays it. It does not run anything, it
does not need OpenSn, and it reports the *recorded* wall time rather than the
microseconds the replay took, because a table of replay times would be a table
of nothing.

A missing fixture is a loud error listing what is available. Silently returning
empty output would produce a results table full of blanks, which looks like a
physics result and is not one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from autoopensn.runner.base import RunRequest, RunResult, Runner, RunnerError

RESERVED = {"stdout.txt", "stderr.txt", "meta.json", "README.md"}


class FixtureMissing(RunnerError):
    """No recording exists for the requested case."""


class FakeRunner(Runner):
    """Replays runs recorded under a fixture directory."""

    name = "fake"

    def __init__(self, fixture_dir: Path, *, copy_outputs: bool = True):
        self.fixture_dir = Path(fixture_dir)
        self.copy_outputs = copy_outputs
        if not self.fixture_dir.is_dir():
            raise RunnerError(f"no fixture directory at {self.fixture_dir}")

    # --- introspection ---

    def available_cases(self) -> list[str]:
        return sorted(
            child.name
            for child in self.fixture_dir.iterdir()
            if child.is_dir() and (child / "stdout.txt").exists()
        )

    def fixture_for(self, case_id: str) -> Path:
        path = self.fixture_dir / case_id
        if not (path / "stdout.txt").exists():
            raise FixtureMissing(
                f"no recorded run for case {case_id!r} in {self.fixture_dir}. "
                f"Recorded cases: {', '.join(self.available_cases()) or 'none'}"
            )
        return path

    # --- the interface ---

    def run(self, request: RunRequest) -> RunResult:
        fixture = self.fixture_for(request.case_id)
        meta = self._meta(fixture)

        request.write_inputs()
        directory = Path(request.directory)

        if self.copy_outputs:
            for child in fixture.iterdir():
                if child.name in RESERVED or child.name.startswith("."):
                    continue
                target = directory / child.name
                if child.is_dir():
                    shutil.copytree(child, target, dirs_exist_ok=True)
                else:
                    shutil.copy2(child, target)

        stderr_path = fixture / "stderr.txt"
        result = RunResult(
            case_id=request.case_id,
            exit_code=int(meta.get("exit_code", 0)),
            stdout=(fixture / "stdout.txt").read_text(),
            stderr=stderr_path.read_text() if stderr_path.exists() else "",
            wall_time=float(meta.get("wall_time", 0.0)),
            directory=directory,
            runner=self.name,
            command=("<replayed>", str(fixture)),
            timed_out=bool(meta.get("timed_out", False)),
            replayed=True,
            started_at=meta.get("recorded_at"),
        )
        result.persist()
        return result

    def config(self) -> dict[str, Any]:
        return {"runner": self.name, "fixtures": str(self.fixture_dir.resolve())}

    # --- internals ---

    @staticmethod
    def _meta(fixture: Path) -> dict[str, Any]:
        path = fixture / "meta.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            raise RunnerError(f"{path}: not valid JSON: {exc}") from exc


def record_fixture(
    result: RunResult,
    fixture_dir: Path,
    *,
    case_id: Optional[str] = None,
    provenance: str = "recorded from a real run",
    outputs: tuple[str, ...] = (),
) -> Path:
    """Write a completed run into the fixture layout, so it can be replayed.

    This is how a hand-written fixture is eventually replaced by a real one: run
    the case once on a machine that has OpenSn, and call this with the result.
    """
    target = Path(fixture_dir) / (case_id or result.case_id)
    target.mkdir(parents=True, exist_ok=True)
    (target / "stdout.txt").write_text(result.stdout)
    if result.stderr:
        (target / "stderr.txt").write_text(result.stderr)
    (target / "meta.json").write_text(
        json.dumps(
            {
                "exit_code": result.exit_code,
                "wall_time": result.wall_time,
                "timed_out": result.timed_out,
                "recorded_at": result.started_at,
                "runner": result.runner,
                "provenance": provenance,
            },
            indent=2,
        )
    )
    for name in outputs:
        source = Path(result.directory) / name
        if source.exists():
            shutil.copy2(source, target / name)
    return target
