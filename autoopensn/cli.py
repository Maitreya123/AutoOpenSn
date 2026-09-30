"""The ``autoopensn`` command line.

Five commands, one per stage, plus ``study`` to chain them. Every stage runs on
its own and writes its output to a file, which is the point: a user should be
able to generate a spec, read it, edit it, and only then run it. A tool that
goes from a sentence to a running simulation in one step is a tool whose
mistakes are discovered by the cluster.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

from autoopensn import PINNED_OPENSN_COMMIT, __version__
from autoopensn.parse.table import convergence_table, format_table
from autoopensn.runner import (
    DockerRunner,
    FakeRunner,
    LocalMPIRunner,
    ReferenceRunner,
    Runner,
    RunnerError,
)
from autoopensn.spec import SpecError, load_spec, save_spec
from autoopensn.store import RunStore
from autoopensn.study import run_study, write_scripts
from autoopensn.templates import TemplateError, list_templates, load_template

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Parameter studies for the OpenSn neutron transport code.",
)

DEFAULT_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "reed_gmres"


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _make_runner(kind: str, fixtures: Optional[Path], num_procs: int) -> Runner:
    if kind == "reference":
        # strict=False so a study touching an unsupported scenario reports those
        # cases as unavailable rather than stopping the whole run.
        return ReferenceRunner(strict=False)
    if kind == "docker":
        runner = DockerRunner()
        try:
            runner.preflight()
        except RunnerError as exc:
            _fail(str(exc))
        return runner
    if kind == "fake":
        directory = fixtures or DEFAULT_FIXTURES
        try:
            return FakeRunner(directory)
        except RunnerError as exc:
            _fail(str(exc))
    if kind == "local":
        runner = LocalMPIRunner()
        try:
            runner.preflight()
        except RunnerError as exc:
            _fail(
                f"{exc}\n\nRun with --runner fake to replay recorded output instead."
            )
        if not runner.pyopensn_available():
            typer.secho(
                "warning: pyopensn is not importable by this interpreter; the run "
                "will probably fail at import time. See docs/installing_opensn.md.",
                fg=typer.colors.YELLOW,
                err=True,
            )
        return runner
    _fail(
        f"unknown runner {kind!r}; expected 'reference', 'fake', 'local' or 'docker'"
    )
    raise AssertionError("unreachable")


@app.command()
def ui(
    port: int = typer.Option(8501, help="Port to serve on."),
    headless: bool = typer.Option(True, help="Do not open a browser automatically."),
) -> None:
    """Launch the Streamlit interface.

    The page is a front end over the same functions these commands call, so
    anything it shows can also be produced here, and anything produced here
    shows up there.
    """
    import subprocess

    from autoopensn.ui import APP_PATH

    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.port",
        str(port),
        "--server.headless",
        "true" if headless else "false",
    ]
    try:
        raise SystemExit(subprocess.call(command))
    except FileNotFoundError:
        _fail(
            "streamlit is not installed. Install the interface extra:\n"
            "    pip install -e '.[ui]'"
        )


@app.command()
def version() -> None:
    """Print the version and the pinned OpenSn commit."""
    typer.echo(f"autoopensn {__version__}")
    typer.echo(f"pinned OpenSn commit: {PINNED_OPENSN_COMMIT}")


@app.command()
def templates(
    name: Optional[str] = typer.Argument(None, help="Show one template's parameters."),
) -> None:
    """List the available problem templates, or describe one."""
    if name is None:
        for template_name in list_templates():
            template = load_template(template_name)
            typer.echo(f"{template_name}: {template.description.strip().splitlines()[0]}")
        return

    try:
        template = load_template(name)
    except TemplateError as exc:
        _fail(str(exc))

    typer.echo(f"{template.name}\n{template.description.strip()}\n")
    typer.echo(f"derived from: {template.source}")
    typer.echo(f"opensn commit: {template.opensn_commit}")
    typer.echo(f"observables: {', '.join(template.observables)}\n")
    for group in ("mesh", "materials", "source", "quadrature", "solver", "boundary", "output", "runtime"):
        names = template.names_in_group(group)
        if not names:
            continue
        typer.echo(f"[{group}]")
        for parameter_name in names:
            declaration = template.parameters[parameter_name]
            bounds = []
            if declaration.choices:
                bounds.append("one of " + ", ".join(map(str, declaration.choices)))
            if declaration.minimum is not None:
                bounds.append(f">= {declaration.minimum:g}")
            if declaration.maximum is not None:
                bounds.append(f"<= {declaration.maximum:g}")
            suffix = f"  ({'; '.join(bounds)})" if bounds else ""
            typer.echo(f"  {parameter_name} = {declaration.default!r}{suffix}")
        typer.echo("")


@app.command()
def spec(
    prompt: str = typer.Argument(..., help="The study, described in plain language."),
    template: Optional[str] = typer.Option(None, help="Force a scenario instead of choosing one."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Where to write the spec."),
    review: bool = typer.Option(True, help="Run the domain reviewer over the generated spec."),
) -> None:
    """Turn a prompt into a spec and print it. Runs nothing.

    This stage calls a language model. The spec it produces is validated against
    the scenario's declared parameter ranges before you see it, so a spec that
    prints is a spec that renders.
    """
    from autoopensn import narrate

    try:
        draft = narrate.request_to_spec(prompt, template=template, review=review)
    except Exception as exc:
        _fail(str(exc))

    if draft.candidates:
        typer.secho(
            "considered: " + ", ".join(s.name for s in draft.candidates),
            err=True,
            fg=typer.colors.BRIGHT_BLACK,
        )
    for note in draft.notes:
        typer.secho(f"reviewer note: {note}", err=True, fg=typer.colors.YELLOW)

    if not draft.ok:
        for error in draft.errors:
            typer.secho(error, err=True, fg=typer.colors.RED)
        _fail("no valid spec was produced. The raw model output is above.")

    typer.secho(draft.summary(), err=True, fg=typer.colors.BRIGHT_BLACK)
    if output:
        save_spec(draft.spec, output)
        typer.secho(f"wrote {output}", err=True)
    typer.echo(draft.spec.to_yaml())


@app.command()
def render(
    spec_path: Path = typer.Argument(..., help="Spec to render."),
    output: Path = typer.Option(Path("runs/render"), "--output", "-o", help="Directory for scripts."),
) -> None:
    """Render a spec into one script per sweep point. Does not run anything."""
    try:
        loaded = load_spec(spec_path)
        written = write_scripts(loaded, output)
    except (SpecError, TemplateError) as exc:
        _fail(str(exc))

    for path in written:
        typer.echo(str(path))
    typer.echo(f"\n{len(written)} script(s) in {output}")


@app.command()
def run(
    spec_path: Path = typer.Argument(..., help="Spec to run."),
    runner: str = typer.Option("reference", help="'reference' solves 1D scenarios here and now; 'fake' replays fixtures; 'local' runs MPI; 'docker' runs in a container."),
    fixtures: Optional[Path] = typer.Option(None, help="Fixture directory for the fake runner."),
    run_root: Path = typer.Option(Path("runs"), help="Where run directories are written."),
    cache: bool = typer.Option(True, help="Reuse cached results for identical points."),
    cache_path: Path = typer.Option(Path("runs/cache.db"), help="Run cache database."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write the full table as CSV."),
    full: bool = typer.Option(False, help="Print every column, not the four-column view."),
) -> None:
    """Run a spec and print the results table."""
    try:
        loaded = load_spec(spec_path)
    except (SpecError, TemplateError) as exc:
        _fail(str(exc))

    active_runner = _make_runner(runner, fixtures, loaded.num_procs)

    def say(message: str) -> None:
        typer.secho(message, err=True, fg=typer.colors.BRIGHT_BLACK)

    with RunStore(cache_path) as store:
        try:
            result = run_study(
                loaded,
                active_runner,
                run_root=run_root,
                store=store,
                use_cache=cache,
                progress=say,
            )
        except RunnerError as exc:
            _fail(str(exc))

    table = result.table if full else convergence_table(result.table)
    typer.echo("")
    typer.echo(format_table(table))
    typer.echo(f"\n{result.summary()}")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        result.table.to_csv(output, index=False)
        typer.echo(f"wrote {output}")

    if result.failures:
        raise typer.Exit(code=2)


@app.command()
def explain(
    table_path: Path = typer.Argument(..., help="Results table, as written by `run --output`."),
    spec_path: Optional[Path] = typer.Argument(None, help="The spec that produced it."),
    question: Optional[str] = typer.Option(None, help="A specific question about the results."),
) -> None:
    """Explain a results table in prose.

    The model receives the table and the spec. It does not receive logs or run
    directories, so anything it says should be findable in the table.
    """
    import pandas as pd

    from autoopensn import narrate

    try:
        frame = pd.read_csv(table_path)
        loaded = load_spec(spec_path) if spec_path else None
        explanation = narrate.explain(frame, loaded, question=question)
    except (SpecError, TemplateError, narrate.NarrationError) as exc:
        _fail(str(exc))
    except FileNotFoundError as exc:
        _fail(str(exc))

    typer.echo(explanation.text)
    for warning in explanation.warnings:
        typer.secho(warning, err=True, fg=typer.colors.YELLOW)


@app.command()
def study(
    prompt: str = typer.Argument(..., help="The study, described in plain language."),
    template: Optional[str] = typer.Option(None, help="Force a scenario instead of choosing one."),
    runner: str = typer.Option("reference", help="'reference' solves 1D scenarios here and now; 'fake' replays fixtures; 'local' runs MPI; 'docker' runs in a container."),
    fixtures: Optional[Path] = typer.Option(None, help="Fixture directory for the fake runner."),
    run_root: Path = typer.Option(Path("runs"), help="Where run directories are written."),
    cache_path: Path = typer.Option(Path("runs/cache.db"), help="Run cache database."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Write the table as CSV."),
) -> None:
    """Prompt to spec to scripts to runs to table to narrative, in one command.

    Every stage is also available on its own, which is the better way to work:
    this one goes from a sentence to a running simulation without stopping for
    you to read the spec.
    """
    from autoopensn import narrate

    def say(message: str) -> None:
        typer.secho(message, err=True, fg=typer.colors.BRIGHT_BLACK)

    say("generating a spec…")
    try:
        draft = narrate.request_to_spec(prompt, template=template)
    except Exception as exc:
        _fail(str(exc))
    if not draft.ok:
        for error in draft.errors:
            typer.secho(error, err=True, fg=typer.colors.RED)
        _fail("no valid spec was produced.")

    typer.echo(draft.spec.to_yaml())
    say(draft.summary())

    active_runner = _make_runner(runner, fixtures, draft.spec.num_procs)
    with RunStore(cache_path) as store:
        try:
            result = run_study(
                draft.spec, active_runner, run_root=run_root, store=store, progress=say
            )
        except RunnerError as exc:
            _fail(str(exc))

    typer.echo("")
    typer.echo(format_table(convergence_table(result.table)))
    typer.echo(f"\n{result.summary()}")

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        result.table.to_csv(output, index=False)
        say(f"wrote {output}")

    if not any(r.ok for r in result.results):
        _fail(
            "no case produced a result, so there is nothing to explain. "
            "Install OpenSn and use --runner local, or record fixtures for these cases."
        )

    say("explaining…")
    try:
        explanation = narrate.explain(result.table, draft.spec)
    except narrate.NarrationError as exc:
        _fail(str(exc))

    typer.echo("")
    typer.echo(explanation.text)
    for warning in explanation.warnings:
        typer.secho(warning, err=True, fg=typer.colors.YELLOW)


@app.command()
def scenarios(
    query: Optional[str] = typer.Argument(None, help="Narrow the list to what matches."),
    rebuild: bool = typer.Option(False, help="Re-read the tutorials from the OpenSn checkout."),
    limit: int = typer.Option(15, help="How many to show."),
    show: Optional[str] = typer.Option(None, help="Print one scenario in full."),
) -> None:
    """List the OpenSn tutorials available as scenarios.

    Each is a real tutorial from the pinned checkout, with the parameters read
    out of its own script and, for nearly all of them, the correct answer
    recorded by OpenSn's test suite.
    """
    from autoopensn.tutorials import catalog

    if rebuild:
        from autoopensn.tutorials import rebuild as rebuild_all

        try:
            report = rebuild_all()
        except Exception as exc:
            _fail(str(exc))
        typer.secho(report.summary(), fg=typer.colors.GREEN)
        for name, error in report.failures:
            typer.secho(f"  {name}: {error}", fg=typer.colors.RED)

    available = catalog.load()
    if not available:
        _fail("the catalog is empty. Run `autoopensn scenarios --rebuild`.")

    if show:
        scenario = catalog.by_name(show, available)
        if scenario is None:
            _fail(f"no scenario named {show!r}")
        typer.echo(scenario.catalog_entry())
        return

    listed = catalog.rank(query, available, limit=limit) if query else available[:limit]
    for scenario in listed:
        gold = "✓" if scenario.has_gold else " "
        knobs = f"{len(scenario.parameters):3d} knobs"
        typer.echo(f"{gold} {scenario.name:34s} {knobs}  {scenario.title[:44]}")
    typer.echo(f"\n{len(available)} scenario(s) available; ✓ marks a recorded correct answer")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
