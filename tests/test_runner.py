"""Tests for the Runner interface and both implementations.

The real runner is exercised for everything except actually launching MPI,
which is gated behind an environment variable and skipped by default. That is
the point of the split: the parts of a runner that have bugs, command
construction, timeout handling, the run directory layout, are testable without
a supercomputer.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from autoopensn.runner import (
    FakeRunner,
    FixtureMissing,
    LocalMPIRunner,
    RunRequest,
    RunResult,
    RunnerError,
    run_directory,
)
from autoopensn.runner.fake import record_fixture


def request_for(tmp_path: Path, case_id: str = "reference") -> RunRequest:
    return RunRequest(
        case_id=case_id,
        script="print('hello')\n",
        directory=tmp_path / case_id,
        num_procs=1,
        timeout_seconds=30.0,
        parameters={"l_abs_tol": 1.0e-10},
    )


# --- run directories --------------------------------------------------------


def test_run_directory_is_created(tmp_path):
    path = run_directory(tmp_path, "study", "case")
    assert path.is_dir()
    assert path == tmp_path / "study" / "case"


def test_run_directory_sanitizes_its_components(tmp_path):
    path = run_directory(tmp_path, "../escape", "a/b")
    assert tmp_path in path.parents or path.parent.parent == tmp_path
    assert ".." not in str(path.relative_to(tmp_path))


def test_write_inputs_records_the_script_and_parameters(tmp_path):
    request = request_for(tmp_path)
    request.write_inputs(spec_yaml="name: x\n")
    assert (request.directory / "script.py").read_text() == request.script
    assert json.loads((request.directory / "parameters.json").read_text())["l_abs_tol"] == 1e-10
    assert (request.directory / "spec.yaml").read_text() == "name: x\n"


# --- FakeRunner -------------------------------------------------------------


def test_fake_runner_replays_recorded_output(tmp_path, gmres_fixtures):
    runner = FakeRunner(gmres_fixtures)
    result = runner.run(request_for(tmp_path))
    assert result.ok
    assert result.replayed
    assert "Absorption=1.006178e+02" in result.stdout
    assert result.wall_time > 0


def test_fake_runner_reports_the_recorded_wall_time_not_the_replay_time(
    tmp_path, gmres_fixtures
):
    """A table of replay times would be a table of nothing."""
    runner = FakeRunner(gmres_fixtures)
    result = runner.run(request_for(tmp_path))
    recorded = json.loads((gmres_fixtures / "reference" / "meta.json").read_text())
    assert result.wall_time == pytest.approx(recorded["wall_time"])
    assert result.wall_time > 1.0


def test_fake_runner_writes_the_run_directory(tmp_path, gmres_fixtures):
    runner = FakeRunner(gmres_fixtures)
    result = runner.run(request_for(tmp_path))
    assert (result.directory / "script.py").exists()
    assert (result.directory / "stdout.txt").exists()
    assert (result.directory / "result.json").exists()


def test_missing_fixture_is_a_loud_error_listing_what_exists(tmp_path, gmres_fixtures):
    runner = FakeRunner(gmres_fixtures)
    with pytest.raises(FixtureMissing) as excinfo:
        runner.run(request_for(tmp_path, case_id="no_such_case"))
    assert "no_such_case" in str(excinfo.value)
    assert "reference" in str(excinfo.value)


def test_fake_runner_needs_a_real_directory(tmp_path):
    with pytest.raises(RunnerError):
        FakeRunner(tmp_path / "nowhere")


def test_fake_runner_lists_the_seven_study_cases(gmres_fixtures):
    assert len(FakeRunner(gmres_fixtures).available_cases()) == 7


def test_fake_runner_copies_output_files(tmp_path, gmres_fixtures):
    fixture_root = tmp_path / "fixtures"
    shutil.copytree(gmres_fixtures / "reference", fixture_root / "reference")
    (fixture_root / "reference" / "flux.csv").write_text("z,phi\n0.0,1.0\n")

    runner = FakeRunner(fixture_root)
    result = runner.run(request_for(tmp_path / "runs"))
    assert (result.directory / "flux.csv").read_text().startswith("z,phi")


def test_fake_runner_replays_a_recorded_failure(tmp_path):
    fixture_root = tmp_path / "fixtures" / "diverged"
    fixture_root.mkdir(parents=True)
    (fixture_root / "stdout.txt").write_text("00:00:01.0 WGS groups [0-0] iteration = 300\n")
    (fixture_root / "stderr.txt").write_text("RuntimeError: solver diverged\n")
    (fixture_root / "meta.json").write_text(json.dumps({"exit_code": 1, "wall_time": 12.0}))

    runner = FakeRunner(tmp_path / "fixtures")
    result = runner.run(request_for(tmp_path / "runs", case_id="diverged"))
    assert not result.ok
    assert "exit code 1" in result.failure_reason()
    assert "diverged" in result.failure_reason()


def test_fake_runner_config_distinguishes_fixture_sets(tmp_path, gmres_fixtures, default_fixtures):
    assert FakeRunner(gmres_fixtures).config() != FakeRunner(default_fixtures).config()


# --- RunResult --------------------------------------------------------------


def test_result_round_trips_through_its_directory(tmp_path, gmres_fixtures):
    runner = FakeRunner(gmres_fixtures)
    result = runner.run(request_for(tmp_path))
    reloaded = RunResult.from_directory(result.directory)
    assert reloaded.stdout == result.stdout
    assert reloaded.exit_code == result.exit_code
    assert reloaded.wall_time == pytest.approx(result.wall_time)


def test_timeout_is_distinguishable_from_a_crash(tmp_path):
    result = RunResult(
        case_id="c",
        exit_code=124,
        stdout="",
        stderr="",
        wall_time=1800.0,
        directory=tmp_path,
        runner="local",
        timed_out=True,
    )
    assert not result.ok
    assert "timed out" in result.failure_reason()


def test_recording_a_fixture_round_trips(tmp_path, gmres_fixtures):
    runner = FakeRunner(gmres_fixtures)
    original = runner.run(request_for(tmp_path / "runs"))

    target = record_fixture(original, tmp_path / "new_fixtures", provenance="test")
    assert (target / "stdout.txt").exists()
    assert json.loads((target / "meta.json").read_text())["provenance"] == "test"

    replayed = FakeRunner(tmp_path / "new_fixtures").run(request_for(tmp_path / "runs2"))
    assert replayed.stdout == original.stdout


# --- LocalMPIRunner ---------------------------------------------------------


def test_command_is_mpiexec_n_python_script(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    runner = LocalMPIRunner()
    request = RunRequest(
        case_id="c", script="", directory=tmp_path, num_procs=4, timeout_seconds=1.0
    )
    command = runner.command_for(request)
    assert command[0].endswith("mpiexec")
    assert command[1:3] == ("-n", "4")
    assert command[-1] == "script.py"


def test_extra_launcher_arguments_are_passed_through(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    runner = LocalMPIRunner(extra_launcher_args=("--oversubscribe",))
    request = RunRequest(case_id="c", script="", directory=tmp_path)
    assert "--oversubscribe" in runner.command_for(request)


def test_missing_launcher_explains_the_alternative(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    runner = LocalMPIRunner()
    with pytest.raises(RunnerError) as excinfo:
        runner.resolve_launcher()
    message = str(excinfo.value)
    # Someone hitting this has no OpenSn, so the message has to name a way to
    # get a result today as well as a way to install one.
    assert "reference solver" in message
    assert "docs/running_opensn.md" in message


def test_named_launcher_that_is_absent_is_reported(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RunnerError) as excinfo:
        LocalMPIRunner(launcher="srun").resolve_launcher()
    assert "srun" in str(excinfo.value)


def test_local_runner_really_runs_a_script(tmp_path, monkeypatch):
    """Exercises the subprocess path without MPI, by standing in for the launcher.

    ``mpiexec -n 1 python script.py`` is, for the purposes of this test, just
    ``env -- python script.py``. What is being checked is that the runner
    captures both streams, records an exit code, and persists the directory.
    """
    monkeypatch.setattr(shutil, "which", lambda name: shutil.os.fspath("/usr/bin/env"))
    runner = LocalMPIRunner(launcher="/usr/bin/env", python="python3")
    request = RunRequest(
        case_id="smoke",
        script="import sys\nprint('out')\nprint('err', file=sys.stderr)\nsys.exit(3)\n",
        directory=tmp_path / "smoke",
        num_procs=1,
        timeout_seconds=60.0,
    )
    # env ignores "-n 1" only if we drop it, so build the command explicitly.
    monkeypatch.setattr(
        LocalMPIRunner, "command_for", lambda self, req: ("/usr/bin/env", "python3", "script.py")
    )
    result = runner.run(request)
    assert result.exit_code == 3
    assert "out" in result.stdout
    assert "err" in result.stderr
    assert not result.ok
    assert (result.directory / "stdout.txt").read_text().strip() == "out"


def test_local_runner_times_out_without_crashing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        LocalMPIRunner,
        "command_for",
        lambda self, req: ("python3", "-c", "import time; time.sleep(30)"),
    )
    runner = LocalMPIRunner(launcher="/usr/bin/env", python="python3")
    request = RunRequest(
        case_id="slow", script="", directory=tmp_path / "slow", timeout_seconds=0.5
    )
    result = runner.run(request)
    assert result.timed_out
    assert result.exit_code == 124
    assert "killed after" in result.stderr


@pytest.mark.skipif(
    os.environ.get("AUTOOPENSN_ALLOW_LOCAL_RUNS") != "1",
    reason="real OpenSn runs are opt-in; set AUTOOPENSN_ALLOW_LOCAL_RUNS=1",
)
def test_real_opensn_run(tmp_path, reed_template):
    """Only runs when explicitly enabled and OpenSn is actually installed."""
    runner = LocalMPIRunner()
    runner.preflight()
    if not runner.pyopensn_available():
        pytest.skip("pyopensn is not importable")
    request = RunRequest(
        case_id="reed_default",
        script=reed_template.render(),
        directory=tmp_path / "reed",
        num_procs=1,
        timeout_seconds=1800.0,
    )
    result = runner.run(request)
    assert result.ok, result.stderr
    assert "Absorption=" in result.stdout
