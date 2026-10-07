"""Tests for the cluster runner.

None of these connect to anything. That is the whole design of the module: the
decision about what will execute on a shared compute node is made by pure
functions, so it can be read back and asserted here, on a laptop, with no
cluster account and no network.

The failure this is guarding against is not an SSH error, which is loud and
self-explaining. It is running a subtly wrong command on someone else's
machine: the wrong process count, a module that silently did not load, a
directory built by pasting strings together. Those are only catchable by
looking at the command.
"""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest

from autoopensn.runner import RemoteRunner, RunRequest, RunnerError
from autoopensn.runner.remote import DEFAULT_MODULES, DEFAULT_MODULE_PATH


def request_for(tmp_path: Path, case_id: str = "baseline", num_procs: int = 4) -> RunRequest:
    return RunRequest(
        case_id=case_id,
        script="print('hello')\n",
        directory=tmp_path / "study" / case_id,
        template="reed_1d",
        num_procs=num_procs,
        timeout_seconds=60.0,
        parameters={"l_abs_tol": 1.0e-8},
    )


# --- what would run ---------------------------------------------------------


def test_remote_directory_mirrors_the_local_one(tmp_path):
    """A run directory here and its counterpart there share a name."""
    runner = RemoteRunner(remote_root="autoopensn-runs")
    request = request_for(tmp_path, "l_abs_tol-1.0e-8")
    assert runner.remote_dir(request) == "autoopensn-runs/study/l_abs_tol-1.0e-8"


def test_no_remote_path_relies_on_tilde_expansion(tmp_path):
    """A quoted tilde is never expanded, and every path here is quoted.

    With ``~/autoopensn-runs`` as the root, ``mkdir -p '~/…'`` made a directory
    literally named ``~``, scp expanded its own tilde and copied the script to
    the real home, and the run then looked for it in the wrong one. Found by
    reading the generated command before the first real run, not by a failure.
    """
    request = request_for(tmp_path)
    for root in ("~/autoopensn-runs", "autoopensn-runs"):
        runner = RemoteRunner(remote_root=root)
        assert "~" not in runner.remote_dir(request)
        assert "~" not in runner.remote_script(request)


def test_module_preamble_prepends_rather_than_replaces():
    """The site's own modulefiles must survive; losing them is confusing."""
    preamble = RemoteRunner().module_preamble()
    # ${MODULEPATH:-} rather than $MODULEPATH: the script runs under set -u,
    # where an unset variable is an error rather than an empty string.
    assert f"export MODULEPATH={DEFAULT_MODULE_PATH}:${{MODULEPATH:-}}" in preamble
    for module in DEFAULT_MODULES:
        assert f"module load {module}" in preamble


def test_module_preamble_is_empty_when_nothing_to_load():
    """A node with OpenSn already on PATH should not be handed module commands."""
    runner = RemoteRunner(modules=(), module_path=None, pythonpath=None)
    assert runner.module_preamble() == ""


def test_remote_script_stops_on_a_failed_module_load(tmp_path):
    """Without `set -e` a missing module falls through to the system python."""
    script = RemoteRunner().remote_script(request_for(tmp_path))
    assert script.splitlines()[0] == "set -eu"


def test_remote_script_runs_the_right_command(tmp_path):
    runner = RemoteRunner()
    script = runner.remote_script(request_for(tmp_path, num_procs=8))
    assert "module load opensn/gcc/15" in script
    assert "module load python3/3.12.3" in script
    # Python is linked against MPI and cannot start before it is loaded.
    assert script.index("opensn/gcc/15") < script.index("python3/3.12.3")
    assert 'PYTHONPATH="$HOME/opensn/build:' in script
    assert "cd " in script
    assert "\nmpiexec -n 8 python3 script.py\n" in script


def test_process_count_comes_from_the_request(tmp_path):
    """A sweep point asking for 16 ranks must not silently get 1."""
    runner = RemoteRunner()
    for count in (1, 4, 32):
        script = runner.remote_script(request_for(tmp_path, num_procs=count))
        assert f"-n {count} " in script


def test_command_uses_a_login_shell(tmp_path):
    """`module` is a shell function from the site profile; -l or it is missing."""
    runner = RemoteRunner()
    request = request_for(tmp_path)
    command = runner.command_for(request)
    assert command[0] == "ssh"
    assert command[-1].startswith("bash -l -c ")


def test_the_remote_shell_receives_the_script_intact(tmp_path):
    """ssh joins its arguments with spaces and the remote shell parses them again.

    Passed as four arguments, the remote side read `bash -l -c set -eu …`: it
    ran `set` alone, which dumped every shell variable into the saved output,
    and ran the rest outside `set -eu`. This parses the remote command the way
    the remote shell will and checks bash gets the whole script as one string.
    """
    runner = RemoteRunner()
    request = request_for(tmp_path)
    remote = runner.command_for(request)[-1]
    assert shlex.split(remote) == ["bash", "-l", "-c", runner.remote_script(request)]


def test_command_never_prompts_for_a_password(tmp_path):
    """A study of seven cases must not stop on case one waiting for a human."""
    command = RemoteRunner().command_for(request_for(tmp_path))
    assert "BatchMode=yes" in command


def test_paths_with_shell_metacharacters_are_quoted(tmp_path):
    """A case id is derived from parameter values and reaches a remote shell."""
    runner = RemoteRunner(remote_root="runs with spaces")
    request = request_for(tmp_path, "weird; rm -rf me")
    script = runner.remote_script(request)

    cd_line = next(line for line in script.splitlines() if line.startswith("cd "))
    words = shlex.split(cd_line)
    # One word, not four. If the quoting were wrong the shell would read `rm`
    # as a second command rather than as part of a directory name.
    assert len(words) == 2, f"cd took {len(words) - 1} arguments: {words[1:]}"
    assert words[1] == "runs with spaces/study/weird; rm -rf me"


# --- identity ---------------------------------------------------------------


def test_config_distinguishes_hosts_and_toolchains():
    """A gcc build and a clang build are not interchangeable evidence."""
    gcc = RemoteRunner(modules=("opensn/gcc/15",)).config()
    clang = RemoteRunner(modules=("opensn/clang",)).config()
    other_host = RemoteRunner(host="class02").config()

    assert gcc != clang
    assert gcc != other_host
    assert gcc["runner"] == "remote"


def test_config_is_not_the_local_runner():
    """The cache must never serve a cluster result as a local one, or the reverse."""
    from autoopensn.runner import LocalMPIRunner

    assert RemoteRunner().config() != LocalMPIRunner().config()


# --- connecting -------------------------------------------------------------


def test_preflight_explains_a_missing_ssh(monkeypatch):
    monkeypatch.setattr("autoopensn.runner.remote.shutil.which", lambda name: None)
    with pytest.raises(RunnerError, match="not on PATH"):
        RemoteRunner().preflight()


def test_preflight_reports_an_unreachable_host(monkeypatch):
    """The message must separate "cannot connect" from "OpenSn did not load"."""
    monkeypatch.setattr("autoopensn.runner.remote.shutil.which", lambda name: f"/usr/bin/{name}")

    class Failed:
        returncode = 255
        stdout = ""
        stderr = "ssh: connect to host class01 port 22: Operation timed out"

    monkeypatch.setattr("autoopensn.runner.remote._run_process", lambda *a, **k: Failed())
    with pytest.raises(RunnerError, match="cannot reach"):
        RemoteRunner().preflight()


def test_preflight_reports_a_missing_module(monkeypatch):
    monkeypatch.setattr("autoopensn.runner.remote.shutil.which", lambda name: f"/usr/bin/{name}")

    calls = {"n": 0}

    class Result:
        def __init__(self, returncode, stderr=""):
            self.returncode = returncode
            self.stdout = ""
            self.stderr = stderr

    def fake_run(*args, **kwargs):
        calls["n"] += 1
        # First call is the reachability probe, second loads the module.
        if calls["n"] == 1:
            return Result(0)
        return Result(1, "Unable to locate a modulefile for 'opensn/gcc/15'")

    monkeypatch.setattr("autoopensn.runner.remote._run_process", fake_run)
    with pytest.raises(RunnerError, match="environment did not load"):
        RemoteRunner().preflight()


def test_a_hanging_ssh_becomes_a_runner_error_not_a_crash(monkeypatch):
    """A dropped-packet firewall must not escape as TimeoutExpired.

    Runner promises that a failure to *attempt* a run is a RunnerError. When
    this leaked, the automatic runner choice crashed the whole request on an
    unreachable cluster instead of falling back to the solver that works —
    which is the one case the fallback exists for.
    """
    import subprocess

    monkeypatch.setattr("autoopensn.runner.remote.shutil.which", lambda name: f"/usr/bin/{name}")

    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["ssh"], timeout=kwargs.get("timeout", 4))

    monkeypatch.setattr("autoopensn.runner.remote._run_process", hang)
    with pytest.raises(RunnerError, match="did not respond"):
        RemoteRunner(connect_timeout=1).preflight()


def test_output_retrieval_failing_does_not_fail_the_run(tmp_path, monkeypatch):
    """stdout is already in hand; a VTK file that will not copy is not fatal."""
    import subprocess

    monkeypatch.setattr("autoopensn.runner.remote.shutil.which", lambda name: f"/usr/bin/{name}")

    def hang(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["scp"], timeout=1)

    monkeypatch.setattr("autoopensn.runner.remote._run_process", hang)
    runner = RemoteRunner(connect_timeout=1)
    request = request_for(tmp_path)
    request.write_inputs()
    runner._fetch(request)  # must not raise



def test_the_suite_cannot_reach_a_cluster():
    """Guards the guard: with no patch of its own, a test gets no connection.

    If this ever fails, every test that leaves the runner on `auto` is
    submitting real jobs to a shared compute node.
    """
    with pytest.raises(RunnerError):
        RemoteRunner(connect_timeout=1).preflight()



# --- timing the simulation, not the network -----------------------------------


def test_the_run_is_timed_on_the_node_around_mpiexec_alone(tmp_path):
    """Timed from here, the clock included the SSH round trip and module loads:
    on class01, 7.1s around a 0.85s solve."""
    script = RemoteRunner().remote_script(request_for(tmp_path))
    lines = script.splitlines()
    launch = next(i for i, line in enumerate(lines) if line.startswith("mpiexec"))
    start = next(i for i, line in enumerate(lines) if line.startswith("__t0="))
    end = next(i for i, line in enumerate(lines) if line.startswith("__t1="))
    loads = [i for i, line in enumerate(lines) if line.startswith("module load")]
    assert max(loads) < start < launch < end


def test_a_failed_run_keeps_its_exit_code_and_its_timing(tmp_path):
    """set -e would abort before the end time and lose both."""
    script = RemoteRunner().remote_script(request_for(tmp_path))
    lines = script.splitlines()
    assert lines.index("set +e") < next(i for i, l in enumerate(lines) if l.startswith("mpiexec"))
    assert lines[-1] == "exit $__rc"


def test_node_timing_is_read_and_removed_from_stderr():
    from autoopensn.runner.remote import TIMING_MARKER, split_node_timing

    stderr = f"a warning\n{TIMING_MARKER} 1000.250 1001.101\n"
    kept, seconds = split_node_timing(stderr)
    assert seconds == pytest.approx(0.851)
    assert TIMING_MARKER not in kept
    assert kept == "a warning\n"


def test_missing_node_timing_is_none_not_zero():
    """No marker means the run never got that far; zero would be a lie."""
    from autoopensn.runner.remote import split_node_timing

    assert split_node_timing("crashed before mpiexec\n") == ("crashed before mpiexec\n", None)



def test_results_timed_differently_do_not_share_cache_entries():
    """Round-trip and on-node timings fill the same column with different things."""
    assert RemoteRunner().config()["wall_time"] == "on-node"
