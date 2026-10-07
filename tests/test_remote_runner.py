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
    assert " mpiexec -n 8 python3 script.py\n" in script


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
    launch = next(i for i, line in enumerate(lines) if "mpiexec" in line)
    start = next(i for i, line in enumerate(lines) if line.startswith("__t0="))
    end = next(i for i, line in enumerate(lines) if line.startswith("__t1="))
    loads = [i for i, line in enumerate(lines) if line.startswith("module load")]
    assert max(loads) < start < launch < end


def test_a_failed_run_keeps_its_exit_code_and_its_timing(tmp_path):
    """set -e would abort before the end time and lose both."""
    script = RemoteRunner().remote_script(request_for(tmp_path))
    lines = script.splitlines()
    assert lines.index("set +e") < next(i for i, l in enumerate(lines) if "mpiexec" in l)
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


# --- data files ---------------------------------------------------------------


def test_data_files_are_staged_before_the_clock_starts(tmp_path):
    """Copying a mesh is not simulation time."""
    request = RunRequest(
        case_id="c", script="", directory=tmp_path / "s" / "c", num_procs=1,
        data_files=("doc/source/tutorials/a/b/mesh.msh",),
    )
    lines = RemoteRunner().remote_script(request).splitlines()
    copy = next(i for i, l in enumerate(lines) if l.startswith("cp "))
    assert next(i for i, l in enumerate(lines) if l.startswith("cd ")) < copy
    assert copy < next(i for i, l in enumerate(lines) if l.startswith("__t0="))


def test_the_run_mirrors_the_source_tree(tmp_path):
    """Each file lands at its true source path and the script runs from the
    notebook's folder, so `LANL30/OpenMC/x.h5` and `../../../../../test/…`
    resolve exactly as the tutorial wrote them. The first version copied every
    file flat into one folder, which broke any reference with a directory in it."""
    request = RunRequest(
        case_id="c", script="", directory=tmp_path / "s" / "c",
        data_files=(
            "doc/source/tutorials/m/g/LANL30/OpenMC/x.h5",
            "test/assets/mesh/y.msh",
        ),
        workdir="doc/source/tutorials/m/g",
    )
    lines = RemoteRunner().remote_script(request).splitlines()
    assert 'cp -r "$HOME"/opensn/doc/source/tutorials/m/g/LANL30/OpenMC/x.h5 doc/source/tutorials/m/g/LANL30/OpenMC/x.h5' in lines
    assert 'cp -r "$HOME"/opensn/test/assets/mesh/y.msh test/assets/mesh/y.msh' in lines
    assert "mkdir -p doc/source/tutorials/m/g/LANL30/OpenMC" in lines
    # The script follows, and the run happens from the notebook's folder.
    assert "cp script.py doc/source/tutorials/m/g/script.py" in lines
    run_dir_moves = [l for l in lines if l.startswith("cd ")]
    assert run_dir_moves[-1] == "cd doc/source/tutorials/m/g"
    assert lines.index(run_dir_moves[-1]) < next(i for i, l in enumerate(lines) if "mpiexec" in l)


def test_a_workdir_cannot_escape_the_run_directory(tmp_path):
    request = RunRequest(case_id="c", script="", directory=tmp_path / "c", workdir="../elsewhere")
    with pytest.raises(RunnerError, match="inside the run directory"):
        RemoteRunner().remote_script(request)


@pytest.mark.parametrize("bad", ["/etc/passwd", "doc/../../.ssh/id_rsa"])
def test_a_data_file_cannot_escape_the_source_tree(tmp_path, bad):
    request = RunRequest(case_id="c", script="", directory=tmp_path / "c", data_files=(bad,))
    with pytest.raises(RunnerError, match="inside the OpenSn source"):
        RemoteRunner().remote_script(request)


def test_templates_declare_their_data_files_as_source_paths():
    from autoopensn.study import data_files_for
    from autoopensn.templates import load_template

    files = data_files_for(load_template("glovebox_adj"))
    folder = "doc/source/tutorials/applications/detector_response/glovebox_adjoint"
    # The WIMS69 cross-section folder as well as the mesh: the script builds
    # each cross-section path at run time from the folder name.
    assert files == (f"{folder}/WIMS69", f"{folder}/glovebox.msh")
    assert data_files_for(load_template("reed_1d")) == ()


def test_a_helper_module_on_sys_path_is_found_and_staged(tmp_path):
    """The sLDFE tutorial imports a plotter from ../../../../../tools/…"""
    from autoopensn.tutorials.catalog import _data_files

    notebook_dir = tmp_path / "doc" / "source" / "tutorials" / "a" / "b"
    notebook_dir.mkdir(parents=True)
    tools = tmp_path / "tools" / "plotting"
    tools.mkdir(parents=True)
    (tools / "plotter.py").write_text("def plot(): pass\n")
    script = 'import sys\nsys.path.append("../../../../../tools/plotting")\nfrom plotter import plot\n'
    assert _data_files(notebook_dir, script) == ["../../../../../tools/plotting/plotter.py"]


def test_a_helper_that_is_not_there_is_not_recorded(tmp_path):
    """Recording a file that does not exist would make every run fail to stage."""
    from autoopensn.tutorials.catalog import _data_files

    script = 'import sys\nsys.path.append("../nowhere")\nfrom plotter import plot\n'
    assert _data_files(tmp_path, script) == []


def test_a_relative_data_file_is_normalised_into_the_source_tree():
    from autoopensn.study import data_files_for
    from autoopensn.templates import load_template

    assert data_files_for(load_template("sldfe")) == (
        "tools/ang_quad_plotting/plot_sldfe_quadrature.py",
    )



# --- a login node that refuses bursts of connections --------------------------


class _Completed:
    def __init__(self, returncode, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def test_an_ssh_refusal_is_recognised():
    from autoopensn.runner.remote import _connection_failed

    refused = "ssh: connect to host 128.194.17.172 port 22: Connection refused"
    assert _connection_failed(255, "", refused)
    # Something ran and printed: never a connection failure, never retried.
    assert not _connection_failed(255, "OpenSn version 1.0.1\n", refused)
    # A program that exits 255 for its own reasons is not ssh failing.
    assert not _connection_failed(255, "", "Traceback: ValueError")
    assert not _connection_failed(1, "", refused)


def test_a_refused_connection_is_retried_until_it_succeeds(tmp_path, monkeypatch):
    """Running every scenario once had nine refused in a row; waiting clears it."""
    import autoopensn.runner.remote as remote

    replies = iter([
        _Completed(255, "", "ssh: connect to host x port 22: Connection refused"),
        _Completed(255, "", "Connection closed by UNKNOWN port 65535"),
        _Completed(0, "fine\n", ""),
    ])
    calls = []
    monkeypatch.setattr(remote, "_run_process", lambda *a, **k: calls.append(1) or next(replies))
    result = remote._guarded(["ssh", "class01", "true"], timeout=5)
    assert result.returncode == 0
    assert len(calls) == 3


def test_retrying_gives_up_and_reports_the_failure(monkeypatch):
    import autoopensn.runner.remote as remote

    refused = _Completed(255, "", "Connection refused")
    calls = []
    monkeypatch.setattr(remote, "_run_process", lambda *a, **k: calls.append(1) or refused)
    result = remote._guarded(["ssh", "class01", "true"], timeout=5)
    assert result.returncode == 255
    assert len(calls) == len(remote.RETRY_DELAYS) + 1


def test_a_simulation_that_failed_is_not_retried(monkeypatch):
    """Retrying a run that executed would run it twice on a shared machine."""
    import autoopensn.runner.remote as remote

    crashed = _Completed(1, "OpenSn version 1.0.1\n", "RuntimeError: bad mesh")
    calls = []
    monkeypatch.setattr(remote, "_run_process", lambda *a, **k: calls.append(1) or crashed)
    remote._guarded(["ssh", "class01", "run"], timeout=5)
    assert len(calls) == 1


# --- a run that hangs must not outlive its time limit -------------------------


def test_the_node_enforces_the_time_limit(tmp_path):
    """The glovebox tutorial hung after an output error and kept eight ranks
    spinning on a shared node long after this side had reported a timeout,
    because only this side was keeping time. The node now keeps it too."""
    request = RunRequest(case_id="c", script="", directory=tmp_path / "c", num_procs=8,
                         timeout_seconds=600)
    launch = next(l for l in RemoteRunner().remote_script(request).splitlines() if "mpiexec" in l)
    assert launch.startswith("timeout --kill-after=30 600 mpiexec -n 8 ")


def test_this_side_waits_longer_than_the_node(tmp_path, monkeypatch):
    """So that it hears the node report the timeout instead of giving up first."""
    import autoopensn.runner.remote as remote

    seen = {}

    def run(command, **kwargs):
        seen["timeout"] = kwargs["timeout"]
        return _Completed(0, "ok\n", "")

    monkeypatch.setattr(remote, "_run_process", run)
    monkeypatch.setattr(remote.RemoteRunner, "_send", lambda self, request: None)
    request = RunRequest(case_id="c", script="print(1)\n", directory=tmp_path / "c",
                         timeout_seconds=600)
    RemoteRunner(fetch_outputs=False).run(request)
    assert seen["timeout"] > 600 + remote.KILL_GRACE


@pytest.mark.parametrize("code", [124, 137])
def test_a_run_the_node_stopped_is_recorded_as_a_timeout(tmp_path, monkeypatch, code):
    import autoopensn.runner.remote as remote

    monkeypatch.setattr(remote, "_run_process", lambda *a, **k: _Completed(code, "partial\n", ""))
    monkeypatch.setattr(remote.RemoteRunner, "_send", lambda self, request: None)
    request = RunRequest(case_id="c", script="print(1)\n", directory=tmp_path / "c",
                         timeout_seconds=60)
    result = RemoteRunner(fetch_outputs=False).run(request)
    assert result.timed_out
    assert "stopped on class01" in result.stderr
