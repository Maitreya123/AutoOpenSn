"""The SLURM runner, against a fake cluster.

Nothing here reaches Orchard. The process entry point the remote runner uses is
replaced with a script of replies — sbatch's job id, squeue's states, sacct's
record, the job's output files — so the whole submit, poll and collect flow
runs, and the command text can be read back.
"""

from __future__ import annotations

import pytest

import autoopensn.runner.remote as remote
from autoopensn.runner import RunRequest, RunnerError, SlurmRunner
from autoopensn.runner.slurm import parse_sacct
from autoopensn.runner.select import SCHEDULER_ENV, build_runner, choose_runner


def request_for(tmp_path, case_id="baseline", num_procs=4, timeout=600.0):
    return RunRequest(
        case_id=case_id, script="print('hello')\n", directory=tmp_path / "study" / case_id,
        template="reed_1d", num_procs=num_procs, timeout_seconds=timeout,
    )


class Reply:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


class FakeCluster:
    """Answers each command by what it asks for, and remembers them all."""

    def __init__(self, *, queue=("PENDING", "RUNNING"), sacct=None, out="", err=""):
        self.queue = list(queue)
        self.sacct = sacct if sacct is not None else (
            "62500|COMPLETED|0:0|2026-11-11T12:00:00|2026-11-11T12:00:03|2026-11-11T12:00:05|nuen-orch-class001\n"
        )
        self.out, self.err = out, err
        self.commands = []

    def __call__(self, command, **kwargs):
        text = " ".join(command)
        self.commands.append(text)
        if "sbatch --parsable" in text:
            return Reply(stdout="62500\n")
        if "squeue" in text:
            return Reply(stdout=(self.queue.pop(0) + "\n") if self.queue else "")
        if "sacct" in text:
            return Reply(stdout=self.sacct)
        if "slurm-62500.out" in text:
            return Reply(stdout=self.out)
        if "slurm-62500.err" in text:
            return Reply(stdout=self.err)
        return Reply()


@pytest.fixture
def cluster(monkeypatch):
    fake = FakeCluster(
        out="OpenSn version 1.0.1\nAbsorption=1.006178e+02\n",
        err="AUTOOPENSN_NODE_TIMING 100.000 100.850\n",
    )
    monkeypatch.setattr(remote, "_run_process", fake)
    return fake


# --- the batch script -------------------------------------------------------------


def test_the_job_carries_the_partition_ranks_and_a_time_limit(tmp_path):
    script = SlurmRunner().job_script(request_for(tmp_path, num_procs=8, timeout=600))
    lines = script.splitlines()
    assert lines[0] == "#!/bin/bash -l", "module is a function only a login shell defines"
    assert "#SBATCH --partition=orchard-class" in lines
    assert "#SBATCH --ntasks=8" in lines
    # 600 s of run, 30 s of kill grace, 60 s of margin, rounded up to minutes.
    assert "#SBATCH --time=12" in lines


def test_the_job_runs_the_direct_runners_own_program(tmp_path):
    """Module loads, staging and on-node timing are not reimplemented."""
    request = request_for(tmp_path)
    runner = SlurmRunner()
    script = runner.job_script(request)
    assert runner.remote_script(request) in script
    assert script.index('cd "$HOME"') < script.index("set -eu"), \
        "the body's paths are relative to home; a job starts where it was submitted"


def test_no_account_header_unless_one_is_given(tmp_path):
    assert "--account" not in SlurmRunner().job_script(request_for(tmp_path))
    assert "#SBATCH --account=orchard-class" in SlurmRunner(account="orchard-class").job_script(request_for(tmp_path))


def test_a_partition_name_cannot_smuggle_shell(tmp_path):
    with pytest.raises(RunnerError):
        SlurmRunner(partition="orchard-class; rm -rf ~")


def test_submission_happens_on_the_login_node_in_the_run_directory(tmp_path):
    command = SlurmRunner().submit_command(request_for(tmp_path, case_id="c1"))
    assert "orchard" in command
    assert command[-1] == "cd autoopensn-runs/study/c1 && sbatch --parsable job.sh"


# --- reading sacct ------------------------------------------------------------------


def test_sacct_is_read_into_state_exit_code_and_queue_wait():
    record = parse_sacct(
        "62435|COMPLETED|0:0|2026-11-11T11:50:50|2026-11-11T11:50:52|2026-11-11T11:50:58|nuen-orch-class001\n"
        "62435.batch|COMPLETED|0:0|2026-11-11T11:50:52|2026-11-11T11:50:52|2026-11-11T11:50:58|nuen-orch-class001\n"
    )
    assert record["slurm_state"] == "COMPLETED"
    assert record["exit_code"] == 0
    assert record["queue_wait"] == 2.0
    assert record["node"] == "nuen-orch-class001"


def test_a_cancelled_state_keeps_only_its_first_word():
    record = parse_sacct("1|CANCELLED by 536009728|0:15|2026-11-11T11:00:00|Unknown|Unknown|None assigned\n")
    assert record["slurm_state"] == "CANCELLED"
    assert record["started_at"] is None and record["queue_wait"] is None


def test_nothing_readable_is_none():
    assert parse_sacct("") is None


# --- the whole flow -------------------------------------------------------------------


def test_a_job_is_submitted_waited_for_and_collected(tmp_path, cluster):
    result = SlurmRunner(fetch_outputs=False).run(request_for(tmp_path))
    assert result.ok
    assert "Absorption=1.006178e+02" in result.stdout
    assert result.wall_time == pytest.approx(0.85), "the simulation's time, not the queue's"
    assert result.metadata["slurm_job_id"] == "62500"
    assert result.metadata["queue_wait"] == 3.0
    assert result.metadata["node"] == "nuen-orch-class001"
    assert result.metadata["partition"] == "orchard-class"
    assert "AUTOOPENSN_NODE_TIMING" not in result.stderr
    # It polled until the job left the queue.
    assert sum("squeue" in c for c in cluster.commands) == 3


def test_a_failed_job_is_a_failed_row(tmp_path, monkeypatch):
    fake = FakeCluster(
        queue=(), err="Traceback: ModuleNotFoundError\n",
        sacct="62500|FAILED|1:0|2026-11-11T12:00:00|2026-11-11T12:00:01|2026-11-11T12:00:02|nuen-orch-class001\n",
    )
    monkeypatch.setattr(remote, "_run_process", fake)
    result = SlurmRunner(fetch_outputs=False).run(request_for(tmp_path))
    assert not result.ok and result.exit_code == 1


def test_a_job_slurm_cancelled_is_not_reported_as_success(tmp_path, monkeypatch):
    """A scheduler cancellation can carry exit code 0; the row must still fail."""
    fake = FakeCluster(
        queue=(), sacct="62500|CANCELLED by 1|0:0|2026-11-11T12:00:00|2026-11-11T12:00:01|2026-11-11T12:00:02|n\n",
    )
    monkeypatch.setattr(remote, "_run_process", fake)
    result = SlurmRunner(fetch_outputs=False).run(request_for(tmp_path))
    assert not result.ok
    assert "ended CANCELLED" in result.stderr


def test_a_job_that_hit_its_limit_is_a_timeout(tmp_path, monkeypatch):
    fake = FakeCluster(
        queue=(), sacct="62500|TIMEOUT|0:0|2026-11-11T12:00:00|2026-11-11T12:00:01|2026-11-11T12:12:01|n\n",
    )
    monkeypatch.setattr(remote, "_run_process", fake)
    result = SlurmRunner(fetch_outputs=False).run(request_for(tmp_path))
    assert result.timed_out and result.exit_code == 124


def test_a_job_that_never_starts_is_cancelled_not_abandoned(tmp_path, monkeypatch):
    """Giving up waiting must not leave the job in someone else's queue."""
    fake = FakeCluster(queue=["PENDING"] * 50, sacct="")
    monkeypatch.setattr(remote, "_run_process", fake)
    runner = SlurmRunner(fetch_outputs=False, max_queue_wait=0)
    result = runner.run(request_for(tmp_path, timeout=-200))
    assert any("scancel 62500" in c for c in fake.commands)
    assert result.timed_out


def test_an_interruption_while_waiting_cancels_the_job(tmp_path, monkeypatch):
    fake = FakeCluster(queue=["RUNNING"] * 10)
    monkeypatch.setattr(remote, "_run_process", fake)

    def interrupted(seconds):
        raise KeyboardInterrupt

    monkeypatch.setattr(remote, "_sleep", interrupted)
    with pytest.raises(KeyboardInterrupt):
        SlurmRunner(fetch_outputs=False).run(request_for(tmp_path))
    assert any("scancel 62500" in c for c in fake.commands)


def test_sbatch_refusing_the_job_is_a_runner_error(tmp_path, monkeypatch):
    def refuse(command, **kwargs):
        if "sbatch" in " ".join(command):
            return Reply(1, "", "sbatch: error: Invalid partition name specified")
        return Reply()

    monkeypatch.setattr(remote, "_run_process", refuse)
    with pytest.raises(RunnerError, match="Invalid partition"):
        SlurmRunner(fetch_outputs=False).run(request_for(tmp_path))


# --- choosing it ------------------------------------------------------------------------


def test_slurm_and_ssh_results_do_not_share_cache_entries():
    from autoopensn.runner import RemoteRunner

    assert SlurmRunner().config() != RemoteRunner().config()
    assert SlurmRunner().config()["partition"] == "orchard-class"


def test_auto_prefers_slurm_when_the_scheduler_answers(monkeypatch):
    monkeypatch.setattr(remote, "_run_process", lambda c, **k: Reply(stdout="/usr/bin/sbatch\nup\n"))
    runner, note = choose_runner()
    assert isinstance(runner, SlurmRunner)
    assert "SLURM" in note


def test_auto_skips_slurm_when_told_to(monkeypatch):
    monkeypatch.setenv(SCHEDULER_ENV, "ssh")
    monkeypatch.setattr(remote, "_run_process", lambda c, **k: Reply(stdout="/usr/bin/sbatch\nup\n"))
    runner, _ = choose_runner()
    assert not isinstance(runner, SlurmRunner)


def test_a_down_partition_is_refused_by_preflight(monkeypatch):
    monkeypatch.setattr(remote, "_run_process", lambda c, **k: Reply(stdout="/usr/bin/sbatch\ndown\n"))
    with pytest.raises(RunnerError, match="not up"):
        SlurmRunner().preflight()


def test_slurm_is_a_named_runner(monkeypatch):
    monkeypatch.setattr(remote, "_run_process", lambda c, **k: Reply(stdout="/usr/bin/sbatch\nup\n"))
    runner, note = build_runner("slurm")
    assert isinstance(runner, SlurmRunner) and "SLURM" in note
