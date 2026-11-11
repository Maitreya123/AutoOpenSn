"""Running a case as a SLURM batch job on the cluster.

The direct runner (``RemoteRunner``) starts each case by SSHing to the compute
node and running it there. That works, and on a shared machine it is also
impolite: nothing stops two people's studies landing on the same cores at
once. Orchard runs SLURM, and its ``orchard-class`` partition is class01, so
each case can instead be submitted as a job and let the scheduler place it.

What changes and what does not:

- **The work itself does not.** The job's body is the direct runner's own shell
  program — module loads in the right order, the run laid out as a mirror of
  the OpenSn source tree, the simulation timed on the node around ``mpiexec``
  alone. Only how it is started differs.
- **It is submitted from the login node.** ``sbatch`` lives on Orchard's front
  end, and home directories are shared with class01, so the files sent there
  are the files the job sees.
- **It is waited for by polling.** ``squeue`` until the job leaves the queue,
  then ``sacct`` for how it ended: its state, exit code, when it was submitted
  and started, and which node ran it. The time spent queued goes into the run's
  metadata, beside the simulation's own time; the two are different costs and
  are never added together.
- **A stuck run cannot outlive its limit.** The job carries a SLURM time limit
  as well as the node-side ``timeout``, so even if this side vanishes the
  scheduler ends it. If this side gives up waiting, it cancels the job.

The submission, the polling and the parsing of ``sacct`` are each separate, so
the command text and the parsing can be checked without a cluster.
"""

from __future__ import annotations

import math
import re
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from autoopensn.runner import remote as _remote
from autoopensn.runner.base import RunRequest, RunResult, RunnerError
from autoopensn.runner.remote import (
    KILL_GRACE,
    RemoteRunner,
    _guarded,
    split_node_timing,
)

DEFAULT_LOGIN_HOST = "orchard"
"""The ssh alias of the front end, where ``sbatch`` lives."""

DEFAULT_PARTITION = "orchard-class"
"""class01, as a SLURM partition. The default ``orchard`` partition offered
this account a start date forty years out when it was checked."""

SCHEDULER_MARGIN = 60
"""Seconds of SLURM time limit beyond the run's own timeout and kill grace, so
the scheduler only steps in if the node-side ``timeout`` could not."""

POLL_DELAYS = (2.0, 2.0, 3.0, 5.0, 5.0, 10.0)
"""Seconds between checks on a submitted job; the last repeats. Short at first,
because most jobs here start at once and finish in seconds."""

MAX_QUEUE_WAIT = 6 * 3600
"""How long to wait for a job to start before giving up and cancelling it."""

ENDED_BADLY = {
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "PREEMPTED", "BOOT_FAIL", "DEADLINE",
}

_JOB_ID = re.compile(r"^\s*(?P<id>\d+)")


def _seconds_between(start: str, end: str) -> Optional[float]:
    """Seconds from one sacct timestamp to another, or None if either is unset."""
    try:
        return (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    except (TypeError, ValueError):
        return None


def parse_sacct(text: str) -> Optional[dict[str, Any]]:
    """The allocation's line from ``sacct -X -n -P``, as a dict, or None.

    Fields, in order: JobID, State, ExitCode, Submit, Start, End, NodeList.
    ``State`` can carry a reason ("CANCELLED by 536009728"), so only its first
    word is kept; ``ExitCode`` is "code:signal".
    """
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 7 or not parts[0] or "." in parts[0]:
            continue
        job_id, state, exit_code, submit, start, end, nodes = parts[:7]
        try:
            code, signal = (int(x) for x in exit_code.split(":"))
        except ValueError:
            code, signal = None, None
        return {
            "slurm_job_id": job_id,
            "slurm_state": state.split()[0] if state else None,
            "exit_code": code,
            "signal": signal,
            "submitted_at": submit or None,
            "started_at": start if start and start != "Unknown" else None,
            "ended_at": end if end and end != "Unknown" else None,
            "queue_wait": _seconds_between(submit, start),
            "node": nodes or None,
        }
    return None


class SlurmRunner(RemoteRunner):
    """Submits each case as a SLURM job, and waits for it."""

    name = "slurm"

    def __init__(
        self,
        *,
        host: str = DEFAULT_LOGIN_HOST,
        partition: str = DEFAULT_PARTITION,
        account: Optional[str] = None,
        max_queue_wait: float = MAX_QUEUE_WAIT,
        poll_delays: Sequence[float] = POLL_DELAYS,
        **kwargs: Any,
    ):
        super().__init__(host=host, **kwargs)
        for label, value in (("partition", partition), ("account", account)):
            if value is not None and not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
                raise RunnerError(f"{label} contains characters SLURM would not accept: {value!r}")
        self.partition = partition
        self.account = account
        self.max_queue_wait = max_queue_wait
        self.poll_delays = tuple(poll_delays) or POLL_DELAYS

    # --- pure: what would be submitted -------------------------------------

    def time_limit_minutes(self, request: RunRequest) -> int:
        seconds = request.timeout_seconds + KILL_GRACE + SCHEDULER_MARGIN
        return max(1, math.ceil(seconds / 60))

    def job_name(self, request: RunRequest) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.case_id)[:40]
        return f"aos-{cleaned or 'case'}"

    def job_script(self, request: RunRequest) -> str:
        """The batch script: SLURM's headers, then the direct runner's program.

        ``#!/bin/bash -l`` for the same reason the direct runner uses a login
        shell: ``module`` is a function the site's profile defines. The body
        starts from ``$HOME`` because its paths are relative to the home
        directory, while a batch job starts in the directory it was submitted
        from.
        """
        headers = [
            "#!/bin/bash -l",
            f"#SBATCH --job-name={self.job_name(request)}",
            f"#SBATCH --partition={self.partition}",
            *([f"#SBATCH --account={self.account}"] if self.account else []),
            f"#SBATCH --ntasks={request.num_procs}",
            f"#SBATCH --time={self.time_limit_minutes(request)}",
            "#SBATCH --output=slurm-%j.out",
            "#SBATCH --error=slurm-%j.err",
        ]
        return "\n".join(headers) + '\ncd "$HOME"\n' + self.remote_script(request) + "\n"

    def submit_command(self, request: RunRequest) -> tuple[str, ...]:
        directory = shlex.quote(self.remote_dir(request))
        return self.ssh_command(f"cd {directory} && sbatch --parsable job.sh")

    # --- impure: talking to SLURM --------------------------------------------

    def preflight(self) -> None:
        """Reach the login node, find sbatch, and check the partition takes jobs.

        This does not load the OpenSn modules: those exist on class01, not on
        the login node, and checking them would mean running a job. The direct
        runner's preflight and the setup script check them; a job whose modules
        do not load fails with that message in its error log.
        """
        probe = _guarded(self.ssh_command("command -v sbatch && sinfo -h -p "
                                          + shlex.quote(self.partition) + " -o %a"),
                         self.connect_timeout + 10)
        if probe.returncode != 0:
            detail = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else "no detail"
            raise RunnerError(
                f"cannot reach SLURM on {self.host!r}: {detail}. Check the VPN, "
                f"and that `ssh {self.host} sinfo` works by hand."
            )
        lines = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        if not lines or "sbatch" not in lines[0]:
            raise RunnerError(f"{self.host!r} has no sbatch on its PATH")
        if len(lines) < 2 or lines[-1] != "up":
            raise RunnerError(
                f"SLURM partition {self.partition!r} is not up on {self.host!r} "
                f"(sinfo says {lines[-1] if len(lines) > 1 else 'nothing'})"
            )

    def _send(self, request: RunRequest) -> None:
        """Send the script, as the direct runner does, and the batch script too."""
        super()._send(request)
        job = Path(request.directory) / "job.sh"
        job.write_text(self.job_script(request))
        copied = _guarded(
            (
                self.scp, "-o", "BatchMode=yes", "-o", f"ConnectTimeout={self.connect_timeout}",
                *self.ssh_options, str(job), f"{self.host}:{self.remote_dir(request)}/job.sh",
            ),
            self.connect_timeout + 120,
        )
        if copied.returncode != 0:
            raise RunnerError(f"could not copy the batch script to {self.host}: {copied.stderr.strip()[:200]}")

    def _submit(self, request: RunRequest) -> str:
        submitted = _guarded(self.submit_command(request), self.connect_timeout + 60)
        found = _JOB_ID.match(submitted.stdout or "")
        if submitted.returncode != 0 or not found:
            raise RunnerError(
                f"sbatch refused the job: {(submitted.stderr or submitted.stdout).strip()[:300]}"
            )
        return found.group("id")

    def _in_queue(self, job_id: str) -> Optional[str]:
        """The job's state while SLURM still lists it, or None once it has left."""
        checked = _guarded(self.ssh_command(f"squeue -h -j {job_id} -o %T"), self.connect_timeout + 30)
        if checked.returncode != 0:
            # squeue says "Invalid job id" for a job that has already finished
            # and been purged: gone, not an error.
            if "Invalid job id" in (checked.stderr or ""):
                return None
            raise RunnerError(f"squeue failed: {checked.stderr.strip()[:200]}")
        state = checked.stdout.strip()
        return state or None

    def _accounting(self, job_id: str) -> Optional[dict[str, Any]]:
        """How the job ended, from sacct. Retried briefly: it lags squeue."""
        command = self.ssh_command(
            f"sacct -j {job_id} -X -n -P -o JobID,State,ExitCode,Submit,Start,End,NodeList"
        )
        for delay in (0.0, 2.0, 5.0):
            if delay:
                _remote._sleep(delay)
            record = parse_sacct(_guarded(command, self.connect_timeout + 30).stdout or "")
            if record and record["slurm_state"] not in ("PENDING", "RUNNING", "COMPLETING"):
                return record
        return record

    def _read(self, request: RunRequest, name: str) -> str:
        path = shlex.quote(f"{self.remote_dir(request)}/{name}")
        read = _guarded(self.ssh_command(f"cat {path} 2>/dev/null || true"), self.connect_timeout + 60)
        return read.stdout or ""

    def _cancel(self, job_id: str) -> None:
        try:
            _guarded(self.ssh_command(f"scancel {job_id}"), self.connect_timeout + 30)
        except RunnerError:
            pass

    def run(self, request: RunRequest) -> RunResult:
        directory = Path(request.directory)
        request.write_inputs()
        self._send(request)
        job_id = self._submit(request)

        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        deadline = started + self.max_queue_wait + request.timeout_seconds + KILL_GRACE + SCHEDULER_MARGIN
        timed_out_here = False
        polls = 0
        try:
            while self._in_queue(job_id) is not None:
                if time.monotonic() > deadline:
                    timed_out_here = True
                    self._cancel(job_id)
                    break
                _remote._sleep(self.poll_delays[min(polls, len(self.poll_delays) - 1)])
                polls += 1
        except BaseException:
            # Whatever stopped this side waiting, the job must not be left in
            # the queue with nobody to collect it.
            self._cancel(job_id)
            raise

        accounting = self._accounting(job_id) or {}
        stdout = self._read(request, f"slurm-{job_id}.out")
        stderr = self._read(request, f"slurm-{job_id}.err")
        if self.fetch_outputs:
            self._fetch(request)

        stderr, on_node = split_node_timing(stderr)
        state = accounting.get("slurm_state")
        exit_code = accounting.get("exit_code")
        timed_out = timed_out_here or state == "TIMEOUT" or exit_code in _remote.TIMED_OUT_CODES
        if exit_code is None:
            exit_code = 124 if timed_out else (0 if state == "COMPLETED" else 1)
        if timed_out:
            exit_code = 124
            stderr += f"\nAutoOpenSn: SLURM job {job_id} stopped at its time limit.\n"
        elif state in ENDED_BADLY and exit_code == 0:
            # Cancelled or failed by the scheduler without the script's own
            # failure: still a failure, and its row must say so.
            exit_code = 1
            stderr += f"\nAutoOpenSn: SLURM job {job_id} ended {state}.\n"

        metadata = {
            key: accounting.get(key)
            for key in ("slurm_job_id", "slurm_state", "queue_wait", "node", "submitted_at")
        }
        metadata["slurm_job_id"] = metadata["slurm_job_id"] or job_id
        metadata["partition"] = self.partition

        result = RunResult(
            case_id=request.case_id,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            wall_time=on_node if on_node is not None else time.monotonic() - started,
            directory=directory,
            runner=self.name,
            command=self.submit_command(request),
            timed_out=timed_out,
            replayed=False,
            started_at=started_at,
            metadata=metadata,
        )
        result.persist()
        return result

    def config(self) -> dict[str, Any]:
        """Identity for the cache key: everything the direct runner's has, and
        the partition and account, which decide where and as whom it ran."""
        base = super().config()
        base.update(runner=self.name, partition=self.partition, account=self.account)
        return base
