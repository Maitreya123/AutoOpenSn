"""Running a rendered script on a cluster node over SSH.

This exists because OpenSn is genuinely hard to install and already built
somewhere else. On TAMU's Orchard cluster the node ``class01`` carries an
``opensn/gcc/15`` module that supplies PETSc, VTK, HDF5, MPI and a compiled
OpenSn, which is the several-hour build this package otherwise asks people to
do themselves.

The runner is synchronous: it submits nothing to a queue, waits for the
process, and returns when it exits. That matches an interactive allocation
(``srun``/``salloc``, or a node you are allowed to use directly) and it keeps
the whole pipeline downstream of ``Runner`` unchanged, because from the caller's
point of view this behaves exactly like ``LocalMPIRunner`` with a slower
process start. A batch runner that submits with ``sbatch`` and polls a job id is
a different class of thing, because a submitted run outlives the Python process
that asked for it; that belongs in its own module when it is needed.

**Everything that decides what will execute is a pure function.** ``remote_dir``,
``remote_script`` and ``command_for`` take a request and return strings, so the
exact text that would run on a shared machine can be asserted in a test on a
laptop with no cluster account. Only ``run`` and ``preflight`` open a
connection. This is deliberate: the failure that matters here is not "the SSH
call errored", it is "we quietly ran the wrong command on someone else's
compute node", and that one is only catchable by reading the command.

Nothing here is selected by default and the test suite never connects.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from autoopensn.runner.base import RunRequest, RunResult, Runner, RunnerError

DEFAULT_HOST = "class01"
"""An ``ssh`` alias, not a hostname. Orchard's compute nodes are reachable only
through the front end, so the jump belongs in ``~/.ssh/config`` where ``scp``
and every other tool picks it up too, rather than being half-encoded here."""

DEFAULT_MODULE_PATH = "/scratch-local/software/modulefiles"
DEFAULT_MODULES = ("opensn/gcc/15",)
DEFAULT_REMOTE_ROOT = "~/autoopensn-runs"

CONNECT_TIMEOUT = 20
"""Seconds to wait for the SSH handshake. Distinct from the run timeout: a
cluster that is down should say so in seconds, not hold a study for an hour."""

PROBE_TIMEOUT = 4
"""Shorter still, for deciding *whether* to use the cluster at all. A firewall
drops rather than refuses, so an unreachable host costs the full timeout, and
that wait happens before the user sees anything at all. Four seconds is long
enough for a handshake over a VPN and short enough not to read as a hang."""


def _guarded(command: Sequence[str], timeout: float) -> subprocess.CompletedProcess:
    """Run a helper process, turning every way it can fail into RunnerError.

    ``Runner`` promises that a failure to *attempt* a run is a ``RunnerError``.
    A raw ``TimeoutExpired`` escaping from here breaks that promise, and the
    place it breaks it is the automatic runner choice: a cluster that is merely
    unreachable would crash the request instead of falling back to the solver
    that works.

    Timeouts matter more here than for most subprocesses because a firewall
    drops rather than refuses. SSH's own ``ConnectTimeout`` does not always cap
    that — with a jump host there are two handshakes, and a dropped packet can
    hold the process well past it — so the outer timeout is the one that has to
    be authoritative.
    """
    try:
        return subprocess.run(
            list(command), capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as expired:
        raise RunnerError(
            f"{command[0]} did not respond within {timeout:.0f}s: "
            + " ".join(str(part) for part in command[:4])
        ) from expired
    except FileNotFoundError as exc:
        raise RunnerError(f"could not start {command[0]!r}: {exc}") from exc


class RemoteRunner(Runner):
    """Runs scripts on a cluster node over SSH, and waits for them."""

    name = "remote"

    def __init__(
        self,
        *,
        host: str = DEFAULT_HOST,
        modules: Sequence[str] = DEFAULT_MODULES,
        module_path: Optional[str] = DEFAULT_MODULE_PATH,
        remote_root: str = DEFAULT_REMOTE_ROOT,
        python: str = "python3",
        launcher: str = "mpiexec",
        ssh: str = "ssh",
        scp: str = "scp",
        ssh_options: Sequence[str] = (),
        fetch_outputs: bool = True,
        connect_timeout: int = CONNECT_TIMEOUT,
    ):
        self.host = host
        self.modules = tuple(modules)
        self.module_path = module_path
        self.remote_root = remote_root.rstrip("/")
        self.python = python
        self.launcher = launcher
        self.ssh = ssh
        self.scp = scp
        self.ssh_options = tuple(ssh_options)
        self.fetch_outputs = fetch_outputs
        self.connect_timeout = connect_timeout

    # --- pure: what would run -------------------------------------------

    def remote_dir(self, request: RunRequest) -> str:
        """Where this case's files live on the cluster.

        Named after the local run directory's last two components, so a run
        directory here and its counterpart there carry the same name and a
        person reading ``runs/`` can find the remote side without a lookup.
        """
        directory = Path(request.directory)
        parent = directory.parent.name or "run"
        return f"{self.remote_root}/{parent}/{directory.name}"

    def module_preamble(self) -> str:
        """The lines that put OpenSn on PATH, or empty if nothing to load.

        ``MODULEPATH`` is prepended rather than replaced because the site's own
        modulefiles live on it, and a login shell that has lost them is a
        confusing thing to hand someone.
        """
        lines: list[str] = []
        if self.module_path:
            lines.append(f"export MODULEPATH={shlex.quote(self.module_path)}:$MODULEPATH")
        for module in self.modules:
            lines.append(f"module load {shlex.quote(module)}")
        return "\n".join(lines)

    def remote_script(self, request: RunRequest) -> str:
        """The shell program executed on the cluster node.

        ``set -eu`` so that a module that fails to load stops the run instead of
        letting ``mpiexec`` find the system Python and produce an import error
        that reads like a bug in the generated script.
        """
        directory = self.remote_dir(request)
        preamble = self.module_preamble()
        return "\n".join(
            line
            for line in (
                "set -eu",
                preamble,
                f"cd {shlex.quote(directory)}",
                " ".join(
                    (
                        shlex.quote(self.launcher),
                        "-n",
                        str(request.num_procs),
                        shlex.quote(self.python),
                        shlex.quote(request.script_path.name),
                    )
                ),
            )
            if line
        )

    def ssh_command(self, *arguments: str) -> tuple[str, ...]:
        """An ``ssh`` invocation carrying this runner's connection options."""
        return (
            self.ssh,
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            *self.ssh_options,
            self.host,
            *arguments,
        )

    def command_for(self, request: RunRequest) -> tuple[str, ...]:
        """The full command that runs this case, module loads included.

        ``bash -l`` because ``module`` is a shell function defined by the site's
        profile scripts; a non-login shell does not have it, and the failure
        reads as "module: command not found" a long way from its cause.
        """
        return self.ssh_command("bash", "-l", "-c", self.remote_script(request))

    # --- impure: connecting ---------------------------------------------

    def preflight(self) -> None:
        """Check we can reach the host and that OpenSn loads there.

        Run before a study rather than during it. Discovering on case four of
        seven that the module name is wrong wastes the first three and leaves a
        half-populated table.
        """
        if not shutil.which(self.ssh):
            raise RunnerError(
                f"{self.ssh!r} is not on PATH, so no remote run is possible."
            )
        if not shutil.which(self.scp):
            raise RunnerError(f"{self.scp!r} is not on PATH; it is needed to send scripts.")

        probe = _guarded(self.ssh_command("true"), self.connect_timeout + 4)
        if probe.returncode != 0:
            raise RunnerError(
                f"cannot reach {self.host!r} over SSH: "
                f"{probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else 'no detail'}.\n"
                "Check that the VPN is up and that `ssh " + self.host + "` works by hand. "
                "Key-based login is required; this runner never prompts for a password."
            )

        check = "\n".join(
            line
            for line in (
                "set -eu",
                self.module_preamble(),
                "command -v " + shlex.quote(self.launcher),
                "command -v " + shlex.quote(self.python),
            )
            if line
        )
        loaded = _guarded(
            self.ssh_command("bash", "-l", "-c", check), self.connect_timeout + 60
        )
        if loaded.returncode != 0:
            raise RunnerError(
                f"reached {self.host!r}, but the OpenSn environment did not load.\n"
                f"Tried: {', '.join(self.modules) or '(no modules)'}\n"
                f"{loaded.stderr.strip()[:400]}\n"
                "Run `module avail` on the node to see what is actually offered."
            )

    def _send(self, request: RunRequest) -> None:
        """Create the remote directory and copy the script into it."""
        directory = self.remote_dir(request)
        made = _guarded(
            self.ssh_command("mkdir", "-p", shlex.quote(directory)),
            self.connect_timeout + 30,
        )
        if made.returncode != 0:
            raise RunnerError(
                f"could not create {directory} on {self.host}: {made.stderr.strip()[:200]}"
            )

        copied = _guarded(
            (
                self.scp,
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={self.connect_timeout}",
                *self.ssh_options,
                str(request.script_path),
                f"{self.host}:{directory}/{request.script_path.name}",
            ),
            self.connect_timeout + 120,
        )
        if copied.returncode != 0:
            raise RunnerError(
                f"could not copy the script to {self.host}: {copied.stderr.strip()[:200]}"
            )

    def _fetch(self, request: RunRequest) -> None:
        """Bring back anything the script wrote, best effort.

        Deliberately not fatal. The observables this package parses come from
        stdout, which is already in hand by the time this runs, so a failure to
        retrieve a VTK file should not turn a successful simulation into a
        failed row.
        """
        directory = self.remote_dir(request)
        try:
            _guarded(
                (
                    self.scp,
                    "-r",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    f"ConnectTimeout={self.connect_timeout}",
                    *self.ssh_options,
                    f"{self.host}:{directory}/.",
                    str(Path(request.directory)),
                ),
                self.connect_timeout + 300,
            )
        except RunnerError:
            # Best effort, and deliberately so: the observables this package
            # parses come from stdout, which is already in hand. A VTK file that
            # would not copy back must not turn a successful simulation into a
            # failed row.
            pass

    def run(self, request: RunRequest) -> RunResult:
        command = self.command_for(request)
        directory = Path(request.directory)
        request.write_inputs()
        self._send(request)

        environment = {**os.environ, **request.environment}
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        timed_out = False

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                env=environment,
                check=False,
            )
            stdout, stderr, exit_code = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as expired:
            timed_out = True
            stdout = _as_text(expired.stdout)
            stderr = _as_text(expired.stderr) + (
                f"\nAutoOpenSn: gave up waiting after {request.timeout_seconds:.0f}s.\n"
                f"The job may still be running on {self.host}. Check with "
                f"`ssh {self.host} 'pgrep -af {request.script_path.name}'`.\n"
            )
            exit_code = 124
        except FileNotFoundError as exc:
            raise RunnerError(f"could not start {command[0]!r}: {exc}") from exc

        if self.fetch_outputs and not timed_out:
            self._fetch(request)

        result = RunResult(
            case_id=request.case_id,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            wall_time=time.monotonic() - started,
            directory=directory,
            runner=self.name,
            command=command,
            timed_out=timed_out,
            replayed=False,
            started_at=started_at,
        )
        result.persist()
        return result

    def config(self) -> dict[str, Any]:
        """Identity for the cache key.

        The host and the module list are both in it. A result from an
        ``opensn/gcc/15`` build is not interchangeable with one from
        ``opensn/clang``, and a cache that treated them as the same would hide
        exactly the compiler-dependent difference someone would be looking for.
        """
        return {
            "runner": self.name,
            "host": self.host,
            "modules": list(self.modules),
            "launcher": self.launcher,
            "python": self.python,
        }


def _as_text(stream: Any) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)
