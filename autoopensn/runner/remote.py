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
import posixpath
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
DEFAULT_MODULES = ("opensn/gcc/15", "python3/3.12.3")
"""In this order, and the order matters. ``opensn/gcc/15`` is the compiler and
MPI; the site's Python 3.12 is linked against ``libmpi.so.12`` and cannot even
start until that is on the library path. Loaded the other way round, the Python
module fails without a word and the run falls back to the system Python 3.6,
which then fails on ``import pyopensn`` a long way from the cause."""

DEFAULT_SOURCE = "opensn"
"""The OpenSn source checkout on the node, relative to the home directory.

Data files are copied out of it on the node itself rather than uploaded from
here: every tutorial keeps its meshes and cross sections beside its notebook,
the build already needed this checkout at the pinned commit, and so the files
are already on the node at exactly the revision the templates were written
against."""

DEFAULT_PYTHONPATH = "$HOME/opensn/build"
"""Where a build with ``-DOPENSN_WITH_PYTHON_MODULE=ON`` leaves the package: the
extension compiles to ``build/pyopensn/__init__.*.so``, so the build directory
itself is what goes on the path. ``$HOME`` rather than ``~`` because the value is
expanded inside double quotes, where a tilde is not."""
DEFAULT_REMOTE_ROOT = "autoopensn-runs"
"""Relative, and so relative to the remote home directory: both ``ssh host cmd``
and ``scp host:path`` start there. A leading ``~`` looks equivalent and is not.
Every path here is passed through ``shlex.quote``, and a quoted tilde is never
expanded — ``mkdir -p '~/x'`` makes a directory literally named ``~`` while
``scp`` expands its own tilde and copies elsewhere, so the script lands in one
place and the run looks for it in another."""

CONNECT_TIMEOUT = 20
"""Seconds to wait for the SSH handshake. Distinct from the run timeout: a
cluster that is down should say so in seconds, not hold a study for an hour."""

TIMING_MARKER = "AUTOOPENSN_NODE_TIMING"
"""Prefixes the start and end timestamps the node prints around ``mpiexec``."""

NODE_TIMEOUT_MARGIN = 120
"""How much longer this side waits than the node lets the run go on.

The node enforces the run's time limit itself, with coreutils ``timeout``, and
this side waits a little longer so that it hears the node report the timeout
instead of giving up first. When this side gave up first — the only limit
there was — it dropped the connection and the job carried on: the glovebox
tutorial hung after an output error with eight ranks spinning at full CPU, and
was still running on a shared node seventeen minutes after it had been
reported as timed out."""

KILL_GRACE = 30
"""Seconds between asking the run to stop and killing it outright."""

TIMED_OUT_CODES = (124, 137)
"""What ``timeout`` returns: 124 when the run stopped on the polite signal, 137
when it had to be killed."""

PROBE_TIMEOUT = 4
"""Shorter still, for deciding *whether* to use the cluster at all. A firewall
drops rather than refuses, so an unreachable host costs the full timeout, and
that wait happens before the user sees anything at all. Four seconds is long
enough for a handshake over a VPN and short enough not to read as a hang."""


_run_process = subprocess.run
"""The one entry point this module uses to start a process.

A module-level name rather than ``subprocess.run`` directly, so that it can be
replaced for this module alone. Patching ``subprocess.run`` reaches every module
that imported ``subprocess`` — it is the same object — which is how a guard
meant to keep tests off the cluster once disabled the local runner as well."""


SSH_FAILURE = 255
"""ssh's own exit status when it could not connect or lost the connection."""

CONNECTION_PHRASES = (
    "Connection refused",
    "Connection closed",
    "Connection reset",
    "Connection timed out",
    "Operation timed out",
    "kex_exchange_identification",
    "Broken pipe",
)

RETRY_DELAYS = (3.0, 10.0, 30.0)
"""Seconds to wait before each retry of a connection that failed outright.

A login node that sees a burst of new connections refuses them for a while —
running every scenario once refused nine in a row. Waiting and trying again is
what clears it, and it is safe precisely because nothing reached the node: a
retry is only made when ssh failed before anything on the far side ran."""


_sleep = time.sleep
"""Module-level for the same reason as ``_run_process``: patching ``time.sleep``
replaces it in the one shared ``time`` module, for everything."""


def _connection_failed(returncode: int, stdout: str, stderr: str) -> bool:
    """True only when ssh never got a program running on the far side.

    Exit 255 alone is not enough — a remote program can exit 255 too — so the
    message must be ssh's, and nothing may have come back on stdout, which is
    what separates "could not connect" from "connected, ran, then failed".
    """
    return (
        returncode == SSH_FAILURE
        and not stdout.strip()
        and any(phrase in stderr for phrase in CONNECTION_PHRASES)
    )


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
        for delay in (*RETRY_DELAYS, None):
            completed = _run_process(
                list(command), capture_output=True, text=True, timeout=timeout, check=False
            )
            if delay is None or not _connection_failed(
                completed.returncode, completed.stdout, completed.stderr
            ):
                return completed
            _sleep(delay)
        return completed
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
        pythonpath: Optional[str] = DEFAULT_PYTHONPATH,
        source: str = DEFAULT_SOURCE,
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
        if pythonpath and any(c in pythonpath for c in '"`\\\n'):
            # Expanded inside double quotes on the remote side, so these are
            # the characters that could end the string or run something.
            raise RunnerError(f"pythonpath contains a character that is not allowed: {pythonpath!r}")
        self.pythonpath = pythonpath
        self.source = source.rstrip("/")
        root = remote_root.rstrip("/")
        if root == "~":
            root = "."
        elif root.startswith("~/"):
            root = root[2:]
        self.remote_root = root
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
            lines.append(f"export MODULEPATH={shlex.quote(self.module_path)}:${{MODULEPATH:-}}")
        for module in self.modules:
            lines.append(f"module load {shlex.quote(module)}")
        if self.pythonpath:
            # Prepended: the Python module sets PYTHONPATH to its own
            # site-packages, where numpy and mpi4py live, and replacing it
            # would trade one import error for another.
            lines.append(f'export PYTHONPATH="{self.pythonpath}:${{PYTHONPATH:-}}"')
        return "\n".join(lines)

    def remote_script(self, request: RunRequest) -> str:
        """The shell program executed on the cluster node.

        ``set -eu`` so that a module that fails to load stops the run instead of
        letting ``mpiexec`` find the system Python and produce an import error
        that reads like a bug in the generated script.

        The simulation is timed here, on the node, around ``mpiexec`` alone.
        Timed from this side, the clock also ran through the SSH handshake, the
        module loads and the copy back — two to four seconds of network noise
        around a sub-second solve, which is what a "which setting is fastest"
        table was then comparing. OpenSn's own ``Elapsed execution time`` line
        would avoid the network too, but it is printed to a tenth of a second,
        and at that resolution most cases of a fast study read the same.
        The two timestamps go to stderr behind a marker, so stdout stays exactly
        what OpenSn printed.
        """
        directory = self.remote_dir(request)
        preamble = self.module_preamble()
        launch = " ".join(
            (
                "timeout",
                f"--kill-after={KILL_GRACE}",
                str(max(1, int(request.timeout_seconds))),
                shlex.quote(self.launcher),
                "-n",
                str(request.num_procs),
                shlex.quote(self.python),
                shlex.quote(request.script_path.name),
            )
        )
        return "\n".join(
            line
            for line in (
                "set -eu",
                preamble,
                f"cd {shlex.quote(directory)}",
                *self.staging_lines(request),
                "__t0=$(date +%s.%N)",
                # The run's own failure must not abort before the end time is
                # taken, or a crashed case loses its timing and its exit code.
                "set +e",
                launch,
                "__rc=$?",
                "__t1=$(date +%s.%N)",
                f'echo "{TIMING_MARKER} $__t0 $__t1" >&2',
                "exit $__rc",
            )
            if line
        )

    def staging_lines(self, request: RunRequest) -> list[str]:
        """Lay the run directory out as a mirror of the source tree.

        Each data file is copied to its own path inside the source, and the
        script to the notebook's folder, which becomes the working directory.
        A tutorial then resolves ``glovebox.msh``, ``LANL30/OpenMC/x.h5`` and
        ``../../../../../test/assets/mesh/y.msh`` alike, exactly as written.
        Copying everything flat into one folder — the first version of this —
        broke every reference with a directory in it.

        Before the clock starts, so staging is not simulation time, and under
        ``set -e``, so a file that is not there stops the run with cp's own
        message instead of reaching OpenSn as a mesh it could not read.
        """
        lines = []
        for path in request.data_files:
            if path.startswith("/") or ".." in Path(path).parts:
                raise RunnerError(f"data file must be a path inside the OpenSn source: {path!r}")
            parent = posixpath.dirname(path)
            if parent:
                lines.append(f"mkdir -p {shlex.quote(parent)}")
            source = shlex.quote(self.source + "/" + path)
            lines.append(f'cp -r "$HOME"/{source} {shlex.quote(path)}')
        if request.workdir:
            if request.workdir.startswith("/") or ".." in Path(request.workdir).parts:
                raise RunnerError(f"workdir must be inside the run directory: {request.workdir!r}")
            name = shlex.quote(request.script_path.name)
            lines.append(f"mkdir -p {shlex.quote(request.workdir)}")
            lines.append(f"cp {name} {shlex.quote(request.workdir)}/{name}")
            lines.append(f"cd {shlex.quote(request.workdir)}")
        return lines

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

    def remote_command(self, script: str) -> str:
        """``script`` as the one string ssh will hand to the remote shell.

        ssh does not pass its arguments through separately: it joins them with
        spaces and the remote shell parses the result again. Passed as
        ``"bash", "-l", "-c", script``, the remote side read
        ``bash -l -c set -eu …``, ran ``set`` on its own — which prints every
        shell variable, and those lines then headed every saved stdout — and
        ran the rest of the script in the outer shell, where ``set -eu`` was
        never in effect. So the whole program is quoted once, here, and ssh
        gets it as a single argument.

        ``bash -l`` because ``module`` is a shell function defined by the site's
        profile scripts; a non-login shell does not have it.
        """
        return f"bash -l -c {shlex.quote(script)}"

    def command_for(self, request: RunRequest) -> tuple[str, ...]:
        """The full command that runs this case, module loads included."""
        return self.ssh_command(self.remote_command(self.remote_script(request)))

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
            self.ssh_command(self.remote_command(check)), self.connect_timeout + 60
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
            for delay in (*RETRY_DELAYS, None):
                completed = _run_process(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=request.timeout_seconds + KILL_GRACE + NODE_TIMEOUT_MARGIN,
                    env=environment,
                    check=False,
                )
                stdout, stderr, exit_code = completed.stdout, completed.stderr, completed.returncode
                if delay is None or not _connection_failed(exit_code, stdout, stderr):
                    break
                _sleep(delay)
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

        round_trip = time.monotonic() - started
        stderr, on_node = split_node_timing(stderr)
        if exit_code in TIMED_OUT_CODES and not timed_out:
            # The node stopped it. Recorded as a timeout, which is what it was,
            # rather than as a crash with an unexplained exit code.
            timed_out = True
            stderr += f"\nAutoOpenSn: stopped on {self.host} after {request.timeout_seconds:.0f}s.\n"

        result = RunResult(
            case_id=request.case_id,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            # The simulation's own time on the node when the node reported it.
            # Only a run that never got that far — a timeout, a failed copy —
            # falls back to the round trip, which is then the honest figure.
            wall_time=on_node if on_node is not None else round_trip,
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
            "pythonpath": self.pythonpath,
            # Results recorded before this measured wall time from here, round
            # trip included; after it, on the node around mpiexec alone. Same
            # column, different quantity — so they must not share cache entries.
            "wall_time": "on-node",
            "launcher": self.launcher,
            "python": self.python,
        }


def split_node_timing(stderr: str) -> tuple[str, Optional[float]]:
    """Remove the node's timing line from stderr, and return the seconds it gives.

    Returns the stderr without the marker line, so what is stored and shown is
    only what the simulation itself wrote, and None for the time when the
    marker is absent or unreadable.
    """
    kept: list[str] = []
    seconds: Optional[float] = None
    for line in stderr.splitlines(keepends=True):
        if line.startswith(TIMING_MARKER):
            parts = line.split()
            try:
                seconds = max(0.0, float(parts[2]) - float(parts[1]))
            except (IndexError, ValueError):
                seconds = None
            continue
        kept.append(line)
    return "".join(kept), seconds


def _as_text(stream: Any) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)
