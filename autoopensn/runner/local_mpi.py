"""Running a rendered script for real, with MPI.

This is the only module in the package that starts a process, and the only one
that can take twenty minutes. It is opt-in: nothing selects it by default, and
the test suite never invokes it against a real OpenSn.

The contract is narrow on purpose. It writes the script, runs
``mpiexec -n N python script.py`` in the run directory with a timeout, captures
both streams, and returns. It does not interpret the output, retry, or decide
whether the physics is sensible. A simulation that diverges is a ``RunResult``
with a non-zero exit code, and the parser upstream of the table is what notices.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

from autoopensn.runner.base import RunRequest, RunResult, Runner, RunnerError

DEFAULT_LAUNCHERS = ("mpiexec", "mpirun")


class LocalMPIRunner(Runner):
    """Runs scripts on this machine under an MPI launcher."""

    name = "local"

    def __init__(
        self,
        *,
        launcher: Optional[str] = None,
        python: Optional[str] = None,
        extra_launcher_args: Sequence[str] = (),
        environment: Optional[dict[str, str]] = None,
    ):
        self.launcher = launcher
        self.python = python or "python3"
        self.extra_launcher_args = tuple(extra_launcher_args)
        self.environment = dict(environment or {})

    # --- discovery ---

    def resolve_launcher(self) -> str:
        """Find an MPI launcher, or explain what is missing.

        The error is long because the fix is long. Someone hitting this has
        almost certainly not built OpenSn yet, and the useful response is to
        point at the build instructions and at the runner that does not need
        them.
        """
        if self.launcher:
            found = shutil.which(self.launcher)
            if not found:
                raise RunnerError(f"MPI launcher {self.launcher!r} is not on PATH")
            return found
        for candidate in DEFAULT_LAUNCHERS:
            found = shutil.which(candidate)
            if found:
                return found
        raise RunnerError(
            "no MPI launcher found on PATH (looked for "
            + ", ".join(DEFAULT_LAUNCHERS)
            + "). OpenSn and MPI are not installed on this machine. Either follow "
            "docs/installing_opensn.md, or run the study against recorded "
            "fixtures, which is the default."
        )

    @staticmethod
    def pyopensn_available() -> bool:
        """Whether the PyOpenSn module can be imported by this interpreter."""
        try:
            import importlib.util  # noqa: PLC0415

            return importlib.util.find_spec("pyopensn") is not None
        except (ImportError, ValueError):
            return False

    def preflight(self) -> None:
        """Check the environment can host a real run, before writing anything."""
        self.resolve_launcher()
        if not shutil.which(self.python):
            raise RunnerError(f"python interpreter {self.python!r} is not on PATH")

    # --- the interface ---

    def command_for(self, request: RunRequest) -> tuple[str, ...]:
        launcher = self.resolve_launcher()
        return (
            launcher,
            "-n",
            str(request.num_procs),
            *self.extra_launcher_args,
            self.python,
            request.script_path.name,
        )

    def run(self, request: RunRequest) -> RunResult:
        command = self.command_for(request)
        directory = Path(request.directory)
        request.write_inputs()

        environment = {**os.environ, **self.environment, **request.environment}
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        timed_out = False

        try:
            completed = subprocess.run(
                command,
                cwd=str(directory),
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
                f"\nAutoOpenSn: killed after {request.timeout_seconds:.0f}s.\n"
            )
            # A timeout is not a crash and not a success. 124 is what timeout(1)
            # uses, and keeping it distinct from a real exit code means a
            # results table can tell "diverged" from "too slow".
            exit_code = 124
        except FileNotFoundError as exc:
            raise RunnerError(f"could not start {command[0]!r}: {exc}") from exc

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
        return {
            "runner": self.name,
            "launcher": Path(self.launcher).name if self.launcher else "auto",
            "python": self.python,
            "extra_launcher_args": list(self.extra_launcher_args),
        }


def _as_text(stream: Any) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)
