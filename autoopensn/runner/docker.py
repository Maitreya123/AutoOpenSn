"""Running a rendered script inside an OpenSn container.

The third ``Runner``, and on a machine like this one the most practical. A
native build has to agree with the host on architecture, and this host does not
agree with itself: the machine is arm64, its Homebrew is the x86_64 build under
Rosetta, and its Python is a universal binary running arm64. A container
sidesteps all of that, because everything inside it is built together.

It is also the path OpenSn documents. ``distribution/docker/Dockerfile`` builds
the dependency stack and the Python module in one Ubuntu image, which is the
same thing a local build does, with the difference that failure costs an image
layer rather than an afternoon of someone's development environment.

The contract is identical to ``LocalMPIRunner``: write the script, run it with
``mpiexec``, capture both streams, return a ``RunResult``. The only difference
is where the process lives. The run directory is bind-mounted, so the script,
the outputs, stdout, stderr and the result record all land in the same place
they would for a local run, and nothing downstream can tell the difference.
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

DEFAULT_IMAGE = "opensn:latest"
WORKDIR = "/work"


class DockerRunner(Runner):
    """Runs scripts inside a container that has OpenSn built."""

    name = "docker"

    def __init__(
        self,
        image: str = DEFAULT_IMAGE,
        *,
        docker: str = "docker",
        python: str = "python3",
        extra_run_args: Sequence[str] = (),
        user: Optional[str] = None,
    ):
        self.image = image
        self.docker = docker
        self.python = python
        self.extra_run_args = tuple(extra_run_args)
        # Files written inside the container are owned by its user, and a run
        # directory full of root-owned files is a nuisance the first time and a
        # permissions bug every time after. Default to the caller's own ids.
        self.user = user if user is not None else f"{os.getuid()}:{os.getgid()}"

    # --- discovery ---

    def resolve_docker(self) -> str:
        found = shutil.which(self.docker)
        if not found:
            raise RunnerError(
                f"{self.docker!r} is not on PATH. Install Docker, or run against "
                f"recorded fixtures, which is the default."
            )
        return found

    def daemon_running(self) -> bool:
        try:
            completed = subprocess.run(
                [self.resolve_docker(), "info"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return completed.returncode == 0
        except (RunnerError, subprocess.SubprocessError):
            return False

    def image_present(self) -> bool:
        try:
            completed = subprocess.run(
                [self.resolve_docker(), "image", "inspect", self.image],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            return completed.returncode == 0
        except (RunnerError, subprocess.SubprocessError):
            return False

    def preflight(self) -> None:
        """Check a run could happen, with a message naming the missing step."""
        self.resolve_docker()
        if not self.daemon_running():
            raise RunnerError(
                "the Docker daemon is not running. Start Docker Desktop, then try again."
            )
        if not self.image_present():
            raise RunnerError(
                f"no image named {self.image!r}. Build it from the OpenSn checkout:\n"
                f"    docker build -f distribution/docker/Dockerfile -t {self.image} .\n"
                f"See docs/installing_opensn.md."
            )

    # --- the interface ---

    def command_for(self, request: RunRequest) -> tuple[str, ...]:
        docker = self.resolve_docker()
        directory = Path(request.directory).resolve()
        return (
            docker,
            "run",
            "--rm",
            "--user",
            self.user,
            "--volume",
            f"{directory}:{WORKDIR}",
            "--workdir",
            WORKDIR,
            *self.extra_run_args,
            self.image,
            "mpiexec",
            "-n",
            str(request.num_procs),
            # OpenMPI refuses to run as root without this, and a container's
            # default user is root. Harmless when the user is not root.
            "--allow-run-as-root",
            self.python,
            request.script_path.name,
        )

    def run(self, request: RunRequest) -> RunResult:
        command = self.command_for(request)
        directory = Path(request.directory)
        request.write_inputs()

        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        timed_out = False

        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=request.timeout_seconds,
                check=False,
            )
            stdout, stderr, exit_code = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as expired:
            timed_out = True
            stdout = _as_text(expired.stdout)
            stderr = _as_text(expired.stderr) + (
                f"\nAutoOpenSn: killed after {request.timeout_seconds:.0f}s. The "
                f"container may still be running; check `docker ps`.\n"
            )
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
        # The image tag is part of the identity of a result: the same script run
        # against two different OpenSn builds is two different measurements, and
        # the cache must not conflate them.
        return {
            "runner": self.name,
            "image": self.image,
            "python": self.python,
            "extra_run_args": list(self.extra_run_args),
        }


def _as_text(stream: Any) -> str:
    if stream is None:
        return ""
    if isinstance(stream, bytes):
        return stream.decode("utf-8", errors="replace")
    return str(stream)
