"""The Runner interface, and the on-disk layout of a run.

One run per sweep point, each in its own directory:

    runs/<run_id>/<case_id>/
        script.py       the rendered script, with its provenance header
        spec.yaml       the whole spec, so the directory is self-describing
        parameters.json the fully resolved parameters for this point
        stdout.txt
        stderr.txt
        result.json     exit code, wall time, command, runner name
        <outputs>       anything the script itself wrote

The directory is the record. A results table can be rebuilt from it without
rerunning anything, which is what makes the run cache safe to trust and a
failure worth keeping.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


class RunnerError(Exception):
    """A runner could not run: no executable, no fixture, a bad request."""


def _safe(name: str) -> str:
    """A path component that cannot escape its parent or surprise a shell."""
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "_", name).strip("._-")
    return cleaned or "unnamed"


def run_directory(root: Path, run_id: str, case_id: str) -> Path:
    """Directory for one sweep point. Created if absent."""
    path = Path(root) / _safe(run_id) / _safe(case_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class RunRequest:
    """Everything a runner needs to execute one sweep point."""

    case_id: str
    script: str
    """The full text of the rendered script, provenance header included."""
    directory: Path
    """Working directory for this run. The script is written here as script.py."""
    template: str = ""
    """Name of the template this point came from.

    Carried so that a runner which works from the parameters rather than the
    script, such as the reference solver, knows what problem it was handed
    instead of inferring it from the shape of the parameter set.
    """
    num_procs: int = 1
    timeout_seconds: float = 1800.0
    data_files: tuple[str, ...] = ()
    """Files the script opens by bare name — meshes, cross sections — as paths
    inside the OpenSn source tree, which is where every tutorial keeps them.
    A runner that cannot stage them should say so rather than run without."""
    workdir: str = ""
    """Where the script runs, relative to the run directory; empty for the run
    directory itself. A tutorial is written to run from its own notebook folder
    inside the OpenSn source tree, and resolves every data path relative to
    that — including paths that climb out of it, like the meshes it borrows from
    ``../../../../../test/assets``. Running it from its notebook's position in a
    mirror of the tree, with each data file at its true source path, makes every
    such reference resolve exactly as written, without touching the script."""
    parameters: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)

    @property
    def script_path(self) -> Path:
        return Path(self.directory) / "script.py"

    def write_inputs(self, spec_yaml: Optional[str] = None) -> Path:
        """Write the script and its context into the run directory."""
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.script_path.write_text(self.script)
        (directory / "parameters.json").write_text(
            json.dumps(self.parameters, indent=2, sort_keys=True, default=str)
        )
        if spec_yaml is not None:
            (directory / "spec.yaml").write_text(spec_yaml)
        return self.script_path


@dataclass(frozen=True)
class RunResult:
    """What one run produced."""

    case_id: str
    exit_code: int
    stdout: str
    stderr: str
    wall_time: float
    """Seconds. For a replayed run this is the recorded time, not the replay time."""
    directory: Path
    runner: str
    command: tuple[str, ...] = ()
    timed_out: bool = False
    replayed: bool = False
    started_at: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def failure_reason(self) -> Optional[str]:
        """One line saying what went wrong, or None."""
        if self.timed_out:
            return f"timed out after {self.wall_time:.1f}s"
        if self.exit_code == 127:
            # Reserved for "this could not be attempted", which is a different
            # thing from a simulation that ran and failed, and reads differently
            # in a results table.
            return "no recorded run for this case; install OpenSn or record a fixture"
        if self.exit_code != 0:
            tail = [line for line in self.stderr.strip().splitlines() if line.strip()]
            detail = f": {tail[-1][:200]}" if tail else ""
            return f"exit code {self.exit_code}{detail}"
        return None

    def persist(self) -> Path:
        """Write stdout, stderr, and the result record into the run directory."""
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "stdout.txt").write_text(self.stdout)
        (directory / "stderr.txt").write_text(self.stderr)
        record = {
            "case_id": self.case_id,
            "exit_code": self.exit_code,
            "wall_time": self.wall_time,
            "runner": self.runner,
            "command": list(self.command),
            "timed_out": self.timed_out,
            "replayed": self.replayed,
            "started_at": self.started_at or datetime.now(timezone.utc).isoformat(),
        }
        path = directory / "result.json"
        path.write_text(json.dumps(record, indent=2))
        return path

    @classmethod
    def from_directory(cls, directory: Path) -> "RunResult":
        """Rebuild a result from a run directory, without rerunning anything."""
        directory = Path(directory)
        record_path = directory / "result.json"
        if not record_path.exists():
            raise RunnerError(f"no result.json in {directory}")
        record = json.loads(record_path.read_text())
        return cls(
            case_id=record["case_id"],
            exit_code=record["exit_code"],
            stdout=(directory / "stdout.txt").read_text()
            if (directory / "stdout.txt").exists()
            else "",
            stderr=(directory / "stderr.txt").read_text()
            if (directory / "stderr.txt").exists()
            else "",
            wall_time=record["wall_time"],
            directory=directory,
            runner=record.get("runner", "unknown"),
            command=tuple(record.get("command", ())),
            timed_out=record.get("timed_out", False),
            replayed=record.get("replayed", False),
            started_at=record.get("started_at"),
        )


class Runner(ABC):
    """Executes one sweep point and returns what it produced.

    Implementations must not raise on a failed *simulation*. A non-zero exit
    code, a crash, or a timeout is a ``RunResult`` with ``ok`` False, because a
    study where one point diverges is a study with one bad row, not a study that
    stops. ``RunnerError`` is for a failure to *attempt* the run at all: no
    ``mpiexec``, no fixture, an unwritable directory.
    """

    name: str = "runner"

    @abstractmethod
    def run(self, request: RunRequest) -> RunResult:
        """Execute one sweep point."""

    def config(self) -> dict[str, Any]:
        """Identity of this runner, for the cache key.

        Two runs are interchangeable only if the runner that produced them is.
        A result from a four-rank MPI run and a result replayed from a fixture
        are not the same evidence, and the cache must not confuse them.
        """
        return {"runner": self.name}
