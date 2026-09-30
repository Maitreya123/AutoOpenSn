"""Running one sweep point, for real or from a recording.

OpenSn is not installed on every machine this tool runs on, and building it
takes an afternoon. So running is behind an interface with two implementations:

``LocalMPIRunner``
    Really invokes ``mpiexec -n N python script.py`` on this machine. Opt-in.

``DockerRunner``
    The same, inside a container that has OpenSn built. Opt-in, and on a host
    whose toolchain does not agree with itself about architecture, the more
    practical of the two.

``ReferenceRunner``
    Solves the one-dimensional scenarios directly, with this package's own
    solver, and is explicit that the result is not OpenSn's.

``FakeRunner``
    Replays stdout, stderr, exit code, and output files recorded under
    ``tests/fixtures``. The default, and what the entire test suite uses.

The important consequence is that everything downstream of this interface, which
is all the parsing, tabulating, caching, and narrating, is developed and tested
against recorded output. That is not a compromise forced by a missing
dependency. A parser that can only be exercised by a twenty-minute simulation is
a parser nobody refactors.
"""

from autoopensn.runner.base import (
    RunRequest,
    RunResult,
    Runner,
    RunnerError,
    run_directory,
)
from autoopensn.runner.docker import DockerRunner
from autoopensn.runner.fake import FakeRunner, FixtureMissing
from autoopensn.runner.local_mpi import LocalMPIRunner
from autoopensn.runner.reference import ReferenceRunner, UnsupportedScenario, supported

__all__ = [
    "DockerRunner",
    "FakeRunner",
    "FixtureMissing",
    "LocalMPIRunner",
    "ReferenceRunner",
    "RunRequest",
    "RunResult",
    "Runner",
    "RunnerError",
    "UnsupportedScenario",
    "run_directory",
    "supported",
]
