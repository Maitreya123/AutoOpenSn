"""Running one sweep point, for real or from a recording.

OpenSn is hard to install and this package must work without it, so running is
behind an interface with four implementations:

``ReferenceRunner``
    Solves the one-dimensional scenarios directly, with this package's own
    solver, in under a second. The default, and explicit that the result is not
    OpenSn's.

``RemoteRunner``
    Runs on a cluster node over SSH, where OpenSn is already built and supplied
    by a module. The way to run the scenarios the reference solver refuses.

``LocalMPIRunner``
    Invokes ``mpiexec -n N python script.py`` on this machine. Opt-in, and needs
    a local OpenSn build.

``FakeRunner``
    Replays stdout, stderr, exit code, and output files recorded under
    ``tests/fixtures``. What the test suite uses.

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
from autoopensn.runner.fake import FakeRunner, FixtureMissing
from autoopensn.runner.local_mpi import LocalMPIRunner
from autoopensn.runner.reference import ReferenceRunner, UnsupportedScenario, supported
from autoopensn.runner.remote import RemoteRunner

__all__ = [
    "FakeRunner",
    "FixtureMissing",
    "LocalMPIRunner",
    "ReferenceRunner",
    "RemoteRunner",
    "RunRequest",
    "RunResult",
    "Runner",
    "RunnerError",
    "UnsupportedScenario",
    "run_directory",
    "supported",
]
