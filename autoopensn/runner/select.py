"""Choosing a runner by name, and saying which engine produced the numbers.

Shared by the HTTP API and the MCP server, so the two cannot disagree about
what "auto" means or about how a result is labelled. Each choice returns the
runner and a sentence naming the engine: a table that does not say whether it
came from OpenSn or from this package's own solver is a table nobody should
quote.

Errors are ``RunnerError`` for a runner that exists but cannot run here, and
``UnknownRunner`` for a name that is not a runner at all. Turning those into an
HTTP status or a tool error is the caller's business.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from autoopensn.runner.base import Runner, RunnerError
from autoopensn.runner.fake import FakeRunner
from autoopensn.runner.local_mpi import LocalMPIRunner
from autoopensn.runner.reference import ReferenceRunner
from autoopensn.runner.remote import PROBE_TIMEOUT, RemoteRunner
from autoopensn.runner.slurm import SlurmRunner

DEFAULT_FIXTURES = (
    Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "reed_gmres"
)

RUNNER_KINDS = ("auto", "reference", "slurm", "remote", "fake", "local")

SCHEDULER_ENV = "AUTOOPENSN_SCHEDULER"
"""Set to "ssh" to have 'auto' skip SLURM and run cases directly over SSH."""


class UnknownRunner(ValueError):
    """A runner name that is not one of ``RUNNER_KINDS``."""


def choose_runner(host: Optional[str] = None) -> tuple[Runner, str]:
    """SLURM if the cluster's scheduler answers, then the node directly over SSH,
    then this machine's own solver.

    SLURM first because the cluster is shared: submitting through the scheduler
    is how a study avoids landing on cores someone else is already using, and
    its time limit ends a stuck run even if this side goes away. Direct SSH
    remains for when the scheduler is unavailable, or when someone sets
    AUTOOPENSN_SCHEDULER=ssh to skip it.

    The probes are short because someone is waiting on them; the runner that
    actually runs gets the ordinary, patient timeout.
    """
    if os.getenv(SCHEDULER_ENV, "slurm").lower() != "ssh" and not host:
        try:
            SlurmRunner(connect_timeout=PROBE_TIMEOUT).preflight()
            runner = SlurmRunner()
            return runner, (
                f"Running real OpenSn through SLURM, partition {runner.partition} "
                "(class01)."
            )
        except RunnerError:
            pass  # fall through to direct SSH, which says why if it fails too
    try:
        probe = (
            RemoteRunner(host=host, connect_timeout=PROBE_TIMEOUT)
            if host
            else RemoteRunner(connect_timeout=PROBE_TIMEOUT)
        )
        probe.preflight()
        return (
            RemoteRunner(host=host) if host else RemoteRunner()
        ), f"Running real OpenSn on {probe.host}."
    except RunnerError as exc:
        first_line = str(exc).strip().splitlines()[0]
        return (
            ReferenceRunner(strict=False),
            "The cluster is not reachable, so this ran in AutoOpenSn's own 1D "
            f"solver instead. These are not OpenSn's numbers. ({first_line})",
        )


def build_runner(
    kind: str, host: Optional[str] = None, fixtures: Optional[str] = None
) -> tuple[Runner, str]:
    """A runner chosen by name, or by ``choose_runner`` for 'auto'."""
    if kind == "auto":
        return choose_runner(host)
    if kind == "reference":
        # Not strict: a study touching an unsupported scenario reports those
        # cases as unavailable rather than stopping the whole run.
        return ReferenceRunner(strict=False), "AutoOpenSn's own 1D solver. Not OpenSn."
    if kind == "slurm":
        slurm = SlurmRunner()
        slurm.preflight()
        return slurm, f"Running real OpenSn through SLURM, partition {slurm.partition}."
    if kind == "remote":
        remote = RemoteRunner(host=host) if host else RemoteRunner()
        remote.preflight()
        return remote, f"Running real OpenSn on {remote.host}."
    if kind == "fake":
        return (
            FakeRunner(Path(fixtures) if fixtures else DEFAULT_FIXTURES),
            "Replaying recorded output. Nothing was executed.",
        )
    if kind == "local":
        local = LocalMPIRunner()
        local.preflight()
        return local, "Running real OpenSn on this machine."
    raise UnknownRunner(
        f"unknown runner {kind!r}; expected one of {', '.join(RUNNER_KINDS)}"
    )
