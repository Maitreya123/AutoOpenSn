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

from pathlib import Path
from typing import Optional

from autoopensn.runner.base import Runner, RunnerError
from autoopensn.runner.fake import FakeRunner
from autoopensn.runner.local_mpi import LocalMPIRunner
from autoopensn.runner.reference import ReferenceRunner
from autoopensn.runner.remote import PROBE_TIMEOUT, RemoteRunner

DEFAULT_FIXTURES = (
    Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "reed_gmres"
)

RUNNER_KINDS = ("auto", "reference", "remote", "fake", "local")


class UnknownRunner(ValueError):
    """A runner name that is not one of ``RUNNER_KINDS``."""


def choose_runner(host: Optional[str] = None) -> tuple[Runner, str]:
    """The cluster if it answers, this machine's own solver if it does not.

    The probe is short because someone is waiting on it; the runner that
    actually runs gets the ordinary, patient timeout.
    """
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
        f"unknown runner {kind!r}; expected 'auto', 'reference', 'remote', "
        "'fake' or 'local'"
    )
