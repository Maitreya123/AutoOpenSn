"""Shared fixtures.

No test in this suite requires OpenSn, MPI, or a network. The two that touch
the sister repository skip themselves when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoopensn.spec import load_spec
from autoopensn.templates import load_template

TESTS = Path(__file__).resolve().parent
DATA = TESTS / "data"
FIXTURES = TESTS / "fixtures"


@pytest.fixture(autouse=True)
def no_cluster(monkeypatch):
    """Forbid every test from reaching a real cluster.

    The promise that this suite needs no network used to be kept by accident:
    the cluster was unreachable, so the automatic runner choice always fell back
    to the built-in solver. The day cluster access started working, the same
    tests began submitting real jobs to a shared compute node, and a one-minute
    suite ran for ten.

    So the promise is enforced here instead. Every process the remote runner
    starts goes through ``autoopensn.runner.remote._run_process``; this makes it
    fail as if ssh were missing, which the runner turns into a RunnerError and
    the automatic choice turns into a fallback. A test of the runner's own logic
    patches the same name itself, and its patch, applied later, wins.

    It must be that name and not ``subprocess.run``. Patching the latter
    replaces the function in the one shared ``subprocess`` module, and with it
    every other runner's ability to start a process.
    """

    def refuse(*args, **kwargs):
        raise FileNotFoundError("the test suite does not reach the cluster")

    monkeypatch.setattr("autoopensn.runner.remote._run_process", refuse)
    # And no test waits out a real retry back-off: three refusals would cost
    # forty-three seconds of sleeping in any test that reaches one.
    monkeypatch.setattr("autoopensn.runner.remote._sleep", lambda seconds: None)


@pytest.fixture
def data_dir() -> Path:
    return DATA


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES


@pytest.fixture
def gmres_fixtures() -> Path:
    return FIXTURES / "reed_gmres"


@pytest.fixture
def default_fixtures() -> Path:
    return FIXTURES / "reed_default"


@pytest.fixture
def reed_template():
    return load_template("reed_1d")


@pytest.fixture
def gmres_spec():
    return load_spec(DATA / "gmres_convergence.yaml")


@pytest.fixture
def vendored_reed() -> str:
    """reed_balance.py as committed to this repository, byte for byte."""
    return (DATA / "opensn_reed_balance.py").read_text()


@pytest.fixture
def vendored_reed_cbc() -> str:
    return (DATA / "opensn_reed_balance_cbc.py").read_text()
