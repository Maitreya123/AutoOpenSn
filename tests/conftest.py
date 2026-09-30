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
