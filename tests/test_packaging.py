"""Every data file the package reads must ship in the package.

An editable install reads data straight from the source tree, so a file left
out of ``package-data`` is invisible to every test and every developer, and
missing for everyone who installs from GitHub. That is how ``catalog.json``
went missing: a plain install had all 41 templates and an empty scenario
catalogue, and the MCP server's ``list_scenarios`` returned nothing.
"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "autoopensn"


def _tracked_data_files() -> set[str]:
    try:
        listed = subprocess.run(
            ["git", "ls-files", "autoopensn"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return {
        str(Path(name).relative_to("autoopensn"))
        for name in listed
        if not name.endswith(".py")
    }


def test_every_data_file_is_in_package_data():
    config = tomllib.loads((ROOT / "pyproject.toml").read_text())
    patterns = config["tool"]["setuptools"]["package-data"]["autoopensn"]
    shipped = {
        str(path.relative_to(PACKAGE))
        for pattern in patterns
        for path in PACKAGE.glob(pattern)
        if path.is_file()
    }
    missing = sorted(_tracked_data_files() - shipped)
    assert not missing, f"not in [tool.setuptools.package-data]: {missing}"
