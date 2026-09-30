"""The Streamlit interface.

A thin front end over the same functions the command line calls. Nothing in
here computes anything: it loads specs, calls ``run_study``, and draws what
comes back. If a number appears on a page and not in the results table, that is
a bug in this package, not a feature of the interface.
"""

from pathlib import Path

APP_PATH = Path(__file__).resolve().parent / "app.py"

__all__ = ["APP_PATH"]
