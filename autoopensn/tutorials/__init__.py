"""The OpenSn tutorials, treated as a library of runnable scenarios.

Each tutorial is a notebook with prose, a script, and, for nearly all of them, a
recorded correct answer. That makes the tutorial set exactly what this tool
needs: a catalog of problems somebody has already decided are worth solving,
with a way to tell whether a run got the right answer.

Three steps, in order:

``notebook``  read a notebook into a script and its prose
``parameters`` find the knobs in that script, and what they may be set to
``generate``  write a template whose defaults reproduce the script exactly

``catalog`` holds the index that a request is matched against, and ``build``
rebuilds everything from a pinned checkout.
"""

from autoopensn.tutorials.build import BuildReport, rebuild
from autoopensn.tutorials.catalog import Scenario, by_name, load, rank, scan
from autoopensn.tutorials.notebook import Notebook, NotebookError
from autoopensn.tutorials.parameters import Detected, detect

__all__ = [
    "BuildReport",
    "Detected",
    "Notebook",
    "NotebookError",
    "Scenario",
    "by_name",
    "detect",
    "load",
    "rank",
    "rebuild",
    "scan",
]
