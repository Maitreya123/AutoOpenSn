"""Turning run output into a table of numbers.

This is the stage the language model is kept away from. Deciding what happened
in a run is exactly the job where a plausible-sounding summary of a log is worse
than no summary at all, so it is done here, with regular expressions matched
against log formats read out of the OpenSn source at the pinned commit.

Five observables, as the brief specifies:

============ ==========================================================
avg_flux     Volume-averaged scalar flux, printed by the template as
             ``AvgFlux=``.
sweeps       Transport sweeps. Preferred from an explicit ``sweeps = N``
             log line; otherwise derived from the inner-iteration log.
iterations   Inner (within-groupset) iterations.
wall_time    Seconds, measured by the runner, not read from the log.
balance      Particle balance residual, from the printed balance table.
============ ==========================================================

Each parser records where its format came from, so that a log format change
upstream is a failed test with a file and a line number rather than a column of
silent NaNs.
"""

from autoopensn.parse.gold import GoldComparison, compare_to_gold, load_gold
from autoopensn.parse.observables import (
    Observations,
    ParseError,
    key_value,
    parse_stdout,
)
from autoopensn.parse.table import (
    convergence_table,
    observe,
    results_table,
)

__all__ = [
    "GoldComparison",
    "Observations",
    "ParseError",
    "compare_to_gold",
    "convergence_table",
    "key_value",
    "load_gold",
    "observe",
    "parse_stdout",
    "results_table",
]
