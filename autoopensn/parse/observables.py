"""Extracting observables from OpenSn stdout.

Every pattern here was read out of the OpenSn source at commit
2fd4a19ceade4f8da581a5029c68f474d3333ccb. The citations are kept in the code
because they are the only thing that makes these regular expressions auditable:

``WGS groups [0-0] iteration = 12, residual = 3.2e-07``
    ``modules/linear_boltzmann_solvers/discrete_ordinates_problem/
    iterative_methods/wgs_convergence_test.cc`` for the PETSc methods, and
    ``.../classic_richardson.cc`` for classic Richardson. Both build the line
    with ``AppendNumericField``, which writes ``", " + label + " = " + value``
    (``framework/logging/log_format.h``).

``AGS iteration = 0, l2_change = 1.2e-08``
    ``.../iterative_methods/ags_linear_solver.cc``.

``Balance table:`` followed by ``" Balance                     = 3.55e-15"``
    ``.../compute/discrete_ordinates_compute.cc``, printed when the solver is
    constructed with ``compute_balance=True``.

``Absorption=1.006178e+02``
    Printed by the Reed regression script itself, in the ``Key=value`` form the
    OpenSn test harness checks with ``KeyValuePairCheck``
    (``test/src/checks.py``).

**On the sweep count.** At this commit OpenSn does not print the number of
transport sweeps for a steady-state fixed-source solve. It accumulates
``num_sweeps`` and prints only the derived ``avg_sweep_time``
(``ags_linear_solver.cc``). An explicit ``sweeps = N`` line exists, but only on
the k-eigenvalue path (``FormatKEigenFinalSummary``). So this module prefers
that line when present and otherwise derives the count: each inner iteration of
a groupset solve applies the inverse transport operator exactly once, for both
the Krylov methods and classic Richardson, so the sweep count is the total inner
iteration count. ``Observations.sweeps_source`` records which route was taken,
and it belongs in any table where the number matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?"

WGS_ITERATION = re.compile(
    r"WGS groups \[(?P<first>\d+)-(?P<last>\d+)\]\s+iteration\s*=\s*(?P<index>\d+)"
)
AGS_ITERATION = re.compile(r"AGS iteration\s*=\s*(?P<index>\d+)")

# The summary OpenSn prints when a solve ends without converging, from
# FormatIterationSummary in lbs_problem/iterative_methods/iteration_logging.h:
#   WGS groups [0-0] final, status = iteration_limit, iterations = 300, ...
# Without this a run that hit its cap parses as having no status at all, which
# reads in a table as "we do not know" rather than "it did not converge".
WGS_FINAL = re.compile(
    r"WGS groups \[\d+-\d+\] final, status = (?P<status>\w+)"
    r"(?:, iterations = (?P<iterations>\d+))?"
)
STATUS = re.compile(r"status\s*=\s*(?P<status>converged|iteration_limit|failed|not_run)")
# Real OpenSn prefixes every log line with the MPI rank, "[0]  ". The pattern
# was first written against hand-made fixtures that left the prefix out, and
# anchored at the start of the line it could never match a real run. Found the
# first time AutoOpenSn read output from an actual OpenSn build.
RANK_PREFIX = r"(?:\[\d+\]\s*)?"
BALANCE_LINE = re.compile(
    rf"^{RANK_PREFIX}\s*Balance\s*=\s*(?P<value>{NUMBER})\s*$", re.MULTILINE
)
SWEEPS_LOGGED = re.compile(rf"\bsweeps\s*=\s*(?P<value>\d+)")
AVG_SWEEP_TIME = re.compile(rf"avg_sweep_time\s*=\s*(?P<value>{NUMBER})\s*s\b")


class ParseError(Exception):
    """Output did not contain something the caller required."""


def _to_float(text: str) -> float:
    """Parse a number, tolerating Fortran-style ``d`` exponents."""
    return float(text.replace("d", "e").replace("D", "E"))


def key_value(text: str, key: str) -> Optional[float]:
    """The number immediately following ``key``, using the last occurrence.

    This reproduces the OpenSn regression harness's own rule
    (``KeyValuePairCheck`` in ``test/src/checks.py``): find the key anywhere in
    a line, take the first whitespace-delimited word after it, and if the key
    appears more than once, the last one wins. Matching that rule exactly is
    what lets gold values be compared without a second interpretation of the
    same log.
    """
    found: Optional[float] = None
    for line in text.splitlines():
        position = line.find(key)
        if position < 0:
            continue
        remainder = line[position + len(key) :].strip()
        if not remainder:
            continue
        word = remainder.split()[0].strip().rstrip(",;")
        try:
            found = _to_float(word)
        except ValueError:
            continue
    return found


@dataclass
class Observations:
    """Everything read out of one run's stdout."""

    avg_flux: Optional[float] = None
    sweeps: Optional[int] = None
    sweeps_source: str = "none"
    """``logged`` when OpenSn printed it, ``derived`` when inferred, ``none``."""
    iterations: Optional[int] = None
    ags_iterations: Optional[int] = None
    balance_residual: Optional[float] = None
    absorption_rate: Optional[float] = None
    outflow_rate: Optional[float] = None
    inner_status: Optional[str] = None
    avg_sweep_time: Optional[float] = None
    converged: Optional[bool] = None
    solve_blocks: tuple[int, ...] = field(default_factory=tuple)
    """Inner iteration count of each groupset solve, in order."""

    def as_row(self) -> dict:
        return {
            "avg_flux": self.avg_flux,
            "sweeps": self.sweeps,
            "sweeps_source": self.sweeps_source,
            "iterations": self.iterations,
            "ags_iterations": self.ags_iterations,
            "balance_residual": self.balance_residual,
            "absorption_rate": self.absorption_rate,
            "outflow_rate": self.outflow_rate,
            "inner_status": self.inner_status,
            "converged": self.converged,
        }


def _inner_solve_blocks(text: str) -> tuple[list[int], Optional[str]]:
    """Inner iteration counts per groupset solve, and the last status seen.

    A solve is a run of ``iteration = n`` lines with non-decreasing ``n``. The
    index resets when the next solve starts, which is how separate solves are
    told apart without the log naming them.
    """
    blocks: list[int] = []
    current: Optional[int] = None
    last_status: Optional[str] = None

    for line in text.splitlines():
        match = WGS_ITERATION.search(line)
        if match is None:
            continue
        index = int(match.group("index"))
        if current is not None and index <= current:
            blocks.append(current)
            current = index
        else:
            current = index
        status = STATUS.search(line)
        if status:
            last_status = status.group("status")

    if current is not None:
        blocks.append(current)
    return blocks, last_status


def parse_stdout(text: str) -> Observations:
    """Read every observable this module knows about out of one run's stdout."""
    blocks, status = _inner_solve_blocks(text)

    # A final summary line is more authoritative than the per-iteration lines:
    # it is printed precisely when the solve ended badly, and it says so.
    final = None
    for match in WGS_FINAL.finditer(text):
        final = match
    if final is not None:
        status = final.group("status")
        if final.group("iterations") and not blocks:
            blocks = [int(final.group("iterations"))]

    observations = Observations(
        avg_flux=key_value(text, "AvgFlux="),
        absorption_rate=key_value(text, "Absorption="),
        outflow_rate=key_value(text, "OutFlow="),
        solve_blocks=tuple(blocks),
        inner_status=status,
    )

    if blocks:
        observations.iterations = sum(blocks)

    ags = [int(m.group("index")) for m in AGS_ITERATION.finditer(text)]
    if ags:
        observations.ags_iterations = len(ags)

    logged = SWEEPS_LOGGED.search(text)
    if logged:
        observations.sweeps = int(logged.group("value"))
        observations.sweeps_source = "logged"
    elif blocks:
        # One application of the inverse transport operator per inner iteration.
        observations.sweeps = sum(blocks)
        observations.sweeps_source = "derived"

    balance = BALANCE_LINE.search(text)
    if balance:
        observations.balance_residual = _to_float(balance.group("value"))

    sweep_time = AVG_SWEEP_TIME.search(text)
    if sweep_time:
        observations.avg_sweep_time = _to_float(sweep_time.group("value"))

    if status is not None:
        observations.converged = status == "converged"

    return observations
