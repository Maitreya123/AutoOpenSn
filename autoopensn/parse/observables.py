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

# Performance and problem size, as OpenSn reports them about itself. Every one
# was read off a real run on class01 (hdpe_example, eight ranks, 172 groups);
# the hand-written fixtures had none of them, which is why none were parsed.
SWEEP_TIME_PER_UNKNOWN = re.compile(
    rf"sweep_time_per_unknown\s*=\s*(?P<value>{NUMBER})\s*ns\b"
)
UNKNOWNS = re.compile(r"(?<![A-Za-z_])unknowns\s*=\s*(?P<value>\d+)")
MPI_RANKS = re.compile(r"Running OpenSn with (?P<value>\d+) process")
PEAK_MEMORY = re.compile(rf"Total peak host memory usage:\s*(?P<value>{NUMBER})\s*GB")
ELAPSED = re.compile(r"Elapsed execution time:\s*(?P<h>\d+):(?P<m>\d+):(?P<s>\d+(?:\.\d+)?)")
CELLS = re.compile(r"Global cell count\s*:\s*(?P<value>\d+)")
GROUPS = re.compile(r"Number of groups\s*:\s*(?P<value>\d+)")
ANGLES = re.compile(r"Number of angles\s*:\s*(?P<value>\d+)")
SOLVER_CLOCK = re.compile(
    r"(?P<h>\d+):(?P<m>\d+):(?P<s>\d+(?:\.\d+)?) (?P<event>Starting|Finished) solver execution"
)


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
    """Seconds per transport sweep, averaged by OpenSn."""
    sweep_time_per_unknown_ns: Optional[float] = None
    """Nanoseconds per unknown per sweep: the figure that compares across sizes."""
    unknowns: Optional[int] = None
    mpi_ranks: Optional[int] = None
    peak_memory_gb: Optional[float] = None
    opensn_elapsed: Optional[float] = None
    """OpenSn's own total, in seconds. Printed to a tenth of a second."""
    solve_time: Optional[float] = None
    """Seconds between "Starting" and "Finished solver execution", summed over
    solvers. Same tenth-of-a-second resolution as the log's clock."""
    cells: Optional[int] = None
    groups: Optional[int] = None
    angles: Optional[int] = None
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
            "avg_sweep_time": self.avg_sweep_time,
            "sweep_time_per_unknown_ns": self.sweep_time_per_unknown_ns,
            "unknowns": self.unknowns,
            "mpi_ranks": self.mpi_ranks,
            "peak_memory_gb": self.peak_memory_gb,
            "opensn_elapsed": self.opensn_elapsed,
            "solve_time": self.solve_time,
            "cells": self.cells,
            "groups": self.groups,
            "angles": self.angles,
        }


def _inner_solve_blocks(text: str) -> tuple[list[int], Optional[str]]:
    """Sweeps in each groupset solve, and the last status seen.

    A solve is a run of ``iteration = n`` lines with non-decreasing ``n``. The
    index resets when the next solve starts, which is how separate solves are
    told apart without the log naming them.

    **The two families of inner solver number their lines differently**, and
    the count has to follow each one:

    - The PETSc Krylov methods (GMRES, BiCGStab, PETSc Richardson) log the
      initial residual as ``iteration = 0``, before any Krylov step, so the
      last index is the number of steps. These lines report ``residual``.
    - OpenSn's own classic Richardson sweeps first and logs after, starting at
      ``iteration = 0`` for the first sweep (``num_iterations = k + 1`` in
      ``classic_richardson.cc``), so it is the last index plus one. These lines
      report ``phi_change``. Counted like the others, a 100-sweep run read 99 —
      found comparing a Richardson run's log with OpenSn's own summary line.
    """
    blocks: list[int] = []
    current: Optional[int] = None
    pointwise = False
    last_status: Optional[str] = None

    def count(index: int, by_sweep: bool) -> int:
        return index + 1 if by_sweep else index

    for line in text.splitlines():
        match = WGS_ITERATION.search(line)
        if match is None:
            continue
        index = int(match.group("index"))
        if current is not None and index <= current:
            blocks.append(count(current, pointwise))
            pointwise = False
        current = index
        pointwise = pointwise or "phi_change" in line
        status = STATUS.search(line)
        if status:
            last_status = status.group("status")

    if current is not None:
        blocks.append(count(current, pointwise))
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

    _read_performance(text, observations)
    return observations


def _clock(h: str, m: str, s: str) -> float:
    return int(h) * 3600 + int(m) * 60 + float(s)


def _read_performance(text: str, observations: Observations) -> None:
    """Fill in the performance and problem-size fields, where OpenSn printed them."""
    match = SWEEP_TIME_PER_UNKNOWN.search(text)
    if match:
        observations.sweep_time_per_unknown_ns = _to_float(match.group("value"))
    # One line per groupset; a problem with several groupsets has the sum.
    # The same groupset solved twice repeats its line, so each distinct value
    # is counted once — imperfect for two equal groupsets, and the better error
    # to make than doubling every repeated solve.
    sizes = {int(m.group("value")) for m in UNKNOWNS.finditer(text)}
    if sizes:
        observations.unknowns = sum(sizes)
    for pattern, name, cast in (
        (MPI_RANKS, "mpi_ranks", int),
        (PEAK_MEMORY, "peak_memory_gb", _to_float),
        (CELLS, "cells", int),
        (GROUPS, "groups", int),
        (ANGLES, "angles", int),
    ):
        found = pattern.search(text)
        if found:
            setattr(observations, name, cast(found.group("value")))
    elapsed = ELAPSED.search(text)
    if elapsed:
        observations.opensn_elapsed = _clock(elapsed.group("h"), elapsed.group("m"), elapsed.group("s"))
    total, started = 0.0, None
    for event in SOLVER_CLOCK.finditer(text):
        at = _clock(event.group("h"), event.group("m"), event.group("s"))
        if event.group("event") == "Starting":
            started = at
        elif started is not None:
            total += max(0.0, at - started)
            started = None
    if total or any(e.group("event") == "Finished" for e in SOLVER_CLOCK.finditer(text)):
        observations.solve_time = round(total, 3)
