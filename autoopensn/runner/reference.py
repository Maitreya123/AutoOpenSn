"""Running a scenario with AutoOpenSn's own solver, instead of OpenSn.

**Results from this runner are not OpenSn results.** They come from
``autoopensn.solvers.sn1d``, an independent implementation of the same
equations. Every result it produces is labelled, the interface says so on the
page, and the log it writes carries a banner saying it at the top. The point of
this repository is that a plausible number attributed to the wrong source is
worse than no number, and that applies to numbers we compute ourselves.

It exists because OpenSn cannot reasonably be built on every machine that wants
to look at a study. Compiling it means compiling PETSc, VTK and HDF5 first,
which is hours of work and more free disk than a laptop usually has. Without
something like this, a generated study on such a machine renders its scripts
and stops, which is a poor answer to "show me the result".

What makes it worth trusting within its range is that OpenSn's own regression
suite records the answers, and this reproduces them:

    scenario                 quantity                  this solver     OpenSn gold
    reed_1d                  Absorption=               1.006178e+02    100.6178
    reed_1d                  OutFlow=                  3.821562e-01    0.3821562
    first_1d_fixed_source    FOUNDATION_1D_MAX_FLUX=   9.62393909e-01  0.962393909

``tests/test_reference_runner.py`` asserts all three, so a change that breaks the
agreement fails the suite rather than quietly producing different physics.

**It refuses anything it cannot solve.** One dimension, one group, isotropic
scattering. Every other scenario raises, naming what is unsupported, rather than
approximating it. A solver that silently answers a question it does not
understand is the exact failure this project is arranged to prevent.

The output is written in OpenSn's own log format, so the parser, the tables, the
gold comparison and the narration all work on it unchanged. That is deliberate:
the rest of the pipeline should not know or care which solver produced a run,
and the one thing it must know, that this was not OpenSn, travels in the runner
name and the banner rather than in a special case somewhere downstream.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from autoopensn.runner.base import RunRequest, RunResult, Runner, RunnerError
from autoopensn.solvers.sn1d import Problem, Solution, SolverError, solve

BANNER = (
    "AutoOpenSn reference solver. THIS IS NOT OpenSn: the numbers below were "
    "computed by autoopensn.solvers.sn1d, an independent 1D discrete-ordinates "
    "implementation, and validated against the values OpenSn's regression suite "
    "records for this scenario."
)


class UnsupportedScenario(RunnerError):
    """This scenario is outside what the reference solver can solve."""


@dataclass(frozen=True)
class Adapter:
    """How one template's parameters become a problem, and what it prints."""

    template: str
    build: Callable[[dict[str, Any]], Problem]
    report: Callable[[dict[str, Any], Solution], list[tuple[str, str]]]
    balance_table: bool = True


def _reed_problem(p: dict[str, Any]) -> Problem:
    return Problem(
        widths=list(p["widths"]),
        cells_per_region=list(p["nrefs"]),
        sigma_t=list(p["total"]),
        scattering_ratio=list(p["c"]),
        # The Reed benchmark puts a source in the first and fourth regions only.
        source=[
            float(p["src_block0_strength"]), 0.0, 0.0,
            float(p["src_block3_strength"]), 0.0,
        ],
        n_polar=int(p["n_polar"]),
        left_boundary=str(p["bc_zmin_type"]),
        right_boundary=str(p["bc_zmax_type"]),
    )


def _reed_report(p: dict[str, Any], s: Solution) -> list[tuple[str, str]]:
    lines = [
        ("Absorption=", f"{s.absorption_rate:.6e}"),
        ("OutFlow=", f"{s.outflow_rate:.6e}"),
    ]
    if p.get("emit_avg_flux"):
        lines.append(("AvgFlux=", f"{s.average_flux:.9e}"))
    return lines


def _foundation_problem(p: dict[str, Any]) -> Problem:
    strength = p["group_strength"]
    return Problem(
        widths=[float(p["length"])],
        cells_per_region=[int(p["num_cells"])],
        sigma_t=[float(p["sigma_t"])],
        scattering_ratio=[float(p["c"])],
        source=[float(strength[0] if isinstance(strength, (list, tuple)) else strength)],
        n_polar=int(p["n_polar"]),
    )


def _foundation_report(p: dict[str, Any], s: Solution) -> list[tuple[str, str]]:
    return [("FOUNDATION_1D_MAX_FLUX=", f"{s.max_flux:.8e}")]


ADAPTERS: dict[str, Adapter] = {
    "reed_1d": Adapter("reed_1d", _reed_problem, _reed_report, balance_table=True),
    "first_1d_fixed_source": Adapter(
        "first_1d_fixed_source", _foundation_problem, _foundation_report, balance_table=False
    ),
}


def supported(template: str) -> bool:
    return template in ADAPTERS


def _timestamp(seconds: float) -> str:
    minutes, seconds = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:04.1f}"


def render_log(
    solution: Solution,
    reported: list[tuple[str, str]],
    *,
    balance_table: bool,
    seconds: float,
) -> str:
    """The solve, written in the log format OpenSn uses.

    Not imitation for its own sake. The parser, the gold comparison and the
    tables were written against OpenSn's formats and tested against them, and
    emitting the same shapes means none of that needs a second code path for
    this runner.
    """
    lines = [f"[AutoOpenSn] {BANNER}", ""]

    clock = 0.05
    step = max(seconds / max(len(solution.residuals), 1), 0.001)
    for index, residual in enumerate(solution.residuals):
        line = (
            f"{_timestamp(clock)} WGS groups [0-0] iteration = {index}"
            f", residual = {residual:.6e}"
        )
        if index == len(solution.residuals) - 1 and solution.converged:
            line += ", status = converged"
        lines.append(line)
        clock += step

    status = "converged" if solution.converged else "iteration_limit"
    if not solution.converged:
        # The summary OpenSn prints when a solve ends at its cap. Emitted here
        # for the same reason the rest of the log is: so that a table built from
        # this run says "did not converge" rather than leaving the column blank.
        lines.append(
            f"{_timestamp(clock)} WGS groups [0-0] final, status = {status}"
            f", iterations = {solution.iterations}"
        )
    lines.append(
        f"{_timestamp(clock)} AGS iteration = 0, l2_change = {max(solution.residuals or [0.0]):.6e}"
        f", status = {status}; WGS = {status}"
    )

    if balance_table:
        lines += [
            "",
            "Balance table:",
            f" Absorption rate             = {solution.absorption_rate:.6e}",
            f" Production rate             = {solution.production_rate:.6e}",
            f" In-flow rate                = {solution.inflow_rate:.6e}",
            f" Out-flow rate               = {solution.outflow_rate:.6e}",
            f" Balance                     = {solution.balance:.6e}",
            "",
        ]

    for key, value in reported:
        lines.append(f"{key}{value}")

    return "\n".join(lines) + "\n"


class ReferenceRunner(Runner):
    """Solves a supported scenario directly, without OpenSn."""

    name = "reference"

    def __init__(self, *, strict: bool = True):
        # ``strict`` exists so a study over mixed scenarios can report the
        # unsupported ones as failed rows rather than stopping. It never makes
        # the solver guess at a problem it does not handle.
        self.strict = strict

    def adapter_for(self, request: RunRequest) -> Adapter:
        template = request.template or self._infer(request)
        if template not in ADAPTERS:
            raise UnsupportedScenario(
                f"the reference solver handles {', '.join(sorted(ADAPTERS))} and not "
                f"{template or 'this scenario'}. It solves one-dimensional, one-group "
                f"problems with isotropic scattering; anything else needs OpenSn itself."
            )
        return ADAPTERS[template]

    @staticmethod
    def _infer(request: RunRequest) -> str:
        """Identify the scenario from the parameters present.

        The request carries parameters rather than a template name, so the
        template is recovered from the shape of the parameter set. Matching on a
        distinctive required parameter, rather than on a subset, means a
        scenario that merely resembles a supported one is refused instead of
        being solved as the wrong problem.
        """
        names = set(request.parameters)
        if {"widths", "nrefs", "total", "src_block0_strength"} <= names:
            return "reed_1d"
        if {"num_cells", "length", "sigma_t", "group_strength"} <= names:
            return "first_1d_fixed_source"
        return ""

    def run(self, request: RunRequest) -> RunResult:
        started = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        request.write_inputs()
        parameters = dict(request.parameters)

        try:
            adapter = self.adapter_for(request)
            problem = adapter.build(parameters)
            solution = solve(
                problem,
                method=str(parameters.get("inner_linear_method", "petsc_gmres")),
                tolerance=float(parameters.get("l_abs_tol", 1.0e-9)),
                max_iterations=int(parameters.get("l_max_its", 300)),
                restart=int(parameters.get("gmres_restart_interval", 30)),
            )
        except (UnsupportedScenario, SolverError) as exc:
            if self.strict and isinstance(exc, UnsupportedScenario):
                raise
            elapsed = time.monotonic() - started
            result = RunResult(
                case_id=request.case_id,
                exit_code=2,
                stdout=f"[AutoOpenSn] {BANNER}\n",
                stderr=f"{type(exc).__name__}: {exc}\n",
                wall_time=elapsed,
                directory=Path(request.directory),
                runner=self.name,
                started_at=started_at,
            )
            result.persist()
            return result

        elapsed = time.monotonic() - started
        stdout = render_log(
            solution,
            adapter.report(parameters, solution),
            balance_table=adapter.balance_table,
            seconds=elapsed,
        )

        result = RunResult(
            case_id=request.case_id,
            exit_code=0,
            stdout=stdout,
            stderr="",
            wall_time=elapsed,
            directory=Path(request.directory),
            runner=self.name,
            command=("<reference solver>", f"{solution.method}"),
            replayed=False,
            started_at=started_at,
        )
        result.persist()
        return result

    def config(self) -> dict[str, Any]:
        # The solver is part of the identity of a result. A number from here and
        # a number from OpenSn are different measurements and the cache must not
        # treat them as interchangeable.
        return {"runner": self.name, "solver": "sn1d", "version": 1}
