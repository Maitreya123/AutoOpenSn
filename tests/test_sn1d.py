"""Tests for the reference 1D transport solver.

The solver is not OpenSn, so the question that matters is whether it agrees with
OpenSn where OpenSn has published an answer. Its regression suite records those
answers, and the first two tests are the whole justification for shipping this:
if they fail, the solver is solving a different problem and its numbers should
not be shown to anyone.
"""

from __future__ import annotations

import numpy as np
import pytest

from autoopensn.solvers.sn1d import (
    Problem,
    SlabSolver,
    SolverError,
    gmres,
    solve,
    source_iteration,
)

REED = dict(
    widths=[2.0, 1.0, 2.0, 1.0, 2.0],
    cells_per_region=[200] * 5,
    sigma_t=[50.0, 5.0, 0.0, 1.0, 1.0],
    scattering_ratio=[0.0, 0.0, 0.0, 0.9, 0.9],
    source=[50.0, 0.0, 0.0, 1.0, 0.0],
    n_polar=128,
)

FOUNDATION = dict(
    widths=[4.0],
    cells_per_region=[40],
    sigma_t=[1.0],
    scattering_ratio=[0.0],
    source=[1.0],
    n_polar=8,
)


# --- agreement with OpenSn's recorded answers -------------------------------


def test_reed_matches_the_opensn_gold_values():
    """OpenSn's tests.json records Absorption=100.6178 and OutFlow=0.3821562.

    Compared at the precision the tutorial prints, which is the precision the
    regression harness itself compares at.
    """
    solution = solve(Problem(**REED), method="petsc_gmres", tolerance=1.0e-10, max_iterations=300)
    assert f"{solution.absorption_rate:.6e}" == "1.006178e+02"
    assert f"{solution.outflow_rate:.6e}" == "3.821562e-01"


def test_foundation_matches_its_gold_value():
    """FOUNDATION_1D_MAX_FLUX = 0.962393909, abs_tol 1e-8 in tests.json."""
    solution = solve(
        Problem(**FOUNDATION), method="petsc_gmres", tolerance=1.0e-10, max_iterations=100
    )
    assert solution.max_flux == pytest.approx(0.962393909, abs=1.0e-8)


def test_particle_balance_closes():
    """Production in equals absorption plus leakage out. The physics must balance."""
    solution = solve(Problem(**REED), method="petsc_gmres", tolerance=1.0e-10)
    assert solution.production_rate == pytest.approx(101.0, abs=1e-9)
    assert abs(solution.balance) < 1.0e-9


def test_both_methods_reach_the_same_answer():
    """A different iterative method is a different route to the same solution."""
    krylov = solve(Problem(**REED), method="petsc_gmres", tolerance=1.0e-10, max_iterations=300)
    fixed_point = solve(
        Problem(**REED), method="classic_richardson", tolerance=1.0e-10, max_iterations=2000
    )
    assert krylov.absorption_rate == pytest.approx(fixed_point.absorption_rate, rel=1e-7)
    assert krylov.outflow_rate == pytest.approx(fixed_point.outflow_rate, rel=1e-6)


# --- an analytic case -------------------------------------------------------


def test_a_symmetric_problem_gives_a_symmetric_flux():
    """A slab symmetric about its centre must solve to a symmetric flux.

    Cheap, and it catches a whole class of sweep bugs: a direction handled
    differently from its mirror shows up here immediately.
    """
    problem = Problem(
        widths=[1.0, 2.0, 1.0], cells_per_region=[50, 100, 50],
        sigma_t=[1.0, 0.5, 1.0], scattering_ratio=[0.5, 0.8, 0.5],
        source=[1.0, 0.0, 1.0], n_polar=16,
    )
    solution = solve(problem, method="petsc_gmres", tolerance=1.0e-12, max_iterations=300)
    flux = solution.cell_average_flux
    assert flux == pytest.approx(flux[::-1], rel=1e-9)


def test_refining_the_mesh_converges():
    """The answer must stop moving as the cells shrink, or it is not a solution."""
    def absorption(cells):
        problem = Problem(
            widths=[4.0], cells_per_region=[cells], sigma_t=[1.0],
            scattering_ratio=[0.5], source=[1.0], n_polar=32,
        )
        return solve(problem, method="petsc_gmres", tolerance=1e-12).absorption_rate

    coarse, medium, fine = absorption(20), absorption(40), absorption(80)
    assert abs(medium - fine) < abs(coarse - fine)


def test_a_reflecting_boundary_is_refused_rather_than_approximated():
    with pytest.raises(SolverError) as excinfo:
        Problem(widths=[1.0], cells_per_region=[10], sigma_t=[1.0],
                scattering_ratio=[0.0], source=[1.0], left_boundary="reflecting")
    assert "vacuum only" in str(excinfo.value)


# --- convergence behaviour --------------------------------------------------


def test_a_tighter_tolerance_costs_more_sweeps():
    counts = [
        solve(Problem(**REED), method="petsc_gmres", tolerance=tol).sweeps
        for tol in (1e-4, 1e-6, 1e-8)
    ]
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_a_small_restart_interval_costs_more_sweeps():
    """The property the GMRES convergence study exists to measure.

    This once failed silently: the breakdown test read a Hessenberg entry that
    the Givens rotation had already zeroed, so every cycle stopped after one
    inner iteration and the restart interval changed nothing.
    """
    solver = SlabSolver(Problem(**REED))
    tight = gmres(solver, 1.0e-9, 500, restart=3)[1]
    loose = gmres(solver, 1.0e-9, 500, restart=30)[1]
    assert tight > loose


def test_gmres_beats_source_iteration_on_this_problem():
    solver = SlabSolver(Problem(**REED))
    krylov = gmres(solver, 1.0e-9, 500, restart=30)[1]
    fixed_point = source_iteration(solver, 1.0e-9, 2000)[1]
    assert krylov < fixed_point


def test_the_residual_actually_falls_below_the_tolerance():
    """A converged flag that is not backed by the true residual is a lie."""
    solver = SlabSolver(Problem(**REED))
    phi, sweeps, converged, _ = gmres(solver, 1.0e-9, 500, restart=30)
    assert converged
    rhs = solver.uncollided()
    residual = np.linalg.norm(rhs - solver.apply_operator(phi)) / np.linalg.norm(rhs)
    assert residual < 1.0e-9


def test_hitting_the_iteration_cap_is_reported_not_hidden():
    solution = solve(Problem(**REED), method="classic_richardson", tolerance=1e-14, max_iterations=3)
    assert not solution.converged
    assert solution.sweeps == 3


# --- refusals ---------------------------------------------------------------


def test_a_multiplying_medium_is_refused():
    with pytest.raises(SolverError):
        Problem(widths=[1.0], cells_per_region=[10], sigma_t=[1.0],
                scattering_ratio=[1.5], source=[1.0])


def test_an_odd_polar_count_is_refused():
    with pytest.raises(SolverError):
        Problem(widths=[1.0], cells_per_region=[10], sigma_t=[1.0],
                scattering_ratio=[0.0], source=[1.0], n_polar=7)


def test_mismatched_region_lists_are_refused():
    with pytest.raises(SolverError):
        Problem(widths=[1.0, 2.0], cells_per_region=[10], sigma_t=[1.0],
                scattering_ratio=[0.0], source=[1.0])


def test_a_negative_cross_section_is_refused():
    with pytest.raises(SolverError):
        Problem(widths=[1.0], cells_per_region=[10], sigma_t=[-1.0],
                scattering_ratio=[0.0], source=[1.0])


def test_an_unknown_method_is_refused():
    with pytest.raises(SolverError):
        solve(Problem(**FOUNDATION), method="magic")


def test_bicgstab_says_it_is_not_implemented_rather_than_running_gmres():
    """Claiming to be a method while running another misreports the study."""
    with pytest.raises(SolverError) as excinfo:
        solve(Problem(**FOUNDATION), method="petsc_bicgstab")
    assert "not implemented" in str(excinfo.value)


def test_an_unknown_boundary_is_refused():
    with pytest.raises(SolverError):
        Problem(widths=[1.0], cells_per_region=[10], sigma_t=[1.0],
                scattering_ratio=[0.0], source=[1.0], left_boundary="isotropic")


def test_the_reed_template_defaults_are_solvable():
    """The scenario the runner is pointed at must actually be in range."""
    from autoopensn.templates import load_template
    from autoopensn.runner.reference import ADAPTERS

    template = load_template("reed_1d")
    problem = ADAPTERS["reed_1d"].build(template.defaults())
    assert problem.n_polar == 128
    assert len(problem.widths) == 5


# --- speed ------------------------------------------------------------------


def test_the_reed_problem_solves_fast_enough_to_be_interactive():
    """A thousand cells and a hundred and twenty-eight angles, under two seconds."""
    import time

    started = time.monotonic()
    solve(Problem(**REED), method="petsc_gmres", tolerance=1.0e-9)
    assert time.monotonic() - started < 2.0
