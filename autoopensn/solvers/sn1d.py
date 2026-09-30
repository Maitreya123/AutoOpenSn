"""A 1D slab discrete-ordinates solver, in numpy.

**This is not OpenSn.** It is an independent implementation of the same
equations, written so that the 1D scenarios can run in the interface
immediately, on a machine where OpenSn is not installed and cannot reasonably be
built. Everything it produces is labelled as coming from this solver, never from
OpenSn, and the runner that wraps it says so on every result.

The reason it is trustworthy enough to be worth having is that OpenSn's own
regression suite records what the answer should be, and this reproduces those
answers to every published digit. For the Reed benchmark:

    quantity      this solver     OpenSn gold (tests.json)
    Absorption    1.006178e+02    100.6178
    OutFlow       3.821562e-01    0.3821562

Both match at the precision the tutorial prints and the harness checks, and the
particle balance closes to about 1e-12. ``tests/test_sn1d.py`` asserts this, so
a change that breaks the agreement fails the suite rather than quietly producing
different physics.

## What it solves

The one-group, isotropic-scattering, steady-state transport equation in slab
geometry:

    mu d(psi)/dx + sigma_t psi = (sigma_s phi + Q) / 2,   phi = int psi dmu

with vacuum boundaries, piecewise-constant material data, and an isotropic
volumetric source per region.

## The discretizations, and why these ones

**Angle: Gauss-Legendre.** The same quadrature ``GLProductQuadrature1DSlab``
uses, with weights summing to two, which is what makes the scalar flux and the
leakage integrals come out on OpenSn's scale rather than half or twice it.

**Space: linear discontinuous.** OpenSn uses PWLD, which in one dimension is
linear discontinuous Galerkin with upwinding. Getting this right is what makes
the agreement exact rather than approximate: a diamond-difference or step
scheme solves the same equation and lands several digits away, which would look
like a bug in one of the two codes and be neither.

**Iteration: source iteration or GMRES.** Both, because the difference between
them is the thing several of these studies are about. Source iteration is the
plain fixed point, one sweep per iteration. GMRES is a restarted Krylov method
on the same operator, also one sweep per matrix-vector product, which is why a
sweep count is a fair comparison between them.

## What it does not do

One dimension, one energy group, isotropic scattering, vacuum boundaries, no
acceleration. That is
a small corner of what OpenSn does, and the runner refuses any scenario outside
it rather than approximating. Two dimensions, multigroup, anisotropic
scattering, eigenvalue and transient problems all need the real thing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Sequence

import numpy as np

VACUUM = "vacuum"
REFLECTING = "reflecting"

METHODS = ("classic_richardson", "petsc_richardson", "petsc_gmres", "petsc_bicgstab")


class SolverError(Exception):
    """The problem as posed cannot be solved by this solver."""


@dataclass
class Problem:
    """A one-group slab problem, in the terms the tutorials state one."""

    widths: Sequence[float]
    cells_per_region: Sequence[int]
    sigma_t: Sequence[float]
    scattering_ratio: Sequence[float]
    source: Sequence[float]
    n_polar: int = 128
    left_boundary: str = VACUUM
    right_boundary: str = VACUUM

    def __post_init__(self) -> None:
        lengths = {
            len(self.widths),
            len(self.cells_per_region),
            len(self.sigma_t),
            len(self.scattering_ratio),
            len(self.source),
        }
        if len(lengths) != 1:
            raise SolverError(
                "widths, cells, cross sections, scattering ratios and sources must "
                "all have one entry per region"
            )
        if self.n_polar % 2:
            raise SolverError("the polar angle count must be even for a slab quadrature")
        for name, values in (("sigma_t", self.sigma_t), ("source", self.source)):
            if any(v < 0 for v in values):
                raise SolverError(f"{name} cannot be negative")
        if any(not 0.0 <= c <= 1.0 for c in self.scattering_ratio):
            raise SolverError(
                "a scattering ratio outside [0, 1] makes the medium multiplying, "
                "which this solver does not handle"
            )
        for side, value in (("left", self.left_boundary), ("right", self.right_boundary)):
            if value != VACUUM:
                raise SolverError(
                    f"{side} boundary is {value!r}; this solver handles vacuum only. "
                    f"A reflecting boundary feeds the flux leaving one face back in "
                    f"at the same face, which a single sweep does not have yet and "
                    f"which therefore needs an iteration this solver does not do. "
                    f"Refusing is better than returning the answer to a problem with "
                    f"one boundary quietly left open."
                )


@dataclass
class Solution:
    """What a solve produced, in the quantities the tutorials print."""

    scalar_flux_left: np.ndarray
    scalar_flux_right: np.ndarray
    cell_width: np.ndarray
    absorption_rate: float
    production_rate: float
    inflow_rate: float
    outflow_rate: float
    balance: float
    average_flux: float
    max_flux: float
    iterations: int
    sweeps: int
    converged: bool
    residuals: list[float] = field(default_factory=list)
    method: str = "classic_richardson"

    @property
    def cell_average_flux(self) -> np.ndarray:
        return 0.5 * (self.scalar_flux_left + self.scalar_flux_right)


class SlabSolver:
    """Builds the mesh and applies the transport sweep for one problem."""

    def __init__(self, problem: Problem):
        self.problem = problem

        widths = np.asarray(problem.widths, dtype=float)
        counts = np.asarray(problem.cells_per_region, dtype=int)
        if np.any(counts < 1):
            raise SolverError("every region needs at least one cell")

        self.h = np.repeat(widths / counts, counts)
        region = np.repeat(np.arange(len(widths)), counts)
        self.n_cells = self.h.size

        sigma_t = np.asarray(problem.sigma_t, dtype=float)
        ratio = np.asarray(problem.scattering_ratio, dtype=float)
        self.sigma_t = sigma_t[region]
        self.sigma_s = (sigma_t * ratio)[region]
        self.sigma_a = self.sigma_t - self.sigma_s
        self.q_ext = np.asarray(problem.source, dtype=float)[region]

        # Weights sum to two over [-1, 1], which is the convention that makes
        # phi = sum(w psi) the scalar flux rather than half of it.
        self.mu, self.weights = np.polynomial.legendre.leggauss(problem.n_polar)
        self.forward = self.mu > 0
        self.n_angles = problem.n_polar

    # --- the sweep ----------------------------------------------------------

    def sweep(self, q_left: np.ndarray, q_right: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
        """One transport sweep: invert the streaming-plus-collision operator.

        Vectorised across directions and sequential across cells, which is the
        only way round: the spatial recurrence is inherently ordered, and the
        directions are independent. That makes one sweep a few thousand small
        vector operations rather than a hundred thousand scalar ones.
        """
        half = self.n_angles // 2
        mu_pos = self.mu[self.forward]
        a = np.abs(mu_pos)

        phi_left = np.zeros(self.n_cells)
        phi_right = np.zeros(self.n_cells)
        weights_pos = self.weights[self.forward]

        exiting = {}

        for direction in (+1, -1):
            # Vacuum only: nothing enters through either face. Problem.__post_init__
            # refuses anything else, so there is no lagged boundary term here and
            # the operator the Krylov solver sees stays linear.
            incoming = np.zeros(half)

            order = range(self.n_cells) if direction > 0 else range(self.n_cells - 1, -1, -1)
            upwind = np.zeros((self.n_cells, half))
            downwind = np.zeros((self.n_cells, half))

            for i in order:
                h = self.h[i]
                s = self.sigma_t[i]
                # Source moments of a linear source over the cell, arranged so
                # that "upwind" is the face the sweep enters through.
                if direction > 0:
                    b1 = h * (2.0 * q_left[i] + q_right[i]) / 6.0
                    b2 = h * (q_left[i] + 2.0 * q_right[i]) / 6.0
                else:
                    b1 = h * (2.0 * q_right[i] + q_left[i]) / 6.0
                    b2 = h * (q_right[i] + 2.0 * q_left[i]) / 6.0

                a11 = a / 2.0 + s * h / 3.0
                a12 = a / 2.0 + s * h / 6.0
                a21 = -a / 2.0 + s * h / 6.0
                a22 = a / 2.0 + s * h / 3.0
                determinant = a11 * a22 - a12 * a21

                rhs1 = b1 + a * incoming
                rhs2 = np.full(half, b2)

                u = (rhs1 * a22 - a12 * rhs2) / determinant
                v = (a11 * rhs2 - rhs1 * a21) / determinant

                upwind[i] = u
                downwind[i] = v
                incoming = v

            if direction > 0:
                phi_left += upwind @ weights_pos
                phi_right += downwind @ weights_pos
                exiting["right"] = incoming
            else:
                phi_right += upwind @ weights_pos
                phi_left += downwind @ weights_pos
                exiting["left"] = incoming

        return phi_left, phi_right, exiting

    def scattering_source(self, phi_left: np.ndarray, phi_right: np.ndarray):
        """The within-group scattering source, halved for the slab convention."""
        return 0.5 * self.sigma_s * phi_left, 0.5 * self.sigma_s * phi_right

    def fixed_source(self):
        return 0.5 * self.q_ext, 0.5 * self.q_ext

    # --- operators ----------------------------------------------------------

    def apply_operator(self, phi: np.ndarray) -> np.ndarray:
        """``(I - D L^-1 M S) phi``, the operator the Krylov methods solve on.

        One sweep per application, which is what makes a sweep count a fair
        comparison between the iterative methods.
        """
        left, right = np.split(phi, 2)
        q_left, q_right = self.scattering_source(left, right)
        swept_left, swept_right, _ = self.sweep(q_left, q_right)
        return phi - np.concatenate([swept_left, swept_right])

    def uncollided(self) -> np.ndarray:
        """``D L^-1 M q``: the flux from the fixed source alone, one sweep."""
        q_left, q_right = self.fixed_source()
        left, right, _ = self.sweep(q_left, q_right)
        return np.concatenate([left, right])


# --- iterative methods ------------------------------------------------------


def source_iteration(
    solver: SlabSolver, tolerance: float, max_iterations: int
) -> tuple[np.ndarray, int, bool, list[float]]:
    """The plain fixed point: rebuild the source, sweep, repeat."""
    phi = np.zeros(2 * solver.n_cells)
    residuals: list[float] = []
    reference = None

    for iteration in range(1, max_iterations + 1):
        left, right = np.split(phi, 2)
        scatter_left, scatter_right = solver.scattering_source(left, right)
        fixed_left, fixed_right = solver.fixed_source()
        new_left, new_right, _ = solver.sweep(
            scatter_left + fixed_left, scatter_right + fixed_right
        )
        new_phi = np.concatenate([new_left, new_right])

        change = float(np.linalg.norm(new_phi - phi))
        if reference is None:
            reference = max(float(np.linalg.norm(new_phi)), 1e-30)
        residuals.append(change / reference)
        phi = new_phi

        if residuals[-1] < tolerance:
            return phi, iteration, True, residuals

    return phi, max_iterations, False, residuals


def gmres(
    solver: SlabSolver,
    tolerance: float,
    max_iterations: int,
    restart: int = 30,
) -> tuple[np.ndarray, int, bool, list[float]]:
    """Restarted GMRES on the transport operator.

    Written out rather than taken from scipy because the whole point is to count
    sweeps: every matrix-vector product here is exactly one transport sweep, and
    the restart interval genuinely changes how many are needed, which is what the
    convergence studies measure. A library call would give the same answer and
    hide the quantity being studied.
    """
    rhs = solver.uncollided()
    size = rhs.size
    phi = np.zeros(size)

    reference = max(float(np.linalg.norm(rhs)), 1e-30)
    residuals: list[float] = []
    total = 0

    while total < max_iterations:
        residual = rhs - solver.apply_operator(phi)
        total += 1
        beta = float(np.linalg.norm(residual))
        residuals.append(beta / reference)
        if residuals[-1] < tolerance:
            return phi, total, True, residuals

        inner = min(restart, max_iterations - total)
        if inner <= 0:
            break

        basis = np.zeros((size, inner + 1))
        basis[:, 0] = residual / beta
        hessenberg = np.zeros((inner + 1, inner))
        givens_c = np.zeros(inner)
        givens_s = np.zeros(inner)
        g = np.zeros(inner + 1)
        g[0] = beta
        used = 0

        for k in range(inner):
            candidate = solver.apply_operator(basis[:, k])
            total += 1
            # Modified Gram-Schmidt: stabler than the classical form, and the
            # cost is noise next to a sweep.
            for j in range(k + 1):
                hessenberg[j, k] = float(basis[:, j] @ candidate)
                candidate -= hessenberg[j, k] * basis[:, j]

            # Kept before the rotations overwrite the subdiagonal. It is both the
            # norm of the new basis vector and the breakdown test, and reading it
            # back out of the matrix afterwards is how this loop was previously
            # reduced to GMRES(1): the rotation sets that entry to zero, so the
            # breakdown test fired on every iteration and the restart interval
            # never did anything.
            next_norm = float(np.linalg.norm(candidate))
            hessenberg[k + 1, k] = next_norm

            for j in range(k):
                temp = givens_c[j] * hessenberg[j, k] + givens_s[j] * hessenberg[j + 1, k]
                hessenberg[j + 1, k] = (
                    -givens_s[j] * hessenberg[j, k] + givens_c[j] * hessenberg[j + 1, k]
                )
                hessenberg[j, k] = temp

            denominator = float(np.hypot(hessenberg[k, k], hessenberg[k + 1, k]))
            if denominator == 0.0:
                break
            givens_c[k] = hessenberg[k, k] / denominator
            givens_s[k] = hessenberg[k + 1, k] / denominator
            hessenberg[k, k] = denominator
            hessenberg[k + 1, k] = 0.0
            g[k + 1] = -givens_s[k] * g[k]
            g[k] = givens_c[k] * g[k]

            used = k + 1
            estimate = abs(g[k + 1]) / reference
            residuals.append(estimate)

            if estimate < tolerance:
                break
            if next_norm <= 1.0e-14 * max(beta, 1.0):
                # Lucky breakdown: the Krylov space is exhausted and the current
                # iterate is exact within it.
                break
            if total >= max_iterations:
                break
            basis[:, k + 1] = candidate / next_norm

        if used:
            y = np.linalg.solve(hessenberg[:used, :used], g[:used])
            phi = phi + basis[:, :used] @ y

        if residuals and residuals[-1] < tolerance:
            return phi, total, True, residuals

    return phi, total, False, residuals


# --- the entry point --------------------------------------------------------


def solve(
    problem: Problem,
    *,
    method: str = "petsc_gmres",
    tolerance: float = 1.0e-9,
    max_iterations: int = 300,
    restart: int = 30,
) -> Solution:
    """Solve one slab problem and report the quantities the tutorials print."""
    if method not in METHODS:
        raise SolverError(f"unknown method {method!r}; expected one of {', '.join(METHODS)}")

    solver = SlabSolver(problem)

    if method in ("classic_richardson", "petsc_richardson"):
        phi, iterations, converged, residuals = source_iteration(
            solver, tolerance, max_iterations
        )
    else:
        # BiCGStab is not implemented separately; it is a Krylov method on the
        # same operator with the same per-iteration cost, and claiming to be it
        # while running GMRES would misreport the thing being studied.
        if method == "petsc_bicgstab":
            raise SolverError(
                "petsc_bicgstab is not implemented by the reference solver. Use "
                "petsc_gmres or classic_richardson, or run against OpenSn."
            )
        phi, iterations, converged, residuals = gmres(
            solver, tolerance, max_iterations, restart
        )

    left, right = np.split(phi, 2)

    # One final sweep to recover the exiting angular flux, which the leakage
    # integral needs and the Krylov iterate does not carry.
    scatter_left, scatter_right = solver.scattering_source(left, right)
    fixed_left, fixed_right = solver.fixed_source()
    left, right, exiting = solver.sweep(scatter_left + fixed_left, scatter_right + fixed_right)

    half = solver.n_angles // 2
    weights_pos = solver.weights[solver.forward]
    mu_pos = solver.mu[solver.forward]

    outflow = float(
        np.sum(weights_pos * mu_pos * exiting["right"])
        + np.sum(weights_pos * mu_pos * exiting["left"])
    )

    cell_average = 0.5 * (left + right)
    absorption = float(np.sum(solver.sigma_a * solver.h * cell_average))
    production = float(np.sum(solver.q_ext * solver.h))
    total_width = float(np.sum(solver.h))

    return Solution(
        scalar_flux_left=left,
        scalar_flux_right=right,
        cell_width=solver.h,
        absorption_rate=absorption,
        production_rate=production,
        inflow_rate=0.0,
        outflow_rate=outflow,
        balance=production - absorption - outflow,
        average_flux=float(np.sum(solver.h * cell_average) / total_width),
        max_flux=float(np.max(np.concatenate([left, right]))),
        iterations=iterations,
        sweeps=iterations,
        converged=converged,
        residuals=residuals,
        method=method,
    )
