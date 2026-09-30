# The reference solver

AutoOpenSn ships a 1D discrete-ordinates solver of its own, in numpy. It lets
the one-dimensional scenarios run in the interface in under a second, on a
machine where OpenSn is not installed.

**It is not OpenSn.** Nothing in this repository presents its output as OpenSn's.
The runner is called `reference`, its log opens with a banner saying so, the
interface says so beside the runner selector, and the cache treats a number from
it and a number from OpenSn as different measurements rather than
interchangeable ones.

## Why it exists

Installing OpenSn means compiling PETSc, VTK, HDF5, Boost and Caliper before you
get to OpenSn itself. On eight cores that is one to three hours and 12 to 18 GB
of build trees, and there is no prebuilt wheel, conda package or Homebrew
formula to avoid it. Without something like this, a generated study on an
ordinary laptop renders its scripts and stops, which is a poor answer to "show
me the result".

## Why it can be trusted, within its range

OpenSn's regression suite records what the answers are, and this reproduces
them.

| Scenario | Quantity | Reference solver | OpenSn gold |
| --- | --- | --- | --- |
| `reed_1d` | `Absorption=` | 1.006178e+02 | 100.6178 |
| `reed_1d` | `OutFlow=` | 3.821562e-01 | 0.3821562 |
| `first_1d_fixed_source` | `FOUNDATION_1D_MAX_FLUX=` | 9.62393909e-01 | 0.962393909 |

The first two match at the precision the tutorial prints, which is the precision
the regression harness itself compares at. The third matches to 2e-10 against a
tolerance of 1e-8. The particle balance closes to about 1e-12.

These are asserted in `tests/test_sn1d.py` and again end to end, through the
normal parser and the normal gold comparison, in
`tests/test_reference_runner.py`. A change that breaks the agreement fails the
suite rather than quietly producing different physics.

## What it implements, and why these choices

The one-group, isotropic-scattering, steady-state transport equation in slab
geometry, with vacuum boundaries.

**Angle: Gauss-Legendre**, the same quadrature `GLProductQuadrature1DSlab` uses,
with weights summing to two. That normalisation is what puts the scalar flux and
the leakage on OpenSn's scale rather than half or twice it.

**Space: linear discontinuous.** OpenSn uses PWLD, which in one dimension is
linear discontinuous Galerkin with upwinding. This is the choice that makes the
agreement exact rather than approximate. A diamond-difference or step scheme
solves the same equation and lands several digits away, which would look like a
bug in one of the two codes and be neither.

**Iteration: source iteration and GMRES**, because the difference between them
is what several of these studies are about. Each is one sweep per iteration, so
a sweep count compares them fairly. GMRES is written out rather than taken from
a library precisely because the sweep count is the quantity being studied, and a
library call would hide it.

On the Reed problem, computed rather than recorded:

| Method | Restart | Sweeps to 1e-9 |
| --- | --- | --- |
| GMRES | 30 | 12 |
| GMRES | 10 | 13 |
| GMRES | 5 | 20 |
| GMRES | 3 | 31 |
| Source iteration | n/a | 77 |

## What it refuses

Two dimensions, three dimensions, multigroup, anisotropic scattering,
eigenvalue problems, transients, acceleration, and reflecting boundaries. Every
one of those raises, naming what is unsupported, rather than approximating.

Reflecting boundaries are worth a note because they look easy and are not: the
flux entering a reflecting face is the flux leaving it, which a single sweep does
not have yet, so it needs an iteration this solver does not do. Returning the
answer to a problem with one boundary quietly left open would be worse than
refusing.

Two scenarios are supported today, `reed_1d` and `first_1d_fixed_source`. The
rest of the catalogue needs OpenSn itself.

## How it fits the pipeline

It writes its output in OpenSn's own log format, so the parser, the tables, the
gold comparison and the narration all work on it unchanged. The rest of the
pipeline does not know or care which solver produced a run. The one thing it
must know, that this was not OpenSn, travels in the runner name and the banner
rather than as a special case somewhere downstream.
