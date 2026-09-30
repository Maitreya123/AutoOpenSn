# Scenarios: the OpenSn tutorials as a runnable library

AutoOpenSn's scenario library is the OpenSn tutorial set. Every notebook under
`doc/source/tutorials` at the pinned commit becomes a template with a parameter
sidecar, generated automatically rather than written by hand.

This is worth doing because a tutorial is an unusual artifact: it is a worked
example that is also a regression test. It has prose explaining the physics, a
script that runs, and, for nearly all of them, a `tests.json` recording what the
answer should be. That is exactly the three things a parameter study needs.

| | |
| --- | --- |
| Tutorials catalogued | 40 |
| With a recorded correct answer | 39 |
| Parameters found across them | 930 |
| Hand-written templates also catalogued | 1 (the Reed benchmark) |

## The guarantee

**A generated template rendered with every parameter at its default reproduces
the tutorial's script character for character.** The generator verifies this
before it writes anything, and `tests/test_tutorials.py` re-checks every
template on every test run against the live checkout.

Everything else rests on that. A spec that only moves declared parameters within
declared ranges produces a script that differs from a validated OpenSn tutorial
by exactly the values it changed, and nothing else. The physics is not ours to
edit, and this is the mechanism that stops us editing it by accident.

## How a knob is found

Three shapes in the syntax tree qualify:

1. A module-level assignment of a literal, `num_cells = 40`.
2. A literal keyword argument, `GLProductQuadrature1DSlab(n_polar=8)`.
3. A literal value in a dictionary keyed by strings, `"l_abs_tol": 1.0e-10`.

Each knob records the exact source span its literal occupies, and the generator
replaces those spans back to front. Nothing else is touched: not whitespace, not
comments, not import order. Reprinting the syntax tree instead would reformat
the file and make the byte-exact claim impossible to state.

Deliberately excluded:

- **Lists longer than twelve entries.** The forty-one node positions of a mesh
  are computed from the cell count above them; exposing them would offer a study
  the chance to hand-write a mesh.
- **Tuples.** A group range `(0, 9)` is meaningless varied at one end.
- **Computed values.** A list comprehension is not a constant.
- **Structural arguments.** `RPPLogicalVolume(infx=True)` declares what the
  volume *is*. Flipping it does not tune the problem, it replaces it.
- **Literals whose notation cannot be reproduced.** See below.

## Notation

The tutorials write `4.0`, `2.` and `1.0e-10` for what Python considers the same
kind of thing, and only one spelling of each reproduces the file it came from.
The generator renders each value in every candidate notation and keeps the one
that matches the source text exactly.

A literal whose notation cannot be reproduced stops being a knob rather than
becoming one that corrupts the script. Across all forty tutorials exactly one
literal was skipped this way. That is the conservative direction, and it is why
the byte-exact test passes for every generated template rather than most of them.

## Where a range comes from

Every declared bound records its own provenance, because a limit OpenSn enforces
and a guess scaled from a tutorial's own value are not the same kind of claim.

| Provenance | Meaning |
| --- | --- |
| `opensn` | Read from OpenSn's parameter constraints at the pinned commit. Violating it is an error from the solver. |
| `physics` | A property of the quantity. A scattering ratio above one is a multiplying medium, a different problem. |
| `inferred` | A guard scaled three orders of magnitude from the value the tutorial uses. Not a code limit, not verified. |

The interface shows this on every parameter. An `inferred` bound is the first
thing to widen by hand when a legitimate study is rejected.

## Rebuilding

```shell
autoopensn scenarios --rebuild
```

Reads the tutorials from the pinned checkout, regenerates every template,
re-verifies the byte-exact claim, and saves the catalog. The generated templates
and the catalog are committed rather than built on demand, for two reasons: a
machine with only this repository can still run a study, and a change to the
generator shows up in review as a diff to the physics scripts it produces.

## What a scenario cannot do yet

**Most cannot be run here**, because OpenSn is not installed. A generated study
renders its scripts and stops, with a row per case saying no recording exists.
Only the Reed GMRES study has recorded fixtures, and those are hand-written.

**Fourteen need data files**, meshes and cross-section libraries that live beside
their notebook. The catalog records which, and the runner does not yet stage
them into the run directory.

**One is not sweepable.** `xs_read_ascii` reads a fixed file and prints what is
in it. It is catalogued and marked, and it is kept out of shortlists, because a
parameter study over it is not a thing.
