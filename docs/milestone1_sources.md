# Where Milestone 1's problem setup came from

The brief asks for tutorial 5.2.2, "GMRES Convergence Controls", to be read from
the checkout and templated. **That tutorial does not exist at the pinned
commit.** This note records what was used instead, so that nobody later assumes
the acceptance test was copied from a document they cannot find.

## What is actually in the checkout

At commit `2fd4a19ceade4f8da581a5029c68f474d3333ccb`, the directory the tutorial
would live in is empty:

```
doc/source/tutorials/numerical_methods/iterative_methods/
└── .gitkeep
```

So are its siblings `acceleration/` and `convergence_and_error/`. The
`numerical_methods` index lists exactly one child, `angular_discretization`. The
tutorial set is scaffolded but unwritten.

Searching the whole documentation tree for "Convergence Controls" returns one
file, and it is not a tutorial.

## What was used instead

**The problem setup** comes from the regression script
`test/python/modules/linear_boltzmann_solvers/transport_steady/reed_balance.py`.
This is a better source than a tutorial would have been: it is executed by
OpenSn's own test suite on every commit, and `tests.json` records gold values
for it. The Reed template reproduces it byte for byte at default parameters,
which `tests/test_template_reed.py` asserts on every run.

**The parameter semantics and their sensible ranges** come from
`doc/source/userguide/iterative_methods.rst`, which is where the GMRES
convergence material lives. Its "Convergence Controls" section and its "GMRES
restart interval" subsection are the closest thing in the repository to the
tutorial the brief describes. They are also what the knowledge pack returns
first for the query "GMRES restart interval", which is a useful confirmation
that the retrieval half of this system agrees.

Specifically, the sidecar's declared ranges follow that document:

| Parameter | Document says | Sidecar |
| --- | --- | --- |
| `l_abs_tol` | 1e-6 default; 1e-4 to 1e-5 for quick studies; 1e-7 to 1e-8 when balance quantities matter | 1e-18 to 1e-2, defaulting to the regression script's 1e-9 |
| `gmres_restart_interval` | 30 default; 20 to 50 a reasonable first range; 60 to 100 for difficult problems | 1 to 1000, defaulting to 30 |
| `l_max_its` | 200 default; 100 to 300 normal | 1 to 100000, defaulting to the script's 300 |
| `inner_linear_method` | four named methods | exactly those four |

The lower bounds on `l_abs_tol` and `gmres_restart_interval` are OpenSn's own,
read from `modules/linear_boltzmann_solvers/lbs_problem/groupset/lbs_groupset.cc`
rather than from prose.

**The sweep values** in `tests/data/gmres_convergence.yaml` are the ones the
brief's prompt names: tolerances 1e-4, 1e-6, 1e-8, restart intervals 5, 20, 50,
and a 1e-10 reference. They sit inside the ranges above, except that a restart
interval of 5 is deliberately below the document's recommended range, which is
the point of including it.

**The expected table shape** is the brief's: Case, Relative flux difference,
Sweeps, Wall time. Without the tutorial there is no upstream table to match
column for column.

## What this means for the acceptance test

The test cannot compare against a published table, because there is none. What
it does instead is in `tests/test_milestone1.py`: it checks that the spec expands
to the right seven cases, that each renders to a script differing from the
validated regression script only in the knobs the spec moved, that all five
observables are extracted, and that the table has the required shape.

The numbers themselves come from hand-written fixtures and are not measurements.
See `tests/fixtures/README.md`.

## If the tutorial appears upstream

Re-derive the template from it rather than from `reed_balance.py` if the two
disagree on the problem setup, and add its table to this document so the
acceptance test can compare against it. The template's sidecar records its
source file, so the change is one line plus a re-run of the byte-exact test.
