# Fixtures

Recorded run output, replayed by `FakeRunner` so that the whole test suite runs
on a machine with no OpenSn, no MPI, and no network.

## Prefer the reference solver to these

Since these were written, `ReferenceRunner` arrived and computes the 1D
scenarios for real, in under a second, reproducing OpenSn's recorded gold
values. It is the default runner and it is a better answer than any of this.

These fixtures are still here because they exercise `FakeRunner`, which is what
will replay genuine recordings once someone runs these cases on a machine with
OpenSn built. Until then, prefer `--runner reference` for anything you intend to
look at.

## These fixtures are HAND-WRITTEN

**None of the output in this directory came from running OpenSn.** It was
written by hand, and it must be replaced with a real recording as soon as
someone runs these cases on a machine that has OpenSn built. Until then, treat
every number here as a shape, not a result.

What is faithful, and what is not:

**Faithful: the log formats.** Every line reproduces a format read out of the
OpenSn source at commit `2fd4a19ceade4f8da581a5029c68f474d3333ccb`:

| Line | Source |
| --- | --- |
| `WGS groups [0-0] iteration = N, residual = ...` | `modules/linear_boltzmann_solvers/discrete_ordinates_problem/iterative_methods/wgs_convergence_test.cc` |
| `AGS iteration = N, l2_change = ...` | `.../iterative_methods/ags_linear_solver.cc` |
| `avg_sweep_time = ... s, sweep_time_per_unknown = ... ns` | `.../iterative_methods/ags_linear_solver.cc` |
| `, label = value` field separator | `framework/logging/log_format.h`, `AppendNumericField` |
| `status = converged` | `modules/linear_boltzmann_solvers/lbs_problem/iterative_methods/iteration_logging.h` |
| `Balance table:` block | `.../compute/discrete_ordinates_compute.cc` |
| `HH:MM:SS.s` timestamps | `framework/utils/timer.cc`, `Timer::GetTimeString` |
| `Absorption=`, `OutFlow=` | the Reed regression script itself |

**Faithful: two numbers.** `Absorption=1.006178e+02` and `OutFlow=3.821562e-01`
are the gold values recorded for `reed_balance.py` in the OpenSn regression
suite's `tests.json`. They are consistent with the problem: the total source is
50 per cm over the 2 cm first region plus 1 per cm over the 1 cm fourth region,
so production is 101, and absorption plus outflow recovers it.

**Invented: everything else.** Iteration counts, residual sequences, wall
times, and the average flux are made up. They are internally consistent and
they follow the qualitative behaviour the OpenSn user guide describes for these
controls, which is what lets the acceptance test check the shape of the
convergence study. A tighter tolerance takes more iterations; a restart
interval well below the Krylov dimension the problem needs takes many more; the
flux error tracks the tolerance. None of that is measured.

## Replacing a fixture with a real recording

On a machine with OpenSn built:

```shell
autoopensn run tests/data/gmres_convergence.yaml --runner local
```

then, for each run directory, call `autoopensn.runner.fake.record_fixture` with
the `RunResult` and a truthful `provenance` string. Delete the `hand_written`
flag from `meta.json` when you do, and update this file.

## Layout

    <study>/<case_id>/
        stdout.txt    replayed verbatim
        stderr.txt    optional
        meta.json     exit_code, wall_time, timed_out, provenance
        <outputs>     any other file is copied into the run directory

`reed_gmres/` holds the seven cases of the GMRES convergence study.
`reed_default/baseline/` holds a run at the template's default parameters,
which is the only case where the regression gold values apply.
