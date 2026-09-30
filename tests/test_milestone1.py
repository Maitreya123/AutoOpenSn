"""Milestone 1: the acceptance test for this project.

Given the study behind the prompt

    compare GMRES tolerances 1e-4, 1e-6, 1e-8 and restart intervals 5, 20, 50
    against a 1e-10 reference on the 1D transport problem

the tool must produce a spec, render seven scripts, run them, and produce a
table with columns Case, Relative flux difference, Sweeps, and Wall time.

The spec is written by hand here rather than generated, because the stage that
generates it is a stub. Everything after the spec is the real pipeline.

**What this test does and does not establish.** The runs are replayed from
hand-written fixtures, so the *numbers* in the table are not measurements. What
is established is everything else: that the spec expands to the right seven
cases, that each renders to a script differing from the validated regression
script only in the knobs the spec moved, that the parser finds every observable,
and that the table has the tutorial's shape. When a real recording replaces the
fixtures, this same test becomes a claim about physics without changing a line.
"""

from __future__ import annotations

import ast

import pandas as pd
import pytest

from autoopensn.parse.table import convergence_table
from autoopensn.runner import FakeRunner
from autoopensn.store import RunStore
from autoopensn.study import render_points, run_study, write_scripts

EXPECTED_CASES = {
    "l_abs_tol-1.0e-4",
    "l_abs_tol-1.0e-6",
    "l_abs_tol-1.0e-8",
    "gmres_restart_interval-5",
    "gmres_restart_interval-20",
    "gmres_restart_interval-50",
    "reference",
}


@pytest.fixture
def study(gmres_spec, gmres_fixtures, tmp_path):
    return run_study(
        gmres_spec,
        FakeRunner(gmres_fixtures),
        run_root=tmp_path / "runs",
        store=RunStore(tmp_path / "cache.db"),
    )


# --- the spec ---------------------------------------------------------------


def test_the_spec_is_seven_cases(gmres_spec):
    """Seven, not ten: each dimension varies alone, plus one reference."""
    points = gmres_spec.expand()
    assert len(points) == 7
    assert {point.case_id for point in points} == EXPECTED_CASES


def test_the_spec_declares_a_reference(gmres_spec):
    reference = [point for point in gmres_spec.expand() if point.is_reference]
    assert len(reference) == 1
    assert reference[0].parameters["l_abs_tol"] == pytest.approx(1.0e-10)


# --- rendering --------------------------------------------------------------


def test_seven_scripts_are_rendered(gmres_spec, tmp_path):
    written = write_scripts(gmres_spec, tmp_path / "render")
    assert len(written) == 7
    assert (tmp_path / "render" / "spec.yaml").exists()


def test_every_rendered_script_is_valid_python(gmres_spec):
    for rendered in render_points(gmres_spec):
        ast.parse(rendered.script)


def test_every_script_records_the_pack_commit(gmres_spec):
    for rendered in render_points(gmres_spec):
        assert "2fd4a19ceade4f8da581a5029c68f474d3333ccb" in rendered.script


def test_each_script_differs_from_the_regression_script_only_in_its_knobs(
    gmres_spec, vendored_reed
):
    """The safety claim, made concrete.

    Every generated script is the validated regression script with the flux
    block added and the two swept solver settings changed. Nothing else moves.
    """
    import difflib

    from autoopensn.store.db import strip_provenance

    for rendered in render_points(gmres_spec):
        body = strip_provenance(rendered.script)
        changed = [
            line
            for line in difflib.unified_diff(
                vendored_reed.splitlines(), body.splitlines(), lineterm="", n=0
            )
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        ]
        removed = [line for line in changed if line.startswith("-")]
        # Only the tolerance and the restart interval are ever removed; every
        # other change is an addition (the flux block and its import).
        for line in removed:
            assert "l_abs_tol" in line or "gmres_restart_interval" in line, line


def test_the_tolerance_reaches_the_script(gmres_spec):
    scripts = {r.case_id: r.script for r in render_points(gmres_spec)}
    assert '"l_abs_tol": 1.0e-4,' in scripts["l_abs_tol-1.0e-4"]
    assert '"l_abs_tol": 1.0e-10,' in scripts["reference"]
    assert '"gmres_restart_interval": 5,' in scripts["gmres_restart_interval-5"]


def test_every_script_prints_a_flux(gmres_spec):
    """Without this the relative-difference column is seven blanks."""
    for rendered in render_points(gmres_spec):
        assert "AvgFlux=" in rendered.script


# --- running and tabulating -------------------------------------------------


def test_all_seven_cases_run(study):
    assert len(study.results) == 7
    assert not study.failures


def test_every_run_has_a_directory_with_its_script(study):
    for result in study.results:
        assert (result.directory / "script.py").exists()
        assert (result.directory / "stdout.txt").exists()
        assert (result.directory / "spec.yaml").exists()


def test_the_table_has_the_tutorial_shape(study):
    view = convergence_table(study.table)
    assert list(view.columns) == [
        "Case",
        "Relative flux difference",
        "Sweeps",
        "Wall time (s)",
    ]
    assert len(view) == 7
    assert view.notna().all().all()


def test_every_observable_is_populated(study):
    for column in ("avg_flux", "sweeps", "iterations", "wall_time", "balance_residual"):
        assert study.table[column].notna().all(), column


def test_the_reference_is_its_own_zero(study):
    frame = study.table.set_index("case_id")
    assert frame.loc["reference", "rel_flux_diff"] == pytest.approx(0.0)


def test_tighter_tolerances_are_more_accurate_and_cost_more_sweeps(study):
    """The trend the study exists to show, read off the table.

    With hand-written fixtures this checks that the pipeline preserves an
    ordering, not that OpenSn has one. It becomes a physics claim when the
    fixtures are re-recorded.
    """
    frame = study.table.set_index("case_id")
    order = ["l_abs_tol-1.0e-4", "l_abs_tol-1.0e-6", "l_abs_tol-1.0e-8"]
    differences = [frame.loc[case, "rel_flux_diff"] for case in order]
    sweeps = [frame.loc[case, "sweeps"] for case in order]
    assert differences == sorted(differences, reverse=True)
    assert sweeps == sorted(sweeps)


def test_a_small_restart_interval_costs_sweeps(study):
    frame = study.table.set_index("case_id")
    assert (
        frame.loc["gmres_restart_interval-5", "sweeps"]
        > frame.loc["gmres_restart_interval-50", "sweeps"]
    )


def test_the_sweep_column_says_it_is_derived(study):
    """A reader must not mistake a derived count for a measurement."""
    assert (study.table["sweeps_source"] == "derived").all()


def test_the_table_records_that_these_rows_were_replayed(study):
    assert study.table["replayed"].all()


def test_gold_is_reported_as_inapplicable_with_a_reason(study):
    """None of these cases is the regression case, and the table must say so."""
    assert study.table["gold_passed"].isna().all()
    assert study.table["gold_reason"].notna().all()


def test_the_table_survives_a_round_trip_to_csv(study, tmp_path):
    path = tmp_path / "results.csv"
    study.table.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    assert len(reloaded) == 7
    assert reloaded["sweeps"].tolist() == study.table["sweeps"].tolist()
