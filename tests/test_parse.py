"""Tests for observable extraction and gold comparison.

These are the tests that matter most for trust. The parser is what stands
between raw solver output and a table a language model will narrate, and a
parser that silently returns None produces a story about nothing.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from autoopensn.parse import compare_to_gold, key_value, load_gold, parse_stdout
from autoopensn.parse.gold import GoldError, matches_gold_case
from autoopensn.parse.table import convergence_table, observe, results_table
from autoopensn.runner import FakeRunner, RunRequest


# --- key_value, the OpenSn regression harness's own rule --------------------


def test_key_value_takes_the_word_after_the_key():
    assert key_value("Absorption=1.006178e+02\n", "Absorption=") == pytest.approx(100.6178)


def test_key_value_uses_the_last_occurrence():
    text = "Max-value1=1.0\nMax-value1=2.0\n"
    assert key_value(text, "Max-value1=") == 2.0


def test_key_value_returns_none_when_the_key_is_absent():
    assert key_value("nothing here\n", "AvgFlux=") is None


def test_key_value_ignores_a_key_followed_by_non_numeric_text():
    assert key_value("Status= converged\n", "Status=") is None


def test_key_value_handles_a_space_after_the_key():
    assert key_value("Balance = 3.55e-14\n", "Balance =") == pytest.approx(3.55e-14)


# --- parsing a whole run ----------------------------------------------------


@pytest.fixture
def reference_stdout(gmres_fixtures):
    return (gmres_fixtures / "reference" / "stdout.txt").read_text()


def test_all_five_observables_are_found(reference_stdout):
    observations = parse_stdout(reference_stdout)
    assert observations.avg_flux == pytest.approx(2.387194)
    assert observations.sweeps == 45
    assert observations.iterations == 45
    assert observations.balance_residual == pytest.approx(3.552714e-14)
    # wall_time is the runner's, not the log's; see test_wall_time_comes_from_the_runner.


def test_gold_keys_are_parsed(reference_stdout):
    observations = parse_stdout(reference_stdout)
    assert observations.absorption_rate == pytest.approx(100.6178)
    assert observations.outflow_rate == pytest.approx(0.3821562)


def test_convergence_status_is_read(reference_stdout):
    observations = parse_stdout(reference_stdout)
    assert observations.inner_status == "converged"
    assert observations.converged is True


def test_ags_iterations_are_counted(reference_stdout):
    assert parse_stdout(reference_stdout).ags_iterations == 1


def test_sweep_count_is_labelled_derived(reference_stdout):
    """OpenSn does not print a sweep count for a steady-state source solve."""
    assert parse_stdout(reference_stdout).sweeps_source == "derived"


def test_a_logged_sweep_count_is_preferred_over_the_derivation():
    text = (
        "00:00:01.0 WGS groups [0-0] iteration = 0, residual = 1.000000e+00\n"
        "00:00:01.1 WGS groups [0-0] iteration = 1, residual = 1.000000e-08, status = converged\n"
        "PI final, k_eff = 1.0000000, sweeps = 37\n"
    )
    observations = parse_stdout(text)
    assert observations.sweeps == 37
    assert observations.sweeps_source == "logged"
    assert observations.iterations == 1


def test_tighter_tolerances_take_more_iterations(gmres_fixtures):
    counts = {
        case: parse_stdout((gmres_fixtures / case / "stdout.txt").read_text()).iterations
        for case in ("l_abs_tol-1.0e-4", "l_abs_tol-1.0e-6", "l_abs_tol-1.0e-8")
    }
    assert counts["l_abs_tol-1.0e-4"] < counts["l_abs_tol-1.0e-6"] < counts["l_abs_tol-1.0e-8"]


def test_multiple_solves_are_summed():
    """Two groupset solves is two iteration counts, not one."""
    text = "\n".join(
        [
            "00:00:01.0 WGS groups [0-0] iteration = 0, residual = 1.0e+00",
            "00:00:01.1 WGS groups [0-0] iteration = 1, residual = 1.0e-04",
            "00:00:01.2 WGS groups [0-0] iteration = 2, residual = 1.0e-08, status = converged",
            "00:00:01.3 WGS groups [1-1] iteration = 0, residual = 1.0e+00",
            "00:00:01.4 WGS groups [1-1] iteration = 1, residual = 1.0e-08, status = converged",
            "00:00:01.5 AGS iteration = 0, l2_change = 1.0e-08, status = converged",
        ]
    )
    observations = parse_stdout(text)
    assert observations.solve_blocks == (2, 1)
    assert observations.iterations == 3
    assert observations.sweeps == 3


def test_empty_output_yields_no_observations_rather_than_zeros():
    """Blank is honest. Zero would read as a converged run that took no work."""
    observations = parse_stdout("")
    assert observations.iterations is None
    assert observations.sweeps is None
    assert observations.avg_flux is None
    assert observations.balance_residual is None
    assert observations.sweeps_source == "none"


def test_an_iteration_limit_is_reported_as_such():
    text = (
        "00:00:01.0 WGS groups [0-0] iteration = 300, residual = 4.2e-03"
        ", status = iteration_limit\n"
    )
    observations = parse_stdout(text)
    assert observations.inner_status == "iteration_limit"
    assert observations.converged is False


def test_classic_richardson_lines_parse_too():
    """A different iterative method prints different fields on the same line."""
    text = (
        "00:00:02.1 WGS groups [0-0] iteration = 0, phi_change = 1.0e+00"
        ", psi_change = 1.0e+00, rho_est = 0.0000\n"
        "00:00:02.9 WGS groups [0-0] iteration = 12, phi_change = 3.1e-10"
        ", psi_change = 2.0e-10, rho_est = 0.8912, status = converged\n"
    )
    observations = parse_stdout(text)
    assert observations.iterations == 12
    assert observations.converged is True


# --- gold comparison --------------------------------------------------------


def test_gold_values_load_from_the_vendored_tests_json(data_dir):
    checks = load_gold(data_dir / "opensn_tests.json", "reed_balance.py")
    assert {check.key for check in checks} == {"Absorption=", "OutFlow="}
    assert any(check.expected == pytest.approx(100.6178) for check in checks)


def test_unknown_gold_case_is_an_error(data_dir):
    with pytest.raises(GoldError):
        load_gold(data_dir / "opensn_tests.json", "not_a_case.py")


def test_default_parameters_match_the_regression_case(reed_template):
    applicable, reason = matches_gold_case(reed_template, reed_template.defaults())
    assert applicable, reason


def test_output_only_parameters_do_not_invalidate_gold(reed_template):
    """Printing an extra flux does not change the absorption rate."""
    parameters = {**reed_template.defaults(), "emit_avg_flux": True}
    applicable, _ = matches_gold_case(reed_template, parameters)
    assert applicable


def test_a_changed_tolerance_invalidates_gold(reed_template):
    parameters = {**reed_template.defaults(), "l_abs_tol": 1.0e-6}
    applicable, reason = matches_gold_case(reed_template, parameters)
    assert not applicable
    assert "l_abs_tol" in reason


def test_default_run_passes_its_gold_check(default_fixtures, reed_template, data_dir):
    stdout = (default_fixtures / "baseline" / "stdout.txt").read_text()
    comparison = compare_to_gold(
        stdout,
        reed_template,
        reed_template.defaults(),
        tests_json=data_dir / "opensn_tests.json",
    )
    assert comparison.applicable
    assert comparison.passed
    assert comparison.worst_relative_difference() == pytest.approx(0.0, abs=1e-9)


def test_gold_comparison_is_skipped_with_a_reason_for_a_different_problem(
    reed_template, default_fixtures, data_dir
):
    stdout = (default_fixtures / "baseline" / "stdout.txt").read_text()
    parameters = {**reed_template.defaults(), "n_polar": 64}
    comparison = compare_to_gold(
        stdout, reed_template, parameters, tests_json=data_dir / "opensn_tests.json"
    )
    assert not comparison.applicable
    assert comparison.passed is None
    assert "n_polar" in comparison.reason


def test_a_wrong_answer_fails_its_gold_check(reed_template, data_dir):
    stdout = "Absorption=9.900000e+01\nOutFlow=3.821562e-01\n"
    comparison = compare_to_gold(
        stdout,
        reed_template,
        reed_template.defaults(),
        tests_json=data_dir / "opensn_tests.json",
    )
    assert comparison.applicable
    assert comparison.passed is False


def test_a_missing_key_fails_rather_than_passing_vacuously(reed_template, data_dir):
    comparison = compare_to_gold(
        "nothing useful here\n",
        reed_template,
        reed_template.defaults(),
        tests_json=data_dir / "opensn_tests.json",
    )
    assert comparison.passed is False


def test_gold_falls_back_to_the_sidecar_when_the_checkout_is_absent(
    reed_template, tmp_path
):
    """Most machines have no OpenSn checkout and can still check an answer."""
    stdout = "Absorption=1.006178e+02\nOutFlow=3.821562e-01\n"
    comparison = compare_to_gold(
        stdout, reed_template, reed_template.defaults(), tests_json=tmp_path / "absent.json"
    )
    assert comparison.applicable
    assert comparison.passed


def test_the_sidecar_fallback_still_catches_a_wrong_answer(reed_template, tmp_path):
    comparison = compare_to_gold(
        "Absorption=9.9e+01\nOutFlow=3.821562e-01\n",
        reed_template,
        reed_template.defaults(),
        tests_json=tmp_path / "absent.json",
    )
    assert comparison.passed is False


def test_a_template_with_no_gold_says_so(reed_template, tmp_path):
    bare = replace(reed_template, gold={})
    comparison = compare_to_gold("Absorption=1.0e+02\n", bare, bare.defaults())
    assert not comparison.applicable
    assert "no gold" in comparison.reason


# --- the results table ------------------------------------------------------


def run_all(spec, fixtures, tmp_path):
    runner = FakeRunner(fixtures)
    points = spec.expand()
    results = []
    for point in points:
        request = RunRequest(
            case_id=point.case_id,
            script="",
            directory=tmp_path / point.case_id,
            parameters=point.parameters,
        )
        results.append(runner.run(request))
    return results, points


def test_table_has_one_row_per_point(gmres_spec, gmres_fixtures, tmp_path, reed_template):
    results, points = run_all(gmres_spec, gmres_fixtures, tmp_path)
    frame = results_table(results, points, reed_template)
    assert len(frame) == 7
    assert set(frame["case_id"]) == {point.case_id for point in points}


def test_wall_time_comes_from_the_runner(gmres_spec, gmres_fixtures, tmp_path):
    results, points = run_all(gmres_spec, gmres_fixtures, tmp_path)
    frame = results_table(results, points)
    assert frame["wall_time"].notna().all()
    assert (frame["wall_time"] > 0).all()


def test_relative_flux_difference_is_measured_against_the_reference(
    gmres_spec, gmres_fixtures, tmp_path
):
    results, points = run_all(gmres_spec, gmres_fixtures, tmp_path)
    frame = results_table(results, points).set_index("case_id")
    assert frame.loc["reference", "rel_flux_diff"] == pytest.approx(0.0)
    assert frame.loc["l_abs_tol-1.0e-4", "rel_flux_diff"] > frame.loc[
        "l_abs_tol-1.0e-8", "rel_flux_diff"
    ]


def test_no_reference_means_blank_not_an_arbitrary_baseline(gmres_spec, gmres_fixtures, tmp_path):
    spec = gmres_spec.model_copy(update={"reference": None})
    results, points = run_all(spec, gmres_fixtures, tmp_path)
    frame = results_table(results, points)
    assert frame["rel_flux_diff"].isna().all()


def test_a_failed_run_still_gets_a_row(gmres_spec, gmres_fixtures, tmp_path):
    results, points = run_all(gmres_spec, gmres_fixtures, tmp_path)
    broken = results[0].__class__(
        case_id=results[0].case_id,
        exit_code=1,
        stdout="",
        stderr="RuntimeError: diverged\n",
        wall_time=2.0,
        directory=results[0].directory,
        runner="fake",
    )
    frame = results_table([broken] + results[1:], points)
    row = frame[frame["case_id"] == broken.case_id].iloc[0]
    assert row["ok"] is False or row["ok"] == False  # noqa: E712
    assert "diverged" in row["failure"]
    assert pd.isna(row["sweeps"])


def test_a_point_that_never_ran_is_marked_not_run(gmres_spec, gmres_fixtures, tmp_path):
    results, points = run_all(gmres_spec, gmres_fixtures, tmp_path)
    frame = results_table(results[:-1], points)
    row = frame[frame["case_id"] == points[-1].case_id].iloc[0]
    assert row["failure"] == "not run"


def test_convergence_table_is_the_four_columns(gmres_spec, gmres_fixtures, tmp_path):
    results, points = run_all(gmres_spec, gmres_fixtures, tmp_path)
    view = convergence_table(results_table(results, points))
    assert list(view.columns) == [
        "Case",
        "Relative flux difference",
        "Sweeps",
        "Wall time (s)",
    ]
    assert len(view) == 7


def test_varied_parameters_appear_as_columns(gmres_spec, gmres_fixtures, tmp_path):
    results, points = run_all(gmres_spec, gmres_fixtures, tmp_path)
    frame = results_table(results, points)
    assert "param_l_abs_tol" in frame.columns
    assert "param_gmres_restart_interval" in frame.columns


def test_a_final_summary_line_reports_a_solve_that_did_not_converge():
    """OpenSn prints this instead of a converged status when a solve hits its cap.

    Without reading it, such a run parses as having no status at all, which in a
    table reads as "unknown" rather than "it did not converge".
    """
    text = (
        "00:00:01.0 WGS groups [0-0] iteration = 0, residual = 1.0e+00\n"
        "00:00:09.0 WGS groups [0-0] iteration = 300, residual = 4.2e-03\n"
        "00:00:09.1 WGS groups [0-0] final, status = iteration_limit"
        ", iterations = 300, phi_change = 4.200000e-03\n"
    )
    observations = parse_stdout(text)
    assert observations.inner_status == "iteration_limit"
    assert observations.converged is False



# --- real OpenSn output ------------------------------------------------------

# Verbatim from the first run of reed_1d at defaults on a real OpenSn build
# (commit 2fd4a19, GCC 15.2, MPICH 4.3.2, one rank). Every line carries the
# rank prefix, which the hand-written fixtures did not.
REAL_BALANCE_BLOCK = """\
[0]  Balance table:
[0]   Absorption rate             = 1.006178e+02
[0]   Production rate             = 1.010000e+02
[0]   In-flow rate                = 0.000000e+00
[0]   Out-flow rate               = 3.821562e-01
[0]   Balance                     = -1.084338e-11
[0]  
"""


def test_balance_is_read_from_real_rank_prefixed_output():
    observations = parse_stdout(REAL_BALANCE_BLOCK)
    assert observations.balance_residual == pytest.approx(-1.084338e-11)


def test_balance_is_still_read_without_a_rank_prefix():
    """The fixtures omit the prefix; both shapes must parse."""
    observations = parse_stdout("Balance table:\n Balance                     = 3.55e-15\n")
    assert observations.balance_residual == pytest.approx(3.55e-15)
