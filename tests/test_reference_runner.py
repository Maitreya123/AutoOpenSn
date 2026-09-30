"""Tests for running a scenario with the reference solver instead of OpenSn.

Two things are being checked. That the numbers agree with what OpenSn's
regression suite records, which is the only reason this runner is allowed to
exist. And that it is unmistakable about not being OpenSn, in the log, in the
runner name, and in the cache key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoopensn.parse import parse_stdout
from autoopensn.parse.gold import compare_to_gold
from autoopensn.runner import ReferenceRunner, RunRequest, UnsupportedScenario
from autoopensn.runner.reference import ADAPTERS, BANNER, supported
from autoopensn.store import cache_key
from autoopensn.templates import load_template


def request_for(tmp_path: Path, template: str, overrides: dict | None = None) -> RunRequest:
    loaded = load_template(template)
    return RunRequest(
        case_id="case",
        script="",
        directory=tmp_path / "case",
        template=template,
        parameters=loaded.merge(overrides or {}),
    )


# --- agreement with OpenSn --------------------------------------------------


def test_reed_defaults_pass_the_opensn_gold_check(tmp_path):
    """The whole justification for this runner, end to end through the parser."""
    result = ReferenceRunner().run(request_for(tmp_path, "reed_1d"))
    assert result.ok

    template = load_template("reed_1d")
    comparison = compare_to_gold(result.stdout, template, template.defaults())
    assert comparison.applicable
    assert comparison.passed, comparison.checks


def test_the_foundation_tutorial_passes_its_gold_check(tmp_path):
    result = ReferenceRunner().run(request_for(tmp_path, "first_1d_fixed_source"))
    assert result.ok

    template = load_template("first_1d_fixed_source")
    comparison = compare_to_gold(result.stdout, template, template.defaults())
    assert comparison.applicable
    assert comparison.passed, comparison.checks


# --- the log it writes ------------------------------------------------------


def test_the_log_says_it_is_not_opensn(tmp_path):
    result = ReferenceRunner().run(request_for(tmp_path, "reed_1d"))
    assert "NOT OpenSn" in result.stdout
    assert result.stdout.startswith("[AutoOpenSn]")


def test_the_log_parses_with_the_normal_parser(tmp_path):
    """Emitting OpenSn's own format is what keeps the pipeline single-path."""
    result = ReferenceRunner().run(request_for(tmp_path, "reed_1d"))
    observations = parse_stdout(result.stdout)
    assert observations.iterations is not None
    assert observations.sweeps == observations.iterations
    assert observations.converged is True
    assert observations.absorption_rate == pytest.approx(100.6178, abs=1e-4)
    assert observations.balance_residual is not None


def test_the_flux_is_emitted_when_the_spec_asks_for_it(tmp_path):
    result = ReferenceRunner().run(
        request_for(tmp_path, "reed_1d", {"emit_avg_flux": True})
    )
    assert parse_stdout(result.stdout).avg_flux is not None


def test_the_flux_is_absent_when_it_is_not_asked_for(tmp_path):
    result = ReferenceRunner().run(request_for(tmp_path, "reed_1d"))
    assert parse_stdout(result.stdout).avg_flux is None


def test_the_run_directory_is_written(tmp_path):
    result = ReferenceRunner().run(request_for(tmp_path, "reed_1d"))
    assert (result.directory / "stdout.txt").exists()
    assert (result.directory / "result.json").exists()


# --- convergence behaviour reaches the table --------------------------------


def test_a_tighter_tolerance_shows_up_as_more_sweeps(tmp_path):
    counts = []
    for tolerance in (1e-4, 1e-6, 1e-8):
        result = ReferenceRunner().run(
            request_for(tmp_path / f"t{tolerance}", "reed_1d", {"l_abs_tol": tolerance})
        )
        counts.append(parse_stdout(result.stdout).sweeps)
    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


def test_a_small_restart_interval_shows_up_as_more_sweeps(tmp_path):
    def sweeps(restart):
        result = ReferenceRunner().run(
            request_for(
                tmp_path / f"r{restart}", "reed_1d",
                {"gmres_restart_interval": restart, "l_abs_tol": 1e-9},
            )
        )
        return parse_stdout(result.stdout).sweeps

    assert sweeps(3) > sweeps(30)


def test_richardson_costs_more_sweeps_than_gmres(tmp_path):
    def sweeps(method, cap):
        result = ReferenceRunner().run(
            request_for(
                tmp_path / method, "reed_1d",
                {"inner_linear_method": method, "l_abs_tol": 1e-9, "l_max_its": cap},
            )
        )
        return parse_stdout(result.stdout).sweeps

    assert sweeps("classic_richardson", 2000) > sweeps("petsc_gmres", 300)


def test_hitting_the_iteration_cap_is_reported(tmp_path):
    result = ReferenceRunner().run(
        request_for(tmp_path, "reed_1d", {"l_abs_tol": 1e-14, "l_max_its": 3})
    )
    assert parse_stdout(result.stdout).inner_status == "iteration_limit"


# --- refusals ---------------------------------------------------------------


def test_it_supports_only_what_it_can_solve():
    assert supported("reed_1d")
    assert not supported("transport_2d_1_poly_balance")
    assert set(ADAPTERS) == {"reed_1d", "first_1d_fixed_source"}


def test_an_unsupported_scenario_raises_rather_than_approximating(tmp_path):
    request = RunRequest(
        case_id="case", script="", directory=tmp_path / "c",
        template="mesh_ortho_2d", parameters={"anything": 1},
    )
    with pytest.raises(UnsupportedScenario) as excinfo:
        ReferenceRunner().run(request)
    assert "one-dimensional" in str(excinfo.value)


def test_a_lenient_runner_reports_it_as_a_failed_row_instead(tmp_path):
    request = RunRequest(
        case_id="case", script="", directory=tmp_path / "c",
        template="mesh_ortho_2d", parameters={"anything": 1},
    )
    result = ReferenceRunner(strict=False).run(request)
    assert not result.ok
    assert "one-dimensional" in result.stderr


def test_a_reflecting_boundary_is_refused(tmp_path):
    result = ReferenceRunner(strict=False).run(
        request_for(tmp_path, "reed_1d", {"bc_zmin_type": "reflecting"})
    )
    assert not result.ok
    assert "vacuum only" in result.stderr


# --- identity ---------------------------------------------------------------


def test_its_results_are_not_interchangeable_with_opensn_results():
    """A number from here and a number from OpenSn are different measurements."""
    from autoopensn.runner import FakeRunner

    reference = ReferenceRunner().config()
    assert reference["runner"] == "reference"
    assert reference["solver"] == "sn1d"

    key_here = cache_key("print(1)", "abc", reference)
    key_opensn = cache_key("print(1)", "abc", {"runner": "local", "launcher": "mpiexec"})
    assert key_here != key_opensn


def test_the_banner_names_the_module_that_computed_it():
    assert "autoopensn.solvers.sn1d" in BANNER
