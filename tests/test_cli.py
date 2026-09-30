"""Tests for the command line.

Mostly about the seams: that every stage runs on its own, that the two stubbed
stages say so instead of failing obscurely, and that a bad spec produces a
message rather than a traceback.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from autoopensn.cli import app

runner = CliRunner()


def test_version_reports_the_pinned_commit():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "2fd4a19ceade4f8da581a5029c68f474d3333ccb" in result.stdout


def test_templates_lists_reed():
    result = runner.invoke(app, ["templates"])
    assert result.exit_code == 0
    assert "reed_1d" in result.stdout


def test_templates_describes_parameters_and_their_ranges():
    result = runner.invoke(app, ["templates", "reed_1d"])
    assert result.exit_code == 0
    assert "gmres_restart_interval" in result.stdout
    assert "petsc_gmres" in result.stdout
    assert "reed_balance.py" in result.stdout


def test_render_writes_seven_scripts(tmp_path, data_dir):
    result = runner.invoke(
        app,
        ["render", str(data_dir / "gmres_convergence.yaml"), "-o", str(tmp_path / "out")],
    )
    assert result.exit_code == 0, result.stdout
    assert len(list((tmp_path / "out").glob("*.py"))) == 7


def test_render_does_not_run_anything(tmp_path, data_dir):
    runner.invoke(
        app, ["render", str(data_dir / "gmres_convergence.yaml"), "-o", str(tmp_path / "out")]
    )
    assert not (tmp_path / "out" / "reference" / "stdout.txt").exists()


def test_run_prints_the_table(tmp_path, data_dir, gmres_fixtures):
    result = runner.invoke(
        app,
        [
            "run",
            str(data_dir / "gmres_convergence.yaml"),
            "--fixtures",
            str(gmres_fixtures),
            "--run-root",
            str(tmp_path / "runs"),
            "--cache-path",
            str(tmp_path / "cache.db"),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert "Relative flux difference" in result.stdout
    assert "Sweeps" in result.stdout
    assert "7 cases" in result.stdout


def test_run_writes_csv(tmp_path, data_dir, gmres_fixtures):
    output = tmp_path / "results.csv"
    result = runner.invoke(
        app,
        [
            "run",
            str(data_dir / "gmres_convergence.yaml"),
            "--fixtures",
            str(gmres_fixtures),
            "--run-root",
            str(tmp_path / "runs"),
            "--cache-path",
            str(tmp_path / "cache.db"),
            "-o",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert output.exists()
    assert "avg_flux" in output.read_text().splitlines()[0]


def test_a_bad_spec_is_a_message_not_a_traceback(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\ntemplate: reed_1d\nsolver:\n  gmres_restart_interval: 0\n")
    result = runner.invoke(app, ["render", str(bad)])
    assert result.exit_code == 1
    assert "minimum" in result.stdout + str(result.output)


def test_an_unknown_runner_is_rejected(data_dir):
    result = runner.invoke(
        app, ["run", str(data_dir / "gmres_convergence.yaml"), "--runner", "slurm"]
    )
    assert result.exit_code == 1


# --- the two model-backed stages -------------------------------------------
#
# Driven with a scripted model. These tests never reach a provider: a suite that
# costs money per run is one people stop running.

GENERATED_SPEC = """
name: cli_study
template: reed_1d
template_parameters:
  emit_avg_flux: true
solver:
  l_abs_tol: 1.0e-6
sweep:
  - parameter: l_abs_tol
    values: [1.0e-4, 1.0e-6]
observables: [avg_flux, sweeps]
"""


@pytest.fixture
def scripted(monkeypatch):
    """Replace the provider with a scripted model for the whole CLI."""
    from autoopensn import llm as llm_module

    def make(responses):
        scripted_llm = llm_module.ScriptedLLM(list(responses))
        monkeypatch.setattr(llm_module, "default_llm", lambda **kwargs: scripted_llm)
        return scripted_llm

    return make


def test_spec_command_prints_a_validated_spec(scripted, monkeypatch):
    scripted([GENERATED_SPEC])
    monkeypatch.setattr(
        "autoopensn.narrate.spec_generation.catalog.load", lambda *a, **k: _one_scenario()
    )
    result = runner.invoke(app, ["spec", "vary the tolerance", "--no-review"])
    assert result.exit_code == 0, result.output
    assert "template: reed_1d" in result.stdout


def test_spec_command_writes_a_file(tmp_path, scripted, monkeypatch):
    scripted([GENERATED_SPEC])
    monkeypatch.setattr(
        "autoopensn.narrate.spec_generation.catalog.load", lambda *a, **k: _one_scenario()
    )
    out = tmp_path / "spec.yaml"
    result = runner.invoke(
        app, ["spec", "vary the tolerance", "--no-review", "-o", str(out)]
    )
    assert result.exit_code == 0, result.output
    assert out.exists()


def test_spec_command_reports_a_rejected_spec(scripted, monkeypatch):
    bad = "name: x\ntemplate: reed_1d\nsolver:\n  gmres_restart_interval: 0\n"
    scripted([bad, bad])
    monkeypatch.setattr(
        "autoopensn.narrate.spec_generation.catalog.load", lambda *a, **k: _one_scenario()
    )
    result = runner.invoke(app, ["spec", "break it", "--no-review"])
    assert result.exit_code == 1
    assert "no valid spec" in result.output


def test_explain_command_prints_prose(tmp_path, data_dir, gmres_fixtures, scripted):
    table = tmp_path / "results.csv"
    runner.invoke(
        app,
        ["run", str(data_dir / "gmres_convergence.yaml"), "--fixtures", str(gmres_fixtures),
         "--run-root", str(tmp_path / "runs"), "--cache-path", str(tmp_path / "c.db"),
         "-o", str(table)],
    )
    scripted(["A tighter tolerance costs more sweeps."])
    result = runner.invoke(
        app, ["explain", str(table), str(data_dir / "gmres_convergence.yaml")]
    )
    assert result.exit_code == 0, result.output
    assert "tighter tolerance" in result.stdout


def test_scenarios_command_lists_the_catalog():
    result = runner.invoke(app, ["scenarios", "--limit", "5"])
    assert result.exit_code == 0
    assert "scenario(s) available" in result.stdout


def test_scenarios_command_narrows_by_query():
    result = runner.invoke(app, ["scenarios", "quadrature", "--limit", "3"])
    assert result.exit_code == 0
    assert "quad" in result.stdout.lower()


def _one_scenario():
    from autoopensn.templates import load_template
    from autoopensn.tutorials.catalog import Scenario

    template = load_template("reed_1d")
    return [
        Scenario(
            name="reed_1d",
            title="Reed 1D problem",
            summary="A five-region slab benchmark.",
            notebook="foundations/reed.ipynb",
            section="foundations",
            parameters=list(template.parameters),
            solver_parameters=list(template.solver_parameters),
        )
    ]
