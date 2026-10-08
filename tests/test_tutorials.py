"""Tests for turning OpenSn tutorials into runnable scenarios.

The load-bearing claim is the same one the hand-written Reed template makes, and
it is checked here for every generated template: rendering at default parameters
reproduces the tutorial's script character for character. A generator that
cannot guarantee that is one that quietly edits people's physics, and the whole
point of deriving templates from upstream tutorials is that the physics is not
ours to edit.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from autoopensn import kp_bridge
from autoopensn.templates import list_templates, load_template
from autoopensn.tutorials import catalog, detect
from autoopensn.tutorials.catalog import Scenario
from autoopensn.tutorials.generate import choose_style, knobs_for, render_template_source
from autoopensn.tutorials.notebook import Notebook, NotebookError, script
from autoopensn.tutorials.parameters import constraint_for, group_for

needs_checkout = pytest.mark.skipif(
    not kp_bridge.checkout_available(), reason="the OpenSn checkout is not present"
)

GENERATED = [
    name
    for name in list_templates()
    if load_template(name).generated_from
]


# --- the committed catalog --------------------------------------------------


def test_the_catalog_is_committed_and_populated():
    scenarios = catalog.load()
    assert len(scenarios) >= 40
    assert all(
        scenario.notebook.endswith(".ipynb")
        for scenario in scenarios
        if scenario.section != "benchmarks"
    )


def test_hand_written_templates_are_in_the_catalog_too():
    """The catalog is the menu a request is matched against.

    A runnable scenario missing from it is a scenario nobody can ask for, so it
    holds the hand-written benchmarks as well as the generated tutorials.
    """
    reed = catalog.by_name("reed_1d")
    assert reed is not None
    assert reed.has_gold
    assert "l_abs_tol" in reed.solver_parameters


def test_nearly_every_scenario_has_a_recorded_answer():
    scenarios = catalog.load()
    with_gold = [s for s in scenarios if s.has_gold]
    assert len(with_gold) / len(scenarios) > 0.9


def test_scenarios_know_their_own_parameters():
    scenarios = catalog.load()
    sweepable = [s for s in scenarios if s.sweepable]
    assert len(sweepable) / len(scenarios) > 0.95


def test_a_demonstration_tutorial_is_marked_unsweepable():
    """xs_read_ascii reads a fixed file and prints it. There is nothing to vary."""
    scenario = catalog.by_name("xs_read_ascii")
    assert scenario is not None
    assert not scenario.sweepable
    assert "demonstration only" in scenario.catalog_entry()


def test_an_unsweepable_scenario_is_not_offered_first():
    ranked = catalog.rank("read cross sections from an ascii file", catalog.load(), limit=3)
    assert ranked[0].name != "xs_read_ascii"


def test_scenario_names_are_unique():
    names = [scenario.name for scenario in catalog.load()]
    assert len(set(names)) == len(names)


# --- ranking ----------------------------------------------------------------


def test_ranking_finds_scenarios_by_their_parameters():
    """A request usually names the knob, and the knob belongs to a scenario."""
    ranked = catalog.rank("vary the gmres restart interval", catalog.load(), limit=5)
    assert ranked
    assert all("gmres_restart_interval" in s.parameters for s in ranked[:3])


def test_ranking_finds_scenarios_by_topic():
    ranked = catalog.rank("angular quadrature order", catalog.load(), limit=4)
    assert any("quad" in s.name or "quad" in s.title.lower() for s in ranked)


def test_ranking_an_empty_query_still_returns_something():
    assert catalog.rank("", catalog.load(), limit=3)


# --- the byte-exact claim, for every generated template ---------------------


@needs_checkout
@pytest.mark.parametrize("name", GENERATED)
def test_generated_template_reproduces_its_tutorial(name):
    template = load_template(name)
    notebook = kp_bridge.checkout_path() / "doc/source/tutorials" / template.generated_from
    expected = script(notebook)
    assert template.render() == expected, f"{name} no longer reproduces its tutorial"


@pytest.mark.parametrize("name", GENERATED)
def test_generated_template_renders_valid_python(name):
    ast.parse(load_template(name).render())


@pytest.mark.parametrize("name", GENERATED)
def test_generated_template_declares_where_it_came_from(name):
    template = load_template(name)
    assert template.generated_from.endswith(".ipynb")
    assert template.opensn_commit == "2fd4a19ceade4f8da581a5029c68f474d3333ccb"


@pytest.mark.parametrize("name", GENERATED)
def test_every_generated_default_is_within_its_own_range(name):
    for declaration in load_template(name).parameters.values():
        declaration.validate(declaration.default)


@pytest.mark.parametrize("name", GENERATED)
def test_every_generated_parameter_records_its_provenance(name):
    """A guard and a real code limit must not look alike in a sidecar."""
    for declaration in load_template(name).parameters.values():
        assert declaration.provenance in {"opensn", "physics", "inferred", "declared"}


def test_opensn_constraints_are_used_where_they_exist():
    template = load_template("first_1d_fixed_source")
    assert template.parameters["l_abs_tol"].provenance == "opensn"
    assert template.parameters["gmres_restart_interval"].minimum == 1
    assert "petsc_gmres" in template.parameters["inner_linear_method"].choices


def test_physics_constraints_bound_a_scattering_ratio():
    template = load_template("first_1d_fixed_source")
    ratio = template.parameters["c"]
    assert ratio.provenance == "physics"
    assert ratio.maximum == 1.0


def test_generated_templates_carry_their_gold_values():
    template = load_template("first_1d_fixed_source")
    assert template.gold["values"]["FOUNDATION_1D_MAX_FLUX="] == pytest.approx(0.962393909)


# --- notation preservation --------------------------------------------------


def test_style_detection_distinguishes_the_three_float_notations():
    assert choose_style(1.0e-10, "1.0e-10") == "sci"
    assert choose_style(4.0, "4.0") == "plain"
    assert choose_style(2.0, "2.") == "dot"


def test_style_detection_gives_up_rather_than_corrupting():
    """An unreproducible notation stops being a knob. It never becomes a wrong one."""
    assert choose_style(1.0, "1.00000") is None


def test_an_unreproducible_literal_is_skipped_not_mangled():
    source = "x = 1.00000\ny = 2.0\n"
    kept, skipped = knobs_for(source)
    assert [knob.name for knob in kept] == ["y"]
    assert [item.name for item in skipped] == ["x"]


def test_substitution_leaves_everything_else_alone():
    source = 'a = 1  # a comment\n\n\nb = "two"\n'
    kept, _ = knobs_for(source)
    rendered = render_template_source(source, kept)
    assert "# a comment" in rendered
    assert rendered.count("\n\n\n") == 1


# --- detection --------------------------------------------------------------


def test_assignments_keywords_and_dict_entries_are_all_found():
    source = (
        "num_cells = 40\n"
        "q = Quadrature(n_polar=8)\n"
        'groupset = {"groups_from_to": (0, 0), "l_abs_tol": 1.0e-6}\n'
    )
    names = [item.name for item in detect(source)]
    assert "num_cells" in names
    assert "n_polar" in names
    assert "l_abs_tol" in names


def test_a_long_list_is_data_not_a_knob():
    source = "nodes = [" + ", ".join(str(i) for i in range(30)) + "]\n"
    assert detect(source) == []


def test_a_computed_value_is_not_a_knob():
    source = "nodes = [4.0 * i / 40 for i in range(41)]\n"
    assert detect(source) == []


def test_a_tuple_is_left_alone():
    """Varying one end of a group range on its own is meaningless."""
    source = 'gs = {"groups_from_to": (0, 9)}\n'
    assert [item.name for item in detect(source)] == []


def test_structural_arguments_are_excluded():
    source = "v = RPPLogicalVolume(infx=True, zmin=0.0)\n"
    names = [item.name for item in detect(source)]
    assert "infx" not in names
    assert "zmin" in names


def test_a_generic_key_is_qualified_by_where_it_was_found():
    """Two boundaries both have a 'type'; one knob for both would be a trap."""
    source = 'bcs = [{"name": "zmin", "type": "vacuum"}, {"name": "zmax", "type": "reflecting"}]\n'
    names = [item.name for item in detect(source)]
    assert "zmin_type" in names
    assert "zmax_type" in names


def test_overlapping_knobs_keep_the_outer_one():
    source = "s = Source(group_strength=[1.0])\n"
    items = detect(source)
    assert [item.name for item in items] == ["group_strength"]
    assert items[0].default == [1.0]


def test_grouping_puts_solver_knobs_in_the_solver_group():
    assert group_for("l_abs_tol") == "solver"
    assert group_for("gmres_restart_interval") == "solver"
    assert group_for("num_cells") == "mesh"
    assert group_for("sigma_t") == "materials"
    assert group_for("n_polar") == "quadrature"


def test_an_inferred_bound_is_scaled_from_the_default():
    constraint = constraint_for("some_unknown_knob", 40)
    assert constraint.provenance == "inferred"
    assert constraint.minimum == 0
    assert constraint.maximum >= 40


# --- notebooks --------------------------------------------------------------


def test_reading_a_notebook(tmp_path):
    path = tmp_path / "x.ipynb"
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {"cell_type": "markdown", "source": ["# A Title\n", "\n", "Some prose.\n"]},
                    {"cell_type": "code", "source": ["a = 1\n"]},
                    {"cell_type": "code", "source": ["b = 2\n"]},
                ]
            }
        )
    )
    notebook = Notebook.read(path)
    assert notebook.title == "A Title"
    assert notebook.summary == "Some prose."
    assert notebook.script == "a = 1\n\nb = 2\n"


def test_a_notebook_with_no_code_is_an_error(tmp_path):
    path = tmp_path / "x.ipynb"
    path.write_text(json.dumps({"cells": [{"cell_type": "markdown", "source": ["hi"]}]}))
    with pytest.raises(NotebookError):
        Notebook.read(path)


def test_malformed_json_is_an_error(tmp_path):
    path = tmp_path / "x.ipynb"
    path.write_text("{not json")
    with pytest.raises(NotebookError):
        Notebook.read(path)


# --- drift against the checkout ---------------------------------------------


@needs_checkout
def test_the_committed_catalog_matches_the_checkout():
    """The committed artifacts are generated; this is what catches them going stale."""
    from autoopensn.tutorials.build import tutorials_root

    fresh = catalog.scan(tutorials_root())
    committed = [s for s in catalog.load() if s.section != "benchmarks"]
    assert [s.name for s in fresh] == [s.name for s in committed]
    assert [s.parameters for s in fresh] == [s.parameters for s in committed]
