"""Tests for the scenario spec.

The spec is the boundary between the language-model half of this tool and the
deterministic half, so most of these tests are about what it *refuses*. A spec
that loads is about to become a script that runs.
"""

from __future__ import annotations

import pytest
import yaml

from autoopensn.spec import Spec, SpecError, load_spec, save_spec
from autoopensn.spec.model import SweepDimension


def minimal(**overrides) -> dict:
    base = {
        "name": "test_study",
        "template": "reed_1d",
        "template_parameters": {},
        "solver": {},
    }
    base.update(overrides)
    return base


# --- structural validation --------------------------------------------------


def test_unknown_field_is_rejected():
    with pytest.raises(Exception) as excinfo:
        Spec(**minimal(sweep_dimensions=[{"parameter": "l_abs_tol", "values": [1]}]))
    assert "sweep_dimensions" in str(excinfo.value)


def test_name_and_template_must_be_non_empty():
    with pytest.raises(Exception):
        Spec(**minimal(name="   "))


def test_a_parameter_may_not_be_swept_twice():
    with pytest.raises(Exception) as excinfo:
        Spec(
            **minimal(
                sweep=[
                    {"parameter": "l_abs_tol", "values": [1.0e-4]},
                    {"parameter": "l_abs_tol", "values": [1.0e-6]},
                ]
            )
        )
    assert "l_abs_tol" in str(excinfo.value)


def test_sweep_values_must_be_distinct():
    with pytest.raises(Exception):
        SweepDimension(parameter="l_abs_tol", values=[1.0e-6, 1.0e-6])


def test_sweep_needs_at_least_one_value():
    with pytest.raises(Exception):
        SweepDimension(parameter="l_abs_tol", values=[])


# --- validation against the template ----------------------------------------


def test_unknown_parameter_is_rejected():
    spec = Spec(**minimal(solver={"no_such_knob": 3}))
    with pytest.raises(SpecError) as excinfo:
        spec.validate_against()
    assert "no_such_knob" in str(excinfo.value)
    # The message has to list what is available, because it is the only
    # feedback the spec-generation stage gets.
    assert "l_abs_tol" in str(excinfo.value)


def test_value_below_declared_minimum_is_rejected():
    spec = Spec(**minimal(solver={"gmres_restart_interval": 0}))
    with pytest.raises(SpecError) as excinfo:
        spec.validate_against()
    assert "minimum" in str(excinfo.value)


def test_value_above_declared_maximum_is_rejected():
    spec = Spec(**minimal(template_parameters={"n_polar": 4096}))
    with pytest.raises(SpecError) as excinfo:
        spec.validate_against()
    assert "maximum" in str(excinfo.value)


def test_value_outside_a_list_element_range_is_rejected():
    # The scattering ratio is bounded at 1.0; a multiplying medium is a
    # different problem, not a tuning of this one.
    spec = Spec(**minimal(template_parameters={"c": [0.0, 0.0, 0.0, 1.4, 0.9]}))
    with pytest.raises(SpecError) as excinfo:
        spec.validate_against()
    assert "1.4" in str(excinfo.value)


def test_wrong_list_length_is_rejected():
    spec = Spec(**minimal(template_parameters={"widths": [2.0, 1.0]}))
    with pytest.raises(SpecError) as excinfo:
        spec.validate_against()
    assert "5" in str(excinfo.value)


def test_value_outside_declared_choices_is_rejected():
    spec = Spec(**minimal(solver={"inner_linear_method": "petsc_cg"}))
    with pytest.raises(SpecError) as excinfo:
        spec.validate_against()
    assert "petsc_cg" in str(excinfo.value)


def test_solver_parameter_in_the_wrong_section_is_rejected():
    spec = Spec(**minimal(template_parameters={"l_abs_tol": 1.0e-6}))
    with pytest.raises(SpecError) as excinfo:
        spec.validate_against()
    assert "solver" in str(excinfo.value)


def test_physics_parameter_in_the_solver_section_is_rejected():
    spec = Spec(**minimal(solver={"n_polar": 64}))
    with pytest.raises(SpecError) as excinfo:
        spec.validate_against()
    assert "template_parameters" in str(excinfo.value)


def test_sweep_over_an_unknown_parameter_is_rejected():
    spec = Spec(**minimal(sweep=[{"parameter": "nonexistent", "values": [1, 2]}]))
    with pytest.raises(SpecError):
        spec.validate_against()


def test_sweep_value_outside_the_declared_range_is_rejected():
    spec = Spec(
        **minimal(sweep=[{"parameter": "gmres_restart_interval", "values": [20, 5000]}])
    )
    with pytest.raises(SpecError) as excinfo:
        spec.validate_against()
    assert "5000" in str(excinfo.value)


def test_mismatched_pack_commit_is_rejected():
    spec = Spec(**minimal(pack_commit="0" * 40))
    with pytest.raises(SpecError) as excinfo:
        spec.validate_against()
    assert "commit" in str(excinfo.value)


def test_unknown_observable_is_rejected():
    spec = Spec(**minimal(observables=["k_eff"]))
    with pytest.raises(SpecError) as excinfo:
        spec.validate_against()
    assert "k_eff" in str(excinfo.value)


def test_asking_for_flux_without_emitting_it_is_rejected():
    """The failure this catches is silent: a table of blank fluxes."""
    spec = Spec(**minimal(observables=["avg_flux"], template_parameters={}))
    with pytest.raises(SpecError) as excinfo:
        spec.validate_against()
    assert "emit_avg_flux" in str(excinfo.value)


def test_a_valid_spec_validates():
    spec = Spec(
        **minimal(
            template_parameters={"emit_avg_flux": True},
            solver={"l_abs_tol": 1.0e-6},
            observables=["avg_flux", "sweeps"],
        )
    )
    template = spec.validate_against()
    assert template.name == "reed_1d"


# --- expansion --------------------------------------------------------------


def test_no_sweep_is_one_point():
    spec = Spec(**minimal(observables=["sweeps"]))
    points = spec.expand()
    assert len(points) == 1
    assert points[0].case_id == "baseline"


def test_one_at_a_time_varies_each_dimension_alone(gmres_spec):
    points = gmres_spec.expand()
    assert len(points) == 7
    for point in points:
        assert len(point.varied) == 1


def test_grid_mode_is_the_cross_product():
    spec = Spec(
        **minimal(
            observables=["sweeps"],
            sweep_mode="grid",
            sweep=[
                {"parameter": "l_abs_tol", "values": [1.0e-4, 1.0e-6]},
                {"parameter": "gmres_restart_interval", "values": [5, 20, 50]},
            ],
        )
    )
    points = spec.expand()
    assert len(points) == 6
    assert all(len(point.varied) == 2 for point in points)


def test_duplicate_points_are_dropped():
    """A swept value equal to the baseline must not produce two identical runs."""
    spec = Spec(
        **minimal(
            observables=["sweeps"],
            solver={"gmres_restart_interval": 30},
            sweep=[{"parameter": "gmres_restart_interval", "values": [20, 30]}],
            reference={"gmres_restart_interval": 30},
        )
    )
    points = spec.expand()
    assert len(points) == 2
    # The reference coincides with a swept point, so that point becomes the
    # reference rather than being run a second time.
    assert sum(point.is_reference for point in points) == 1
    reference = next(point for point in points if point.is_reference)
    assert reference.parameters["gmres_restart_interval"] == 30


def test_reference_is_a_separate_point_when_it_differs(gmres_spec):
    points = gmres_spec.expand()
    reference = [point for point in points if point.is_reference]
    assert len(reference) == 1
    assert reference[0].case_id == "reference"
    assert reference[0].parameters["l_abs_tol"] == pytest.approx(1.0e-10)


def test_case_ids_are_unique_stable_and_readable(gmres_spec):
    ids = [point.case_id for point in gmres_spec.expand()]
    assert len(set(ids)) == len(ids)
    assert ids == [point.case_id for point in gmres_spec.expand()]
    assert "l_abs_tol-1.0e-4" in ids
    assert "gmres_restart_interval-5" in ids


def test_expanded_points_carry_every_template_parameter(gmres_spec, reed_template):
    for point in gmres_spec.expand():
        assert set(point.parameters) == set(reed_template.parameters)


def test_baseline_overrides_reach_every_point(gmres_spec):
    for point in gmres_spec.expand():
        assert point.parameters["emit_avg_flux"] is True
        assert point.parameters["inner_linear_method"] == "petsc_gmres"


# --- serialization ----------------------------------------------------------


def test_yaml_round_trip(tmp_path, gmres_spec):
    path = save_spec(gmres_spec, tmp_path / "spec.yaml")
    reloaded = load_spec(path)
    assert reloaded.model_dump() == gmres_spec.model_dump()


def test_yaml_is_readable(gmres_spec):
    """Field order is the order a person reads, not alphabetical."""
    text = gmres_spec.to_yaml()
    assert text.index("name:") < text.index("template:") < text.index("sweep:")
    assert yaml.safe_load(text)["template"] == "reed_1d"


def test_loading_a_bad_spec_names_the_file(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("name: x\ntemplate: reed_1d\nnonsense_field: 3\n")
    with pytest.raises(SpecError) as excinfo:
        load_spec(path)
    assert "broken.yaml" in str(excinfo.value)


def test_loading_non_yaml_is_an_error(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("- just\n- a list\n")
    with pytest.raises(SpecError):
        load_spec(path)
