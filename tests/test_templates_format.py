"""Tests for the number formatters and the sidecar loader.

The formatters look trivial and are not. Reproducing the OpenSn test suite's
two different float notations is what makes the byte-exact template test
possible, and neither notation is what any standard Python formatter produces.
"""

from __future__ import annotations

import pytest
import yaml

from autoopensn.templates.registry import (
    ParameterDeclaration,
    TemplateError,
    format_dot_float,
    format_int,
    format_sci_float,
    load_template,
)


# --- the trailing-dot notation used for material and geometry constants -----


@pytest.mark.parametrize(
    "value,expected",
    [
        (2.0, "2."),
        (1.0, "1."),
        (0.0, "0."),
        (50.0, "50."),
        (0.9, "0.9"),
        (1.5, "1.5"),
        (-2.0, "-2."),
        (0.001, "0.001"),
    ],
)
def test_dot_float(value, expected):
    assert format_dot_float(value) == expected


# --- the tolerance notation -------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (1.0e-9, "1.0e-9"),
        (1.0e-4, "1.0e-4"),
        (1.0e-10, "1.0e-10"),
        (2.5e-6, "2.5e-6"),
        (1.0e-18, "1.0e-18"),
        (0.0, "0.0"),
    ],
)
def test_sci_float(value, expected):
    assert format_sci_float(value) == expected


def test_sci_float_round_trips():
    for value in (1.0e-9, 2.5e-6, 1.0e-18, 3.7e-13):
        assert float(format_sci_float(value)) == value


def test_non_finite_values_are_refused():
    with pytest.raises(TemplateError):
        format_dot_float(float("nan"))
    with pytest.raises(TemplateError):
        format_sci_float(float("inf"))


def test_format_int_refuses_to_truncate():
    assert format_int(30) == "30"
    assert format_int(30.0) == "30"
    with pytest.raises(TemplateError):
        format_int(30.5)


# --- parameter declarations -------------------------------------------------


def declaration(**overrides) -> ParameterDeclaration:
    base = dict(name="x", group="solver", type="float", default=1.0)
    base.update(overrides)
    return ParameterDeclaration(**base)


def test_bool_is_not_an_integer():
    """'l_max_its: true' must not quietly become one iteration."""
    with pytest.raises(TemplateError):
        declaration(type="int").validate(True)


def test_integer_parameters_reject_fractions():
    with pytest.raises(TemplateError):
        declaration(type="int").validate(2.5)


def test_bool_parameters_reject_numbers():
    with pytest.raises(TemplateError):
        declaration(type="bool").validate(1)


def test_string_parameters_reject_numbers():
    with pytest.raises(TemplateError):
        declaration(type="str").validate(3)


def test_list_parameters_reject_scalars():
    with pytest.raises(TemplateError):
        declaration(type="float_list").validate(1.0)


def test_list_parameters_reject_strings():
    """A string is a sequence; it is not a list of floats."""
    with pytest.raises(TemplateError):
        declaration(type="float_list").validate("1.0")


def test_range_applies_to_every_list_element():
    decl = declaration(type="float_list", minimum=0.0, maximum=1.0)
    decl.validate([0.0, 0.5, 1.0])
    with pytest.raises(TemplateError):
        decl.validate([0.0, 1.5])


def test_length_is_enforced():
    decl = declaration(type="int_list", length=3)
    decl.validate([1, 2, 3])
    with pytest.raises(TemplateError):
        decl.validate([1, 2])


def test_choices_are_enforced():
    decl = declaration(type="str", default="a", choices=("a", "b"))
    assert decl.validate("b") == "b"
    with pytest.raises(TemplateError):
        decl.validate("c")


# --- sidecar loading --------------------------------------------------------


def write_template(root, sidecar: dict, script: str = "x = 1\n") -> str:
    directory = root / "custom"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "custom.py.j2").write_text(script)
    (directory / "custom.yaml").write_text(yaml.safe_dump(sidecar))
    return "custom"


def test_sidecar_rejects_unknown_top_level_keys(tmp_path):
    write_template(tmp_path, {"name": "custom", "parameters": {}, "mystery": 1})
    with pytest.raises(TemplateError) as excinfo:
        load_template("custom", root=tmp_path)
    assert "mystery" in str(excinfo.value)


def test_sidecar_rejects_unknown_declaration_keys(tmp_path):
    write_template(
        tmp_path,
        {
            "name": "custom",
            "parameters": {"a": {"group": "solver", "type": "int", "default": 1, "units": "cm"}},
        },
    )
    with pytest.raises(TemplateError) as excinfo:
        load_template("custom", root=tmp_path)
    assert "units" in str(excinfo.value)


def test_sidecar_rejects_an_unknown_group(tmp_path):
    write_template(
        tmp_path,
        {"name": "custom", "parameters": {"a": {"group": "physics", "type": "int", "default": 1}}},
    )
    with pytest.raises(TemplateError) as excinfo:
        load_template("custom", root=tmp_path)
    assert "physics" in str(excinfo.value)


def test_sidecar_rejects_a_default_outside_its_own_range(tmp_path):
    write_template(
        tmp_path,
        {
            "name": "custom",
            "parameters": {"a": {"group": "solver", "type": "int", "default": 99, "maximum": 10}},
        },
    )
    with pytest.raises(TemplateError):
        load_template("custom", root=tmp_path)


def test_sidecar_rejects_an_unsigned_yaml_exponent(tmp_path):
    """YAML 1.1 loads '1.0e6' as a string, which would poison every comparison."""
    directory = tmp_path / "custom"
    directory.mkdir(parents=True)
    (directory / "custom.py.j2").write_text("x = 1\n")
    (directory / "custom.yaml").write_text(
        "name: custom\n"
        "parameters:\n"
        "  a:\n"
        "    group: solver\n"
        "    type: float\n"
        "    default: 1.0\n"
        "    maximum: 1.0e6\n"
    )
    with pytest.raises(TemplateError) as excinfo:
        load_template("custom", root=tmp_path)
    assert "sign" in str(excinfo.value)


def test_sidecar_name_must_match_its_directory(tmp_path):
    write_template(tmp_path, {"name": "something_else", "parameters": {}})
    with pytest.raises(TemplateError) as excinfo:
        load_template("custom", root=tmp_path)
    assert "directory" in str(excinfo.value)


def test_missing_script_is_reported(tmp_path):
    directory = tmp_path / "custom"
    directory.mkdir(parents=True)
    (directory / "custom.yaml").write_text(yaml.safe_dump({"name": "custom", "parameters": {}}))
    with pytest.raises(TemplateError) as excinfo:
        load_template("custom", root=tmp_path)
    assert "custom.py.j2" in str(excinfo.value)


def test_a_typo_in_a_template_is_an_error_not_a_blank(tmp_path):
    """StrictUndefined: a misspelled variable must not render as empty."""
    name = write_template(
        tmp_path,
        {"name": "custom", "parameters": {"alpha": {"group": "solver", "type": "int", "default": 1}}},
        script="value = {{ alpah }}\n",
    )
    template = load_template(name, root=tmp_path)
    with pytest.raises(Exception):
        template.render()
