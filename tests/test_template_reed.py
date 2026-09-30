"""The template test that matters: the render equals the regression script.

This is the load-bearing claim of the whole project. If rendering the Reed
template at its defaults produces exactly the script the OpenSn regression suite
runs and checks against gold values, then a spec that only moves declared knobs
within declared ranges produces a script that is at least as trustworthy as
that one. If it does not, nothing downstream means anything.

Compared twice, against two different copies, because they catch opposite
failures. The vendored copy catches this repository's template drifting. The
live checkout catches the vendored copy going stale against upstream.
"""

from __future__ import annotations

import difflib

import pytest

from autoopensn import kp_bridge
from autoopensn.templates import TemplateError, load_template, render_run_script


def show(expected: str, actual: str) -> str:
    return "".join(
        difflib.unified_diff(
            expected.splitlines(True), actual.splitlines(True), "regression script", "render"
        )
    )


# --- the byte-exact claim ---------------------------------------------------


def test_default_render_matches_the_vendored_regression_script(reed_template, vendored_reed):
    rendered = reed_template.render()
    assert rendered == vendored_reed, show(vendored_reed, rendered)


def test_cbc_render_matches_the_vendored_cbc_regression_script(
    reed_template, vendored_reed_cbc
):
    """The CBC variant differs from the base script by exactly one line.

    Milestone 2 needs a solver-choice knob, and this is the cheapest possible
    evidence that the one in this template produces the real thing rather than
    something that merely looks like it.
    """
    rendered = reed_template.render({"sweep_type": "CBC"})
    assert rendered == vendored_reed_cbc, show(vendored_reed_cbc, rendered)


@pytest.mark.skipif(not kp_bridge.available(), reason="sister repository not present")
def test_vendored_copy_is_current_with_the_pinned_checkout(vendored_reed):
    live = kp_bridge.test_script("reed_balance.py").read_text()
    assert vendored_reed == live, show(live, vendored_reed)


# --- purity -----------------------------------------------------------------


def test_rendering_is_deterministic(reed_template):
    first = reed_template.render({"l_abs_tol": 1.0e-6})
    second = reed_template.render({"l_abs_tol": 1.0e-6})
    assert first == second


def test_rendering_does_not_mutate_the_template(reed_template):
    before = reed_template.defaults()
    reed_template.render({"l_max_its": 999})
    assert reed_template.defaults() == before


# --- what the knobs actually do ---------------------------------------------


def test_tolerance_renders_in_the_source_notation(reed_template):
    rendered = reed_template.render({"l_abs_tol": 1.0e-6})
    assert '"l_abs_tol": 1.0e-6,' in rendered
    assert "1e-06" not in rendered


def test_restart_interval_is_omitted_for_non_gmres_methods(reed_template):
    """It is a GMRES-only control; writing it elsewhere would be noise."""
    rendered = reed_template.render(
        {"inner_linear_method": "classic_richardson", "l_max_its": 1000}
    )
    assert "gmres_restart_interval" not in rendered
    assert '"inner_linear_method": "classic_richardson",' in rendered
    assert '"l_max_its": 1000,' in rendered


def test_restart_interval_is_present_for_gmres(reed_template):
    rendered = reed_template.render({"gmres_restart_interval": 5})
    assert '"gmres_restart_interval": 5,' in rendered


def test_flux_block_is_absent_by_default(reed_template):
    rendered = reed_template.render()
    assert "AvgFlux" not in rendered
    assert "FieldFunctionInterpolationVolume" not in rendered


def test_flux_block_brings_its_own_import(reed_template):
    """An emitted block that forgets its import is a script that dies at line 20."""
    rendered = reed_template.render({"emit_avg_flux": True})
    assert "from pyopensn.fieldfunc import FieldFunctionInterpolationVolume" in rendered
    assert 'print(f"AvgFlux={ffi_avg.GetValue():.9e}")' in rendered


def test_every_render_is_valid_python(reed_template):
    """Whatever the knobs do, the result has to at least parse."""
    import ast

    for overrides in (
        {},
        {"emit_avg_flux": True},
        {"sweep_type": "CBC", "emit_avg_flux": True},
        {"inner_linear_method": "petsc_bicgstab", "l_abs_tol": 1.0e-12},
        {"bc_zmin_type": "reflecting", "nrefs": [10, 10, 10, 10, 10]},
    ):
        ast.parse(reed_template.render(overrides))


def test_material_lists_keep_the_source_notation(reed_template):
    rendered = reed_template.render({"total": [40.0, 5.0, 0.0, 1.0, 1.5]})
    assert "total = [40., 5., 0., 1., 1.5]" in rendered


# --- refusals ---------------------------------------------------------------


def test_rendering_an_undeclared_parameter_is_an_error(reed_template):
    with pytest.raises(TemplateError):
        reed_template.render({"not_a_parameter": 1})


def test_rendering_out_of_range_is_an_error(reed_template):
    with pytest.raises(TemplateError):
        reed_template.render({"c": [0.0, 0.0, 0.0, 2.0, 0.9]})


def test_a_string_that_would_break_quoting_is_refused(reed_template):
    """Naive quoting is fine as long as it refuses what it cannot quote."""
    with pytest.raises(TemplateError):
        reed_template.render({"pyopensn_path": 'has a " quote'})


def test_a_comment_that_would_escape_its_line_is_refused(reed_template):
    """A newline in the header comment would inject the rest as code."""
    with pytest.raises(TemplateError):
        reed_template.render({"header_comment": "innocent\nimport os; os.system('x')"})


# --- provenance -------------------------------------------------------------


def test_run_script_records_the_pack_commit(reed_template):
    script = render_run_script(
        reed_template,
        {},
        pack_commit="abc123",
        spec_name="study",
        case_id="case_one",
        varied={"l_abs_tol": 1.0e-6},
    )
    assert "# pack commit: abc123" in script
    assert "# case:        case_one" in script
    assert "l_abs_tol = 1e-06" in script


def test_run_script_is_the_render_plus_a_header(reed_template, vendored_reed):
    script = render_run_script(
        reed_template, {}, pack_commit="abc123", spec_name="s", case_id="c"
    )
    assert script.endswith(vendored_reed)


def test_run_script_header_is_comments_only(reed_template):
    import ast

    script = render_run_script(
        reed_template, {}, pack_commit="abc123", spec_name="s", case_id="c"
    )
    ast.parse(script)


# --- the sidecar itself -----------------------------------------------------


def test_sidecar_declares_the_observables_the_parser_produces(reed_template):
    assert set(reed_template.observables) == {
        "avg_flux",
        "sweeps",
        "iterations",
        "wall_time",
        "balance_residual",
    }


def test_sidecar_records_where_it_came_from(reed_template):
    assert reed_template.source.endswith("reed_balance.py")
    assert reed_template.opensn_commit == "2fd4a19ceade4f8da581a5029c68f474d3333ccb"


def test_every_declared_default_is_within_its_own_range(reed_template):
    """A sidecar whose default violates its own bound would break every render."""
    for declaration in reed_template.parameters.values():
        declaration.validate(declaration.default)


def test_solver_group_is_what_the_spec_expects(reed_template):
    assert set(reed_template.solver_parameters) == {
        "inner_linear_method",
        "l_abs_tol",
        "l_max_its",
        "gmres_restart_interval",
        "sweep_type",
    }


def test_listing_templates_finds_reed():
    from autoopensn.templates import list_templates

    assert "reed_1d" in list_templates()


def test_loading_an_unknown_template_lists_what_exists():
    with pytest.raises(TemplateError) as excinfo:
        load_template("no_such_template")
    assert "reed_1d" in str(excinfo.value)
