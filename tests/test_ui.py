"""Tests for the Streamlit interface.

Driven headlessly with Streamlit's own ``AppTest``, so the page is executed
rather than merely imported. A page that renders an empty table because an
attribute was renamed is exactly the failure that a smoke import misses.

The page writes run directories and a cache, so both are redirected into a
temporary directory. A test suite that dirties the working tree is one people
learn to run with their eyes closed.
"""

from __future__ import annotations

import pytest

from autoopensn.ui import APP_PATH

streamlit_testing = pytest.importorskip(
    "streamlit.testing.v1", reason="streamlit is an optional extra"
)
AppTest = streamlit_testing.AppTest


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTOOPENSN_RUN_ROOT", str(tmp_path / "runs"))
    monkeypatch.setenv("AUTOOPENSN_CACHE", str(tmp_path / "cache.db"))
    return AppTest.from_file(str(APP_PATH), default_timeout=180)


def test_the_page_loads_without_error(app):
    app.run()
    assert not app.exception, [str(e.value) for e in app.exception]
    assert not app.error, [e.value for e in app.error]


def test_the_scenario_library_is_listed(app):
    app.run()
    captions = " ".join(c.value for c in app.caption)
    assert "scenarios from the OpenSn tutorials" in captions


def _load_and_run(app, spec_path, runner=None):
    """Paste a spec into the editor and run it."""
    app.run()
    if runner is not None:
        app.radio[0].set_value(runner).run()
    editor = next(area for area in app.text_area if area.key == "spec_text")
    editor.set_value(spec_path.read_text()).run()
    next(b for b in app.button if "Run this study" in b.label).click().run()
    return app


def test_there_are_three_tabs():
    """An earlier version had five, two of which rendered the same panel."""
    source = APP_PATH.read_text()
    assert 'st.tabs(["Ask", "Scenarios", "Knowledge pack"])' in source


def test_the_results_panel_is_rendered_once(app, data_dir):
    """Rendering it twice is what forced widget-key namespacing before."""
    source = APP_PATH.read_text()
    assert source.count("show_results(") == 2  # one definition, one call
    assert source.count("show_explanation(") == 2


def test_running_a_spec_produces_the_table(app, data_dir):
    _load_and_run(app, data_dir / "gmres_convergence.yaml")
    assert not app.exception, [str(e.value) for e in app.exception]
    table = app.session_state["outcome_table"]
    assert len(table) == 7
    assert table["sweeps"].notna().all()
    assert table["rel_flux_diff"].notna().all()


def test_replayed_results_are_labelled_as_replayed(app, data_dir):
    """A page that renders invented numbers like measured ones is worse than none."""
    _load_and_run(app, data_dir / "gmres_convergence.yaml", runner="Recorded fixtures")
    warnings = " ".join(w.value for w in app.warning)
    assert "recorded fixtures" in warnings
    assert "hand-written" in warnings


def test_the_reference_solver_is_the_default_and_says_it_is_not_opensn(app):
    app.run()
    assert app.radio[0].value == "Reference solver"
    assert any("Not OpenSn" in message.value for message in app.info)


def test_the_reference_solver_computes_the_study_for_real(app, data_dir):
    """Not replayed: these numbers are computed when the button is pressed."""
    _load_and_run(app, data_dir / "gmres_convergence.yaml")
    table = app.session_state["outcome_table"]
    assert len(table) == 7
    assert not table["replayed"].any()
    assert table["sweeps"].notna().all()
    # The convergence trend the study exists to measure, computed rather than recorded.
    by_case = table.set_index("case_id")
    assert by_case.loc["l_abs_tol-1.0e-4", "sweeps"] < by_case.loc["l_abs_tol-1.0e-8", "sweeps"]


def test_an_invalid_spec_is_reported_not_raised(app):
    app.run()
    editor = next(area for area in app.text_area if area.key == "spec_text")
    editor.set_value(
        "name: broken\ntemplate: reed_1d\nsolver:\n  gmres_restart_interval: 0\n"
    ).run()
    assert not app.exception
    assert any("Spec rejected" in e.value for e in app.error)


def test_the_page_offers_to_generate_a_spec(app):
    app.run()
    assert any("Generate the spec" in b.label for b in app.button)


def test_nothing_is_generated_without_pressing_the_button(app):
    """The model stage must not fire on page load."""
    app.run()
    assert "draft" not in app.session_state
