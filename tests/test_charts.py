"""Charts are derived from what a study varied and what it produced."""

from __future__ import annotations

import json

import pytest

from autoopensn.viz.charts import comparison_chart, measures_in, study_charts

vlc = pytest.importorskip("vl_convert", reason="vl-convert is the renderer used to check specs")


def renders(chart):
    """The spec is valid Vega-Lite if the reference renderer draws it."""
    return vlc.vegalite_to_svg(json.dumps(chart)).startswith("<svg")


TOLERANCES = [
    {"case": f"l_abs_tol={t}", "param_l_abs_tol": t, "sweeps": s, "rel_flux_diff": d,
     "wall_time": w, "converged": True, "is_reference": False}
    for t, s, d, w in ((1e-4, 6, 2.7e-6, 0.25), (1e-6, 8, 1.4e-8, 0.31), (1e-8, 10, 1.0e-10, 0.37))
] + [{"case": "reference", "param_l_abs_tol": 1e-10, "sweeps": 11, "rel_flux_diff": 0.0,
      "wall_time": 0.4, "converged": True, "is_reference": True}]


def test_a_tolerance_sweep_gets_a_log_axis():
    """1e-4, 1e-6, 1e-8 on a linear axis are one point and two on top of each other."""
    chart = study_charts(TOLERANCES, ["l_abs_tol"])[0]
    x = chart["concat"][0]["layer"][0]["encoding"]["x"]
    assert x["scale"] == {"type": "log"}
    assert renders(chart)


def test_the_reference_case_is_not_one_of_the_points():
    chart = study_charts(TOLERANCES, ["l_abs_tol"])[0]
    cases = {v["case"] for v in chart["concat"][0]["data"]["values"]}
    assert "reference" not in cases


def test_flux_difference_is_on_a_log_axis_and_zero_is_left_off():
    """The reference's own difference is 0, which a log axis cannot show."""
    chart = study_charts(TOLERANCES, ["l_abs_tol"])[0]
    panel = next(p for p in chart["concat"] if p["title"] == "Relative flux difference")
    assert panel["layer"][0]["encoding"]["y"]["scale"] == {"type": "log"}
    assert all(v["rel_flux_diff"] > 0 for v in panel["data"]["values"])


def test_a_choice_is_a_bar_chart():
    rows = [
        {"case": "a", "param_inner_linear_method": "classic_richardson", "sweeps": 100, "converged": False},
        {"case": "b", "param_inner_linear_method": "petsc_gmres", "sweeps": 15, "converged": True},
    ]
    chart = study_charts(rows, ["inner_linear_method"])[0]
    assert chart["concat"][0]["mark"] == {"type": "bar"}
    assert renders(chart)


def test_an_unconverged_case_is_marked_as_one():
    """Its sweep count is the cap it hit; drawn like the rest it reads as a result."""
    rows = [
        {"case": "a", "param_inner_linear_method": "classic_richardson", "sweeps": 100, "converged": False},
        {"case": "b", "param_inner_linear_method": "petsc_gmres", "sweeps": 15, "converged": True},
    ]
    values = study_charts(rows, ["inner_linear_method"])[0]["concat"][0]["data"]["values"]
    assert {v["case"]: v["converged"] for v in values} == {"a": "no", "b": "yes"}


def test_a_measure_that_does_not_vary_is_left_out():
    rows = [dict(r, peak_memory_gb=0.1) for r in TOLERANCES]
    assert "peak_memory_gb" not in [m[0] for m in measures_in(rows)]


def test_a_study_that_varied_nothing_has_no_charts():
    assert study_charts([{"case": "baseline", "sweeps": 11}], []) == []


def test_a_grid_sweep_gets_a_heatmap():
    rows = [
        {"case": f"{a},{b}", "param_a": a, "param_b": b, "sweeps": a * b, "converged": True}
        for a in (1, 2) for b in (3, 4)
    ]
    charts = study_charts(rows, ["a", "b"], grid=True)
    assert charts[0]["concat"][0]["mark"] == {"type": "rect"}
    assert renders(charts[0])


def test_studies_in_a_group_overlay_on_one_chart():
    def study(name, sweeps):
        return {"name": name, "created_at": "2026-11-11T12:00:00",
                "cases": [{"row": {"case": f"t={t}", "param_l_abs_tol": t, "sweeps": s,
                                   "converged": True, "is_reference": False}}
                          for t, s in zip((1e-4, 1e-6, 1e-8), sweeps)]}
    chart = comparison_chart([study("first", (6, 8, 10)), study("second", (7, 9, 12))], "l_abs_tol")
    assert {v["study"].split(" ·")[0] for v in chart["data"]["values"]} == {"first", "second"}
    assert chart["encoding"]["x"]["scale"] == {"type": "log"}
    assert renders(chart)


# --- charts in words, for the narration ----------------------------------------------


def test_a_chart_is_described_with_the_points_it_draws():
    from autoopensn.viz.charts import describe_chart

    text = "\n".join(describe_chart(study_charts(TOLERANCES, ["l_abs_tol"])[0]))
    assert 'Chart "Varying l_abs_tol"' in text
    assert "Sweeps against l_abs_tol (log axis)" in text
    assert "Relative flux difference (log axis)" in text
    assert "0.0001 -> 6" in text and "1e-08 -> 10" in text
    assert "reference" not in text, "the reference is not drawn, so it is not described"


def test_an_unconverged_case_is_named_in_the_description():
    from autoopensn.viz.charts import describe_chart

    rows = [
        {"case": "a", "param_inner_linear_method": "classic_richardson", "sweeps": 100, "converged": False},
        {"case": "b", "param_inner_linear_method": "petsc_gmres", "sweeps": 15, "converged": True},
    ]
    text = "\n".join(describe_chart(study_charts(rows, ["inner_linear_method"])[0]))
    assert "drawn as bars" in text
    assert "classic_richardson -> 100 (did not converge)" in text
    assert "did not converge: a." in text


def test_a_heatmap_is_described_cell_by_cell():
    from autoopensn.viz.charts import describe_chart

    rows = [{"case": f"{a},{b}", "param_a": a, "param_b": b, "sweeps": a * b, "converged": True}
            for a in (1, 2) for b in (3, 4)]
    text = "\n".join(describe_chart(study_charts(rows, ["a", "b"], grid=True)[0]))
    assert "Heatmap of Sweeps" in text and "2/4 = 8" in text
