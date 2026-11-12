"""Charts of a study's results, worked out from the study's own structure.

Every chart here is a Vega-Lite specification: a declarative description of a
chart — data, marks, encodings — that a renderer draws. The web page draws them
with vega-embed; a report can render the same specification to SVG; an MCP
client can be handed it as is. One description, many renderers, which is what
makes this a systematic approach rather than a set of hand-made plots.

The charts are derived, not chosen by anyone. What a study varied decides the
chart type, and what it produced decides what is plotted:

- **A varied number** (a tolerance, a cell count) is a line chart. Values that
  span two orders of magnitude or more are plotted on a log axis, which is what
  tolerances need: 1e-4, 1e-6 and 1e-8 on a linear axis are one point and two
  that sit on top of each other.
- **A varied choice** (an inner solver, a sweep type) is a bar chart.
- **Two parameters varied together on a grid** are a heatmap per measure.
- **A measure is plotted only if it says something**: numeric, present for at
  least two cases, and not the same in all of them.
- **A case that did not converge is drawn differently**, and its tooltip says
  so. Its sweep count is the limit it was stopped at, not what the method
  needed, and a chart that drew it like the others would make that number look
  like a result.

Nothing here touches a language model or the network. Given the same rows it
produces the same charts.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

SCHEMA = "https://vega.github.io/schema/vega-lite/v5.json"

# The measures worth charting, in the order they are shown, with their labels
# and whether they belong on a log axis.
MEASURES: tuple[tuple[str, str, bool], ...] = (
    ("sweeps", "Sweeps", False),
    ("rel_flux_diff", "Relative flux difference", True),
    ("wall_time", "Simulation time (s)", False),
    ("avg_sweep_time", "Time per sweep (s)", False),
    ("solve_time", "Solve time (s)", False),
    ("peak_memory_gb", "Peak memory (GB)", False),
    ("sweep_time_per_unknown_ns", "Time per unknown (ns)", False),
)

MAROON = "#7c1d1d"
UNCONVERGED = "#c2410c"


def _number(value: Any) -> Optional[float]:
    """A finite float, or None for anything that is not one."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _says_something(values: Iterable[Any], log: bool = False) -> bool:
    numbers = [n for n in (_number(v) for v in values) if n is not None]
    if log:
        numbers = [n for n in numbers if n > 0]
    return len(numbers) >= 2 and len(set(numbers)) > 1


def _spans_decades(values: Sequence[float]) -> bool:
    positive = [v for v in values if v > 0]
    return len(positive) == len(values) and len(positive) >= 2 and max(positive) / min(positive) >= 100


def _converged_label(row: dict[str, Any]) -> str:
    value = row.get("converged")
    return "no" if value is False else ("yes" if value is True else "unknown")


def measures_in(rows: Sequence[dict[str, Any]], extra: Sequence[str] = ()) -> list[tuple[str, str, bool]]:
    """The measures these rows can usefully be charted on.

    ``extra`` names scenario-specific outputs — a detector response, a
    k-eigenvalue — to consider after the standard ones.
    """
    chosen = []
    candidates = list(MEASURES) + [(name, name.replace("_", " "), False) for name in extra]
    for column, label, log in candidates:
        if _says_something((row.get(column) for row in rows), log=log):
            chosen.append((column, label, log))
    return chosen


def _panel(
    values: list[dict[str, Any]],
    x_title: str,
    x_kind: str,
    x_log: bool,
    column: str,
    label: str,
    y_log: bool,
) -> dict[str, Any]:
    """One small chart: a measure against the varied parameter."""
    x = {"field": "x", "type": x_kind, "title": x_title}
    if x_kind == "quantitative":
        x["scale"] = {"type": "log"} if x_log else {"zero": False}
        x["axis"] = {"format": "~e"} if x_log else {}
    y = {"field": column, "type": "quantitative", "title": label,
         "scale": {"type": "log"} if y_log else {"zero": False}}
    if y_log:
        y["axis"] = {"format": "~e"}
    tooltip = [
        {"field": "case", "title": "Case"},
        {"field": column, "title": label, "format": ".4~g"},
        {"field": "converged", "title": "Converged"},
    ]
    colour = {
        "field": "converged", "type": "nominal", "title": "Converged",
        "scale": {"domain": ["yes", "no", "unknown"], "range": [MAROON, UNCONVERGED, "#8d877f"]},
    }
    plotted = [v for v in values if _number(v.get(column)) is not None and (not y_log or _number(v[column]) > 0)]
    if x_kind == "quantitative":
        layers = [
            {"mark": {"type": "line", "color": MAROON, "strokeWidth": 1.5},
             "encoding": {"x": x, "y": y}},
            {"mark": {"type": "point", "filled": True, "size": 70},
             "encoding": {"x": x, "y": y, "color": colour,
                          "shape": {"field": "converged", "type": "nominal", "legend": None,
                                    "scale": {"domain": ["yes", "no", "unknown"],
                                              "range": ["circle", "cross", "square"]}},
                          "tooltip": tooltip}},
        ]
        body: dict[str, Any] = {"layer": layers}
    else:
        body = {"mark": {"type": "bar"},
                "encoding": {"x": {**x, "sort": None}, "y": y, "color": colour, "tooltip": tooltip}}
    return {"width": 210, "height": 160, "data": {"values": plotted}, "title": label, **body}


def dimension_chart(
    rows: Sequence[dict[str, Any]],
    parameter: str,
    measures: Sequence[tuple[str, str, bool]],
) -> Optional[dict[str, Any]]:
    """Every chosen measure against one varied parameter, side by side."""
    column = f"param_{parameter}"
    varied = [r for r in rows if r.get(column) is not None and not r.get("is_reference")
              and not (isinstance(r.get(column), float) and math.isnan(r[column]))]
    if len(varied) < 2:
        return None
    raw = [r[column] for r in varied]
    numeric = all(_number(v) is not None for v in raw)
    x_kind = "quantitative" if numeric else "nominal"
    x_log = numeric and _spans_decades([float(v) for v in raw])
    values = []
    for row in varied:
        entry = {"x": float(row[column]) if numeric else str(row[column]),
                 "case": row.get("case"), "converged": _converged_label(row)}
        for name, _, _ in measures:
            entry[name] = _number(row.get(name))
        values.append(entry)
    panels = [
        _panel(values, parameter, x_kind, x_log, name, label, log)
        for name, label, log in measures
        if _says_something((v.get(name) for v in values), log=log)
    ]
    if not panels:
        return None
    return {
        "$schema": SCHEMA,
        "title": {"text": f"Varying {parameter}", "anchor": "start", "fontSize": 14},
        "concat": panels,
        "columns": 3,
        "resolve": {"scale": {"y": "independent", "color": "shared"}},
        "config": {"view": {"stroke": None}, "axis": {"grid": True, "gridOpacity": 0.4}},
    }


def grid_chart(
    rows: Sequence[dict[str, Any]],
    first: str,
    second: str,
    measures: Sequence[tuple[str, str, bool]],
) -> Optional[dict[str, Any]]:
    """A heatmap per measure, for two parameters varied together."""
    a, b = f"param_{first}", f"param_{second}"
    cells = [r for r in rows if r.get(a) is not None and r.get(b) is not None and not r.get("is_reference")]
    if len(cells) < 4:
        return None
    values = []
    for row in cells:
        entry = {"x": str(row[a]), "y": str(row[b]), "case": row.get("case"),
                 "converged": _converged_label(row)}
        for name, _, _ in measures:
            entry[name] = _number(row.get(name))
        values.append(entry)
    panels = []
    for name, label, log in measures:
        if not _says_something((v.get(name) for v in values), log=log):
            continue
        panels.append({
            "width": 210, "height": 180, "title": label, "data": {"values": values},
            "mark": {"type": "rect"},
            "encoding": {
                "x": {"field": "x", "type": "ordinal", "title": first, "sort": None},
                "y": {"field": "y", "type": "ordinal", "title": second, "sort": None},
                "color": {"field": name, "type": "quantitative", "title": label,
                          "scale": {"type": "log", "scheme": "reds"} if log else {"scheme": "reds"}},
                "tooltip": [{"field": "case"}, {"field": name, "format": ".4~g"},
                            {"field": "converged"}],
            },
        })
    if not panels:
        return None
    return {"$schema": SCHEMA, "title": {"text": f"{first} × {second}", "anchor": "start"},
            "concat": panels, "columns": 3, "resolve": {"scale": {"color": "independent"}}}


def study_charts(
    rows: Sequence[dict[str, Any]],
    swept: Sequence[str],
    *,
    grid: bool = False,
    extra_measures: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Every chart a study's rows support, as Vega-Lite specifications.

    Empty when there is nothing to compare: a study that varied nothing, or one
    whose cases all failed.
    """
    measures = measures_in(rows, extra_measures)
    if not measures:
        return []
    charts = []
    if grid and len(swept) == 2:
        chart = grid_chart(rows, swept[0], swept[1], measures)
        if chart:
            charts.append(chart)
    for parameter in swept:
        chart = dimension_chart(rows, parameter, measures)
        if chart:
            charts.append(chart)
    return charts


def comparison_chart(
    studies: Sequence[dict[str, Any]],
    parameter: str,
    measure: str = "sweeps",
) -> Optional[dict[str, Any]]:
    """Studies from one group, overlaid: the same varied parameter, one line each.

    ``studies`` are catalogue records with their cases. This is what makes a
    group quick to compare: last week's GMRES study and this week's on one
    axis, so a change shows as a gap between two lines.
    """
    label = dict((name, title) for name, title, _ in MEASURES).get(measure, measure)
    log = dict((name, lg) for name, _, lg in MEASURES).get(measure, False)
    column = f"param_{parameter}"
    values = []
    for study in studies:
        tag = f"{study.get('name')} · {str(study.get('created_at', ''))[:16].replace('T', ' ')}"
        for case in study.get("cases", []):
            row = case.get("row", {})
            x, y = row.get(column), _number(row.get(measure))
            if x is None or y is None or row.get("is_reference") or (log and y <= 0):
                continue
            values.append({"x": x, "y": y, "study": tag, "case": row.get("case"),
                           "converged": _converged_label(row)})
    if len({v["study"] for v in values}) < 1 or len(values) < 2:
        return None
    numeric = all(_number(v["x"]) is not None for v in values)
    x = {"field": "x", "type": "quantitative" if numeric else "nominal", "title": parameter}
    if numeric and _spans_decades([float(v["x"]) for v in values]):
        x["scale"] = {"type": "log"}
    return {
        "$schema": SCHEMA,
        "title": {"text": f"{label} against {parameter}, by study", "anchor": "start"},
        "width": 520, "height": 260,
        "data": {"values": values},
        "mark": {"type": "line", "point": {"filled": True, "size": 60}} if numeric else {"type": "bar"},
        "encoding": {
            "x": x,
            "y": {"field": "y", "type": "quantitative", "title": label,
                  "scale": {"type": "log"} if log else {"zero": False}},
            "color": {"field": "study", "type": "nominal", "title": "Study"},
            **({"xOffset": {"field": "study"}} if not numeric else {}),
            "tooltip": [{"field": "study"}, {"field": "case"}, {"field": "y", "title": label},
                        {"field": "converged"}],
        },
    }


# --- describing charts in words ---------------------------------------------------


def _g(value: Any) -> str:
    """A number as the chart's tooltip would show it: four significant figures."""
    number = _number(value)
    return "?" if number is None else f"{number:.4g}"


def describe_chart(chart: dict[str, Any]) -> list[str]:
    """What a chart from this module draws, in plain sentences.

    Built from the chart's own data and encodings, so the words describe the
    picture on the page exactly: the same points, the same axes, the same
    unconverged cases. This is what lets the narration explain a chart without
    ever being shown one.
    """
    title = chart.get("title")
    heading = title.get("text") if isinstance(title, dict) else (title or "Chart")
    panels = chart.get("concat") or [chart]
    lines = [f'Chart "{heading}", {len(panels)} panel{"s" if len(panels) != 1 else ""}:']
    for panel in panels:
        encoding = panel.get("encoding") or (panel.get("layer") or [{}])[-1].get("encoding", {})
        values = (panel.get("data") or chart.get("data") or {}).get("values", [])
        x, y, colour = encoding.get("x", {}), encoding.get("y", {}), encoding.get("color", {})
        label = panel.get("title") or y.get("title") or colour.get("title") or "value"
        if isinstance(label, dict):
            label = label.get("text", "value")
        mark = panel.get("mark", "line")
        mark = mark.get("type") if isinstance(mark, dict) else mark
        if mark == "rect":
            field = colour.get("field")
            scale = "log" if colour.get("scale", {}).get("type") == "log" else "linear"
            cells = "; ".join(f"{v.get('x')}/{v.get('y')} = {_g(v.get(field))}" for v in values
                              if _number(v.get(field)) is not None)
            lines.append(f"- Heatmap of {label} ({scale} colour scale, darker is higher) over "
                         f"{x.get('title')} (across) and {y.get('title')} (down): {cells}.")
            continue
        field = y.get("field")
        x_axis = x.get("title", "x") + (" (log axis)" if x.get("scale", {}).get("type") == "log" else "")
        y_axis = label + (" (log axis)" if y.get("scale", {}).get("type") == "log" else "")
        points = [v for v in values if _number(v.get(field)) is not None]
        if not points:
            continue
        kind = "bars" if mark == "bar" else "line"
        series = colour.get("field") if colour.get("field") == "study" else None
        shown = "; ".join(
            f"{v.get('x')} -> {_g(v.get(field))}"
            + (f" [{v.get('study')}]" if series else "")
            + (" (did not converge)" if v.get("converged") == "no" else "")
            for v in points
        )
        lowest = min(points, key=lambda v: _number(v[field]))
        highest = max(points, key=lambda v: _number(v[field]))
        lines.append(
            f"- {y_axis} against {x_axis}, drawn as {kind}: {shown}. "
            f"Lowest {_g(lowest[field])} at {lowest.get('x')}, highest {_g(highest[field])} at {highest.get('x')}."
        )
    unconverged = sorted({str(v.get("case")) for p in panels
                          for v in (p.get("data") or chart.get("data") or {}).get("values", [])
                          if v.get("converged") == "no"})
    if unconverged:
        lines.append("- Drawn in orange with a cross because they did not converge: " + ", ".join(unconverged) + ".")
    return lines
