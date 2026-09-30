"""Assembling parsed runs into a results table.

One row per sweep point, columns for the observables, and a relative flux
difference computed against the reference case when the spec declares one. The
table is the last deterministic artifact in the pipeline and the only thing the
narration stage is allowed to see.

A failed run gets a row too, with its observables blank and its failure reason
in a column. Dropping it would make a study of nine cases silently become a
study of eight, which is how a divergence becomes a conclusion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from autoopensn.parse.gold import compare_to_gold
from autoopensn.parse.observables import key_value, parse_stdout
from autoopensn.runner.base import RunResult
from autoopensn.spec.model import Spec, SweepPoint
from autoopensn.templates import Template


def observe(
    result: RunResult,
    point: SweepPoint,
    template: Optional[Template] = None,
    *,
    tests_json: Optional[Path] = None,
) -> dict[str, Any]:
    """One table row: the point, what it produced, and whether it worked."""
    observations = parse_stdout(result.stdout)

    row: dict[str, Any] = {
        "case_id": point.case_id,
        "case": point.label(),
        "is_reference": point.is_reference,
        "ok": result.ok,
        "failure": result.failure_reason(),
        "wall_time": result.wall_time,
        "replayed": result.replayed,
    }
    row.update(observations.as_row())

    # Each scenario prints its own quantities, and its tests.json says which
    # ones upstream thought were worth checking. Those become columns too, so a
    # tutorial that reports a detector response is tabulated on the detector
    # response rather than on the five generic observables alone.
    if template is not None:
        for key, observable in (template.gold or {}).get("keys", {}).items():
            row[observable] = key_value(result.stdout, key)

    for name, value in sorted(point.varied.items()):
        row[f"param_{name}"] = value

    if template is not None:
        comparison = compare_to_gold(
            result.stdout, template, point.parameters, tests_json=tests_json
        )
        row.update(comparison.as_row())

    return row


def results_table(
    results: Sequence[RunResult],
    points: Sequence[SweepPoint],
    template: Optional[Template] = None,
    *,
    tests_json: Optional[Path] = None,
) -> pd.DataFrame:
    """A DataFrame of every sweep point, with a relative flux difference column.

    ``results`` and ``points`` are matched by case id, not by position, so a
    study that ran out of order or partially still tabulates correctly.
    """
    by_case = {result.case_id: result for result in results}
    rows = []
    for point in points:
        result = by_case.get(point.case_id)
        if result is None:
            rows.append(
                {
                    "case_id": point.case_id,
                    "case": point.label(),
                    "is_reference": point.is_reference,
                    "ok": False,
                    "failure": "not run",
                    "wall_time": None,
                }
            )
            continue
        rows.append(observe(result, point, template, tests_json=tests_json))

    frame = pd.DataFrame(rows)
    return _add_relative_flux_difference(frame)


def _add_relative_flux_difference(frame: pd.DataFrame) -> pd.DataFrame:
    """Relative flux difference of each case against the reference case.

    Absent a reference case, or absent a flux for it, the column is all NaN
    rather than silently comparing against the first row. A study that forgot
    to declare a reference should produce blanks, not a plausible number
    measured from an arbitrary baseline.
    """
    frame = frame.copy()
    frame["rel_flux_diff"] = pd.NA

    if "avg_flux" not in frame.columns or "is_reference" not in frame.columns:
        return frame

    reference_rows = frame[frame["is_reference"].fillna(False).astype(bool)]
    if reference_rows.empty:
        return frame

    reference_flux = reference_rows.iloc[0]["avg_flux"]
    if pd.isna(reference_flux) or reference_flux == 0:
        return frame

    frame["rel_flux_diff"] = (
        (frame["avg_flux"] - reference_flux).abs() / abs(reference_flux)
    )
    return frame


def convergence_table(frame: pd.DataFrame) -> pd.DataFrame:
    """The four-column view the GMRES convergence study is judged on.

    Case, relative flux difference, sweeps, wall time. This is a presentation of
    ``results_table``, not a second computation: every number in it is already
    in the full frame, which is what a reader should go to when a row looks
    surprising.
    """
    columns = {
        "case": "Case",
        "rel_flux_diff": "Relative flux difference",
        "sweeps": "Sweeps",
        "wall_time": "Wall time (s)",
    }
    present = [name for name in columns if name in frame.columns]
    view = frame[present].rename(columns=columns)
    return view.reset_index(drop=True)


def format_table(frame: pd.DataFrame, float_format: str = "%.3e") -> str:
    """The table as text, for a terminal."""
    with pd.option_context("display.max_columns", None, "display.width", 200):
        return frame.to_string(index=False, float_format=lambda v: float_format % v)
