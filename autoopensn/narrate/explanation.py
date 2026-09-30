"""Explaining a finished results table.

The easier of the two model stages, and the one with the sharper rule: the model
receives the table and nothing else. Not the logs, not the run directories, not
the raw solver output. If a quantity is not a column, it is not available to the
narration, and the fix is to parse it into a column rather than to hand the model
more text to interpret.

That rule is what makes the narrative checkable. Every number in it should be
findable in the table by eye, and ``unsupported_numbers`` does that check
mechanically afterwards, because a model asked not to invent numbers still
occasionally will.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import pandas as pd

from autoopensn.llm import LLM, LLMError
from autoopensn.narrate.base import NarrationError, load_prompt
from autoopensn.spec.model import Spec

# Columns that describe bookkeeping rather than results. Passing them to the
# model invites a narrative about run directories.
#
# ``replayed`` is deliberately NOT here. The prompt requires the narrative to
# say when a number was replayed from a recording rather than measured, and a
# rule the model cannot satisfy because we withheld the evidence is worse than
# no rule: it reads as compliance while producing a narrative that presents
# recorded numbers as fresh ones.
DROP_COLUMNS = ("case_id", "is_reference", "gold_case")

MAX_ROWS = 60


@dataclass
class Explanation:
    """A narrative, and what it was written from."""

    text: str
    table: Optional[pd.DataFrame] = None
    model: Optional[str] = None
    elapsed: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.text


def table_for_prompt(table: pd.DataFrame, max_rows: int = MAX_ROWS) -> str:
    """The table as the model sees it: every result column, no bookkeeping.

    All-empty columns are dropped. A column of blanks tells the model nothing
    and invites it to explain the absence as a finding.
    """
    frame = table.copy()
    frame = frame.drop(columns=[c for c in DROP_COLUMNS if c in frame.columns])
    frame = frame.dropna(axis=1, how="all")
    if len(frame) > max_rows:
        frame = frame.head(max_rows)
    return frame.to_string(index=False)


def _context(spec: Optional[Spec]) -> str:
    if spec is None:
        return ""
    lines = [f"STUDY: {spec.name}"]
    if spec.description:
        lines.append(f"PURPOSE: {spec.description.strip()}")
    lines.append(f"SCENARIO: {spec.template}")
    for dimension in spec.sweep:
        values = ", ".join(str(v) for v in dimension.values)
        lines.append(f"VARIED: {dimension.parameter} over {values}")
    if spec.reference:
        overrides = ", ".join(f"{k}={v}" for k, v in spec.reference.items())
        lines.append(f"REFERENCE CASE: {overrides}")
    return "\n".join(lines)


NUMBER = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


def unsupported_numbers(text: str, table: pd.DataFrame, tolerance: float = 0.02) -> list[str]:
    """Numbers in the narrative that do not appear in the table.

    A cheap, deliberately forgiving check. It exists to catch a narrative that
    invents a figure, not to police rounding, so a number within a couple of
    percent of a table value counts as supported, and small integers are
    ignored because they are usually counts of cases or ordinals in prose.
    """
    values: list[float] = []
    for column in table.columns:
        series = pd.to_numeric(table[column], errors="coerce").dropna()
        values.extend(float(v) for v in series)

    unsupported: list[str] = []
    for match in NUMBER.finditer(text):
        token = match.group(0)
        try:
            number = float(token)
        except ValueError:
            continue
        if number == int(number) and abs(number) <= 100:
            continue
        if any(
            abs(number - value) <= tolerance * max(abs(value), 1e-30) for value in values
        ):
            continue
        unsupported.append(token)
    return unsupported


def explain(
    table: pd.DataFrame,
    spec: Optional[Spec] = None,
    *,
    llm: Optional[LLM] = None,
    question: Optional[str] = None,
    check_numbers: bool = True,
) -> Explanation:
    """Explain the trends in a finished results table."""
    if table is None or len(table) == 0:
        raise NarrationError("there are no results to explain")

    if llm is None:
        from autoopensn.llm import default_llm  # noqa: PLC0415

        llm = default_llm()

    system = load_prompt("explain")
    parts = [_context(spec), "RESULTS TABLE:\n" + table_for_prompt(table)]
    if question:
        parts.append(f"The reader asked specifically: {question}")
    parts.append("Explain what these results show.")
    user = "\n\n".join(part for part in parts if part)

    chooser = getattr(llm, "model_for", None)
    model = chooser("thorough") if callable(chooser) else None

    try:
        completion = llm.complete(system, user, temperature=0.3, max_tokens=1600, model=model)
    except LLMError as exc:
        raise NarrationError(str(exc)) from exc

    explanation = Explanation(
        text=completion.text.strip(),
        table=table,
        model=completion.model,
        elapsed=completion.elapsed,
    )

    if check_numbers:
        invented = unsupported_numbers(explanation.text, table)
        if invented:
            explanation.warnings.append(
                "These figures do not appear in the results table and may have "
                "been invented: " + ", ".join(sorted(set(invented))[:8])
            )
    return explanation
