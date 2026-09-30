"""Reading an OpenSn tutorial notebook.

A tutorial is a Jupyter notebook alternating prose and code. OpenSn's own test
harness converts one to a script with ``nbconvert`` and runs it, and records
gold values for the result in a ``tests.json`` beside it. So a tutorial is
already a runnable, checked scenario upstream; this module is what lets us treat
it as one.

Two products come out of a notebook, and they are used for different things:

``script(path)``
    The code cells, concatenated. This is what runs, and it is what a generated
    template must reproduce exactly at default parameters.

``prose(path)``
    The markdown cells. This is what a language model reads when deciding which
    scenario answers a question, and what the generated sidecar quotes when
    describing a parameter. The prose is the part of a tutorial that says what
    the numbers mean, and throwing it away would leave the model guessing from
    variable names.

We concatenate rather than shelling out to ``nbconvert``. The two differ only in
the banner comments nbconvert inserts; the code is identical, and a pure
function with no optional dependency is worth more here than byte-compatibility
with a tool we do not run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

CELL_SEPARATOR = "\n\n"


class NotebookError(Exception):
    """A notebook could not be read or contains nothing runnable."""


def _cells(path: Path) -> list[dict[str, Any]]:
    try:
        document = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise NotebookError(f"{path}: not valid JSON: {exc}") from exc
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise NotebookError(f"{path}: no cells")
    return cells


def _source(cell: dict[str, Any]) -> str:
    source = cell.get("source", "")
    if isinstance(source, list):
        return "".join(source)
    return str(source)


def script(path: Path) -> str:
    """The notebook's code cells as one runnable script.

    Empty cells are dropped. Trailing whitespace on each cell is stripped so
    that the separator between cells is exactly one blank line, which makes the
    result stable against a notebook whose cells happen to end with newlines.
    """
    blocks = []
    for cell in _cells(path):
        if cell.get("cell_type") != "code":
            continue
        text = _source(cell).rstrip()
        if text.strip():
            blocks.append(text)
    if not blocks:
        raise NotebookError(f"{path}: no code cells")
    return CELL_SEPARATOR.join(blocks) + "\n"


def prose(path: Path) -> str:
    """The notebook's markdown cells, joined."""
    blocks = [
        _source(cell).rstrip()
        for cell in _cells(path)
        if cell.get("cell_type") == "markdown" and _source(cell).strip()
    ]
    return "\n\n".join(blocks)


def title(path: Path) -> str:
    """The notebook's first heading, or its filename.

    The first markdown heading is the tutorial's own name for itself, which is
    a far better label in a catalog than a slug derived from a filename.
    """
    for cell in _cells(path):
        if cell.get("cell_type") != "markdown":
            continue
        for line in _source(cell).splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
    return Path(path).stem.replace("_", " ")


def summary(path: Path, limit: int = 600) -> str:
    """The opening prose, up to ``limit`` characters.

    Headings and code fences are dropped: what is wanted is the paragraph a
    person would read to decide whether this tutorial is about their problem.
    """
    paragraphs: list[str] = []
    for cell in _cells(path):
        if cell.get("cell_type") != "markdown":
            continue
        for block in _source(cell).split("\n\n"):
            stripped = block.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("```"):
                continue
            paragraphs.append(" ".join(stripped.split()))
        if sum(len(p) for p in paragraphs) >= limit:
            break
    joined = " ".join(paragraphs)
    if len(joined) <= limit:
        return joined
    return joined[:limit].rsplit(" ", 1)[0] + "…"


@dataclass(frozen=True)
class Notebook:
    """One tutorial notebook, read once."""

    path: Path
    title: str
    summary: str
    prose: str
    script: str

    @classmethod
    def read(cls, path: Path) -> "Notebook":
        path = Path(path)
        return cls(
            path=path,
            title=title(path),
            summary=summary(path),
            prose=prose(path),
            script=script(path),
        )
