"""Shared pieces for the two narration stages.

Prompts live in files, not string literals, so that changing what the model is
asked shows up in review as a diff to a document a domain expert can read
without opening a Python module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"


class NarrationError(Exception):
    """A narration stage failed, or was asked for something it cannot do."""


def load_prompt(name: str) -> str:
    """Read one prompt from ``prompts/``."""
    path = PROMPT_DIR / f"{name}.md"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in PROMPT_DIR.glob("*.md")))
        raise NarrationError(f"no prompt named {name!r}. Available: {available or 'none'}")
    return path.read_text()


@dataclass
class Citation:
    """One piece of retrieved evidence, in the shape the reviewer expects.

    The sister repository's ``SMEReviewer`` reads ``number``, ``pid``, ``text``,
    ``location``, and ``revision_known`` off the citations it is given. Rather
    than import its Citation class, which is entangled with its answer model,
    this carries the same five fields. It is a shim, and a small one, and it is
    the price of not making the reviewer depend on this package.
    """

    number: int
    pid: Optional[str]
    text: str
    location: str
    revision_known: bool = True
    url: Optional[str] = None
    title: Optional[str] = None

    def as_evidence(self, limit: int = 1600) -> str:
        heading = f"[{self.number}] {self.location}"
        if self.title:
            heading += f" — {self.title}"
        body = self.text[:limit]
        if not self.revision_known:
            heading += "  (revision unknown; may predate the pinned commit)"
        return f"{heading}\n{body}"


def citations_from_hits(hits: Sequence[Any]) -> list[Citation]:
    """Turn knowledge-pack hits into citations, numbered from one."""
    citations: list[Citation] = []
    for number, hit in enumerate(hits, start=1):
        citations.append(
            Citation(
                number=number,
                pid=getattr(hit, "pid", None),
                text=getattr(hit, "text", "") or "",
                location=hit.citation() if hasattr(hit, "citation") else str(hit),
                revision_known=bool(getattr(hit, "commit", None)),
                url=getattr(hit, "url", None),
                title=getattr(hit, "title", None),
            )
        )
    return citations


def evidence_block(citations: Sequence[Citation], limit: int = 1600) -> str:
    return "\n\n".join(citation.as_evidence(limit) for citation in citations)
