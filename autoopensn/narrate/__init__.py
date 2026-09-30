"""The two LLM calls, and only two.

``request_to_spec(prompt) -> SpecDraft``
    Turns a request into a validated scenario spec, choosing from the OpenSn
    tutorials, grounded in the knowledge pack for the pinned commit, range-
    checked in code, and reviewed by the sister repository's domain reviewer.

``explain(table, spec) -> Explanation``
    Reads the finished results table and says what it shows.

Nothing else in this package calls a language model. In particular, no model
writes a simulation script, decides whether a run succeeded, or reads a raw log.
Those are the three places where a fluent wrong answer is indistinguishable from
a right one, and all three are handled by code that can be tested.

``request_to_spec`` returns a draft rather than a bare spec. The user is meant to
read the spec before anything runs, and reading it is easier with the scenarios
that were considered, the evidence the model saw, and the reviewer's notes
attached to it.
"""

from autoopensn.narrate.base import (
    Citation,
    NarrationError,
    PROMPT_DIR,
    citations_from_hits,
    evidence_block,
    load_prompt,
)
from autoopensn.narrate.explanation import Explanation, explain, unsupported_numbers
from autoopensn.narrate.spec_generation import (
    SpecDraft,
    build_spec,
    declaration_lines,
    parse_spec,
    request_to_spec,
)

__all__ = [
    "Citation",
    "Explanation",
    "NarrationError",
    "PROMPT_DIR",
    "SpecDraft",
    "build_spec",
    "citations_from_hits",
    "declaration_lines",
    "evidence_block",
    "explain",
    "load_prompt",
    "parse_spec",
    "request_to_spec",
    "unsupported_numbers",
]
