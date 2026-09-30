"""Turning a plain-language request into a validated scenario spec.

This is one of the two places a language model is allowed, and the harder of the
two. The model is choosing what will actually be computed, so the question is
not whether it can produce a plausible spec but what happens when it produces a
wrong one.

Four things stand between a wrong answer and a running simulation, in order:

**The menu is closed.** The model picks a scenario from a shortlist of real
OpenSn tutorials and sets parameters that scenario declares. It cannot name a
parameter that does not exist, because the only names it is given are the ones
read out of the tutorial's own script.

**The ranges are checked in code.** ``Spec.validate_against`` compares every
value against the range in the sidecar. The model's opinion about what is legal
is not consulted. This is the check that cannot be talked out of.

**One repair round.** A rejected spec goes back with the exact error and the
full parameter declarations. One round, not a loop: a model that cannot fix a
named range violation with the range in front of it will not fix it on the
fourth attempt either, and an unbounded repair loop is how a tool burns an hour
producing nothing.

**A domain reviewer, advisory only.** The sister repository's ``SMEReviewer``
reads the spec against the retrieved evidence. Its objections are routed the way
that module routes them: one that the evidence can fix is worth a regeneration,
one that disputes the evidence itself becomes a note and is never fed back. The
reviewer cannot overturn validation. That ordering matters and is the same
lesson the sister repository records: an advisory layer must not be able to
overrule a decisive one.

The stage returns a draft rather than a bare spec, because the user is supposed
to read it before anything runs, and reading it is easier with the candidates,
the evidence, and the reviewer's notes attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

from autoopensn import PINNED_OPENSN_COMMIT
from autoopensn.llm import LLM, LLMError, extract_block
from autoopensn.narrate.base import (
    Citation,
    NarrationError,
    citations_from_hits,
    evidence_block,
    load_prompt,
)
from autoopensn.spec.model import Spec, SpecError
from autoopensn.templates import Template, TemplateError, list_templates, load_template
from autoopensn.tutorials import catalog
from autoopensn.tutorials.catalog import Scenario

SHORTLIST = 5
MAX_DECLARATIONS = 40
"""Parameters shown in full for the leading candidate.

A tutorial can have a hundred knobs. Showing all of them for five candidates
would produce a prompt nobody reads, model included. The leading candidate gets
its solver knobs plus as many others as fit; the rest are named but not
detailed, and the repair round can show the full set for whichever scenario was
actually chosen.
"""


@dataclass
class SpecDraft:
    """A generated spec and everything needed to judge it.

    Deliberately not a bare ``Spec``. The point of splitting spec generation
    from running is that a person reads the spec first, and a reader needs to
    know which scenarios were considered, what evidence was in front of the
    model, and whether the domain reviewer objected.
    """

    spec: Optional[Spec]
    prompt: str
    scenario: Optional[Scenario] = None
    candidates: list[Scenario] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    attempts: list[str] = field(default_factory=list)
    """Raw model output for each attempt, oldest first."""
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """Source-level reviewer findings: things the evidence says that the
    reviewer believes are wrong. Shown, never fed back."""
    revisions: int = 0
    review_status: str = "off"
    model: Optional[str] = None
    elapsed: float = 0.0

    @property
    def ok(self) -> bool:
        return self.spec is not None

    def summary(self) -> str:
        if not self.ok:
            return f"no valid spec after {len(self.attempts)} attempt(s)"
        points = len(self.spec.expand())
        parts = [f"{self.spec.template}", f"{points} run{'s' if points != 1 else ''}"]
        if self.revisions:
            parts.append(f"{self.revisions} revision")
        if self.notes:
            parts.append(f"{len(self.notes)} reviewer note(s)")
        return ", ".join(parts)


# --- prompt assembly --------------------------------------------------------


def declaration_lines(template: Template, limit: int = MAX_DECLARATIONS) -> str:
    """Every parameter of a template, as the model needs to see it."""
    names = list(template.solver_parameters)
    names += [n for n in template.non_solver_parameters if n not in names]

    lines: list[str] = []
    for name in names[:limit]:
        declaration = template.parameters[name]
        bits = [f"  {name} ({declaration.group}, {declaration.type})"]
        bits.append(f"default {declaration.default!r}")
        if declaration.choices:
            bits.append("one of " + ", ".join(map(repr, declaration.choices)))
        else:
            bounds = []
            if declaration.minimum is not None:
                bounds.append(f">= {declaration.minimum:g}")
            if declaration.maximum is not None:
                bounds.append(f"<= {declaration.maximum:g}")
            if bounds:
                bits.append(" and ".join(bounds))
        if declaration.provenance == "inferred":
            bits.append("range inferred, a guard not a code limit")
        lines.append("; ".join(bits))

    remaining = len(names) - limit
    if remaining > 0:
        lines.append(f"  ... and {remaining} more parameter(s) not listed here")
    return "\n".join(lines)


def scenario_block(
    scenarios: Sequence[Scenario], leader: Optional[Template] = None
) -> str:
    """The shortlist, with full declarations for the leading candidate."""
    blocks = []
    for index, scenario in enumerate(scenarios):
        entry = scenario.catalog_entry()
        if index == 0 and leader is not None:
            entry += "\n  observables: " + ", ".join(leader.observables)
            entry += "\n  parameters:\n" + declaration_lines(leader)
        blocks.append(entry)
    return "\n\n".join(blocks)


def build_request(
    prompt: str,
    scenarios: Sequence[Scenario],
    leader: Optional[Template],
    citations: Sequence[Citation],
) -> str:
    parts = [f"REQUEST:\n{prompt}\n"]
    parts.append("SCENARIOS AVAILABLE (choose exactly one):\n" + scenario_block(scenarios, leader))
    if citations:
        parts.append(
            "EVIDENCE from the OpenSn documentation and source at the pinned commit:\n"
            + evidence_block(citations, limit=1200)
        )
    parts.append("Return the spec as YAML. Nothing else.")
    return "\n\n".join(parts)


# --- parsing ----------------------------------------------------------------


def parse_spec(raw: str) -> tuple[Optional[dict], Optional[str]]:
    """The YAML document out of a model response, or why it could not be read."""
    text = extract_block(raw)
    if not text.strip():
        return None, "the model returned nothing"
    try:
        document = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return None, f"the model's output is not valid YAML: {exc}"
    if not isinstance(document, dict):
        return None, "expected a YAML mapping of spec fields"
    return document, None


def build_spec(document: dict, pack_commit: str) -> Spec:
    """A validated ``Spec`` from a parsed document, or ``SpecError``."""
    document = dict(document)
    document.setdefault("pack_commit", pack_commit)
    # A model that writes a commit is guessing; the pinned one is a fact.
    document["pack_commit"] = pack_commit
    try:
        spec = Spec(**document)
    except Exception as exc:
        raise SpecError(str(exc)) from exc
    spec.validate_against()
    return spec


# --- the stage --------------------------------------------------------------


def request_to_spec(
    prompt: str,
    *,
    llm: Optional[LLM] = None,
    scenarios: Optional[Sequence[Scenario]] = None,
    template: Optional[str] = None,
    pack_commit: str = PINNED_OPENSN_COMMIT,
    shortlist: int = SHORTLIST,
    ground: bool = True,
    review: bool = True,
    reviewer: Any = None,
    max_repairs: int = 1,
) -> SpecDraft:
    """Turn a request into a validated spec, or explain why it could not be.

    Never raises for a model that produced something unusable. The draft comes
    back with ``ok`` False, the raw attempts, and the validation errors, because
    the useful response to a bad spec is to show the user what was tried.
    """
    if llm is None:
        from autoopensn.llm import default_llm  # noqa: PLC0415

        llm = default_llm()

    available = list(scenarios) if scenarios is not None else catalog.load()
    if not available:
        raise NarrationError(
            "the scenario catalog is empty. Build it with `autoopensn scenarios "
            "--rebuild`, which reads the tutorials from the pinned OpenSn checkout."
        )

    draft = SpecDraft(spec=None, prompt=prompt)

    # 1. Narrow the menu. Deterministic: a lexical rank over titles, summaries,
    #    and the parameters each scenario's own script declares.
    if template:
        chosen = catalog.by_name(template, available)
        if chosen is None:
            raise NarrationError(
                f"no scenario named {template!r}. Available: "
                + ", ".join(sorted(s.name for s in available)[:12])
                + " …"
            )
        draft.candidates = [chosen]
    else:
        draft.candidates = catalog.rank(prompt, available, limit=shortlist)

    leader = _load(draft.candidates[0].name) if draft.candidates else None

    # 2. Ground the request in the pinned documentation.
    if ground:
        draft.citations = _ground(prompt)

    system = load_prompt("request_to_spec")
    user = build_request(prompt, draft.candidates, leader, draft.citations)

    # 3. Generate, then repair once against the exact validation error.
    spec, model_name, elapsed = None, None, 0.0
    for attempt in range(max_repairs + 1):
        try:
            completion = llm.complete(
                system, user, temperature=0.1, model=_model_for(llm, "fast")
            )
        except LLMError as exc:
            draft.errors.append(str(exc))
            return draft

        draft.attempts.append(completion.text)
        model_name, elapsed = completion.model, elapsed + completion.elapsed

        document, problem = parse_spec(completion.text)
        if document is None:
            draft.errors.append(problem or "unreadable output")
        else:
            try:
                spec = build_spec(document, pack_commit)
                break
            except (SpecError, TemplateError) as exc:
                draft.errors.append(str(exc))
                document_template = str(document.get("template", "")) or None
                user = _repair_request(user, completion.text, str(exc), document_template)

        if attempt == max_repairs:
            break

    draft.model, draft.elapsed = model_name, elapsed
    if spec is None:
        return draft

    draft.spec = spec
    draft.scenario = catalog.by_name(spec.template, available)

    # 4. Domain review. Advisory, and never able to overturn validation.
    if review and draft.citations:
        _review(draft, llm, reviewer, system, pack_commit)

    return draft


def _repair_request(
    original: str, produced: str, error: str, template_name: Optional[str]
) -> str:
    """The follow-up that gets exactly one chance to fix a rejected spec.

    It carries the full parameter declarations for whichever scenario the model
    actually chose, which is usually the missing information: the first prompt
    details the leading candidate, and a model that picked a different one was
    working from names alone.
    """
    detail = ""
    if template_name:
        try:
            template = _load(template_name)
            detail = (
                f"\n\nEvery parameter {template_name} declares:\n"
                + declaration_lines(template, limit=200)
                + "\n\nObservables it can produce: "
                + ", ".join(template.observables)
            )
        except TemplateError:
            detail = (
                f"\n\nThere is no scenario named {template_name!r}. Choose one from "
                f"the shortlist above, copying its name exactly."
            )

    return (
        original
        + "\n\n---\n\nYour previous answer was rejected.\n\nYOU WROTE:\n"
        + produced.strip()[:2000]
        + f"\n\nIT WAS REJECTED BECAUSE:\n{error}"
        + detail
        + "\n\nReturn a corrected spec as YAML. Change only what the rejection "
        "names; do not rewrite the rest."
    )


def _load(name: str) -> Optional[Template]:
    try:
        return load_template(name)
    except TemplateError:
        return None


def _model_for(llm: LLM, task: str) -> Optional[str]:
    chooser = getattr(llm, "model_for", None)
    return chooser(task) if callable(chooser) else None


def _ground(prompt: str, k: int = 5) -> list[Citation]:
    """Retrieved evidence for the request, or none if the pack is unavailable."""
    try:
        from autoopensn import kp_bridge  # noqa: PLC0415

        if not kp_bridge.available():
            return []
        return citations_from_hits(kp_bridge.search(prompt, k=k))
    except Exception:
        # Grounding improves the spec; it is not load-bearing. Validation is.
        return []


def _review(
    draft: SpecDraft,
    llm: LLM,
    reviewer: Any,
    system: str,
    pack_commit: str,
) -> None:
    """Run the sister repository's domain reviewer over the generated spec.

    Failure here is never fatal. A reviewer that times out or errors leaves the
    spec exactly as it was: an unavailable second opinion must not cost the user
    their study.
    """
    try:
        if reviewer is None:
            from autoopensn import kp_bridge  # noqa: PLC0415

            raw_client = getattr(llm, "client", None)
            if raw_client is None:
                draft.review_status = "off"
                return
            reviewer = kp_bridge.reviewer(raw_client)
        if reviewer is None:
            draft.review_status = "off"
            return
    except Exception as exc:
        draft.review_status = "unavailable"
        draft.errors.append(f"reviewer unavailable: {exc}")
        return

    spec_text = draft.spec.to_yaml()
    context = evidence_block(draft.citations, limit=1600)

    try:
        if not reviewer.should_review(spec_text, draft.citations):
            draft.review_status = "skipped"
            return
        first = reviewer.review(draft.prompt, spec_text, context, citations=draft.citations)
    except Exception as exc:
        draft.review_status = "unavailable"
        draft.errors.append(f"review failed: {exc}")
        return

    if not first.ok:
        draft.review_status = "unavailable"
        return

    # Source-level: the evidence says it and the reviewer thinks the evidence is
    # wrong. No rewrite can satisfy that, so it is shown and never fed back.
    draft.notes = [finding.explanation for finding in first.source_level]

    fixable = first.answer_level
    if not fixable:
        draft.review_status = "flagged" if draft.notes else "clean"
        return

    objections = "\n".join(f"- {finding.explanation}" for finding in fixable)
    repair = (
        f"REQUEST:\n{draft.prompt}\n\nYOU PRODUCED:\n{spec_text}\n\n"
        f"A subject-matter reviewer raised these problems:\n{objections}\n\n"
        "Return a corrected spec as YAML, using only the parameters and ranges "
        "you were given. Keep everything that was right."
    )

    try:
        completion = llm.complete(system, repair, temperature=0.1, model=_model_for(llm, "fast"))
    except LLMError:
        draft.review_status = "flagged"
        return

    draft.attempts.append(completion.text)
    document, problem = parse_spec(completion.text)
    if document is None:
        draft.review_status = "flagged"
        return
    try:
        revised = build_spec(document, pack_commit)
    except (SpecError, TemplateError) as exc:
        # The reviewer's suggestion produced an invalid spec. The validated one
        # stands. This is the ordering that matters: advisory never overturns
        # decisive.
        draft.errors.append(f"revision after review was rejected, keeping the original: {exc}")
        draft.review_status = "flagged"
        return

    draft.spec = revised
    draft.revisions = 1
    draft.review_status = "revised"
