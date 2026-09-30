# The language-model layer

Two calls, at the two ends of the pipeline. Everything between them is
deterministic. This document says how they are wired, what stands between a
wrong answer and a running simulation, and what remains weak.

## Where the client comes from

The provider is the sister repository's `LLMClient`, reused rather than
rewritten. It already handles the TAMU AI Chat gateway, the Groq fallback, and
the fact that the TAMU client silently discards `temperature` and `max_tokens`.
Keys come from the sister repository's `.env`.

`autoopensn/llm.py` wraps it and adds the four things a workflow tool needs and
a chat client does not:

**A timeout.** The TAMU client exposes none, so a call runs on a worker thread
and is abandoned after 120 seconds, the same device the sister repository's
reviewer uses. The request keeps running until the provider closes it. The cost
of giving up is one wasted completion; the cost of not giving up is a study that
never finishes.

**JSON and YAML that survive a chatty model.** Models fence their output,
apologise before it and explain after it. `extract_block` finds the document.

**Model choice.** Two models, following the sister repository's measured split.
Spec generation uses the fast model, because structured output picked from a
menu we supply is a lookup. Explanation uses the thorough model, because that is
where the extra development shows. The ids are read from the sister
repository's config rather than duplicated here.

| Stage | Model | Typical latency |
| --- | --- | --- |
| Request to spec | `protected.gpt-5.4-mini` | 3 to 5 seconds |
| Explain results | `protected.gpt-5.5` | 10 to 15 seconds |

**An offline mode.** `ScriptedLLM` returns canned text in order. Every test in
this repository uses it, so the suite never reaches a provider and a live call
is opt-in. A stage whose only exercise is a live call is a stage nobody
refactors, and a suite that costs money per run is one people stop running.

## Stage one: request to spec

Four things stand between a wrong answer and a running simulation, in order.

**1. The menu is closed.** The model picks a scenario from a shortlist of real
OpenSn tutorials and sets parameters that scenario declares. It cannot name a
parameter that does not exist, because the only names it is given are the ones
read out of the tutorial's own script.

The shortlist is chosen deterministically, by a lexical rank over titles,
summaries and, most usefully, the parameters each scenario declares. A request
that says "restart interval" matches the scenarios whose scripts contain
`gmres_restart_interval`. That is the most specific signal available and it does
not cost a model call.

**2. The ranges are checked in code.** `Spec.validate_against` compares every
value against the sidecar. The model's opinion about what is legal is not
consulted. This is the check that cannot be talked out of.

**3. One repair round.** A rejected spec goes back with the exact error and the
full parameter declarations for whichever scenario was actually chosen. One
round, not a loop. A model that cannot fix a named range violation with the
range in front of it will not fix it on the fourth attempt, and an unbounded
repair loop is how a tool burns an hour producing nothing.

**4. A domain reviewer, advisory only.** The sister repository's `SMEReviewer`
reads the spec against the retrieved evidence, and its objections are routed the
way that module routes them:

- `supported_by_evidence = False` means the spec says something the evidence
  does not. A correct spec exists inside the evidence already given, so it is
  worth exactly one regeneration.
- `supported_by_evidence = True` means the reviewer disputes the evidence
  itself. No rewrite can satisfy that, so it becomes a note and is never fed
  back.

**The reviewer cannot overturn validation.** A revision the reviewer asked for
that violates a declared range is discarded and the validated spec stands. This
is the same lesson the sister repository records about its own advisory layer,
and `tests/test_narrate.py` asserts it.

## Stage two: results to narrative

The model receives the table and the spec. Not the logs, not the run
directories, not raw solver output. If a quantity is not a column, it is not
available to the narration, and the fix is to parse it into a column rather than
hand the model more text to interpret.

Afterwards, `unsupported_numbers` checks every figure in the prose against the
table. It is deliberately forgiving: within two percent counts as supported, and
small integers are ignored because prose counts cases and ranks results. It
exists to catch an invented figure, not to police rounding. A flagged figure is
shown to the reader as a warning rather than silently removed.

The prompt also requires the narrative to say when numbers were replayed from a
recording rather than measured. That rule is only satisfiable because the
`replayed` column is passed through; a rule the model cannot satisfy because we
withheld the evidence would read as compliance while producing a narrative that
presents recorded numbers as fresh ones.

## Both prompts are files

`autoopensn/narrate/prompts/` holds them as Markdown. Changing what the model is
asked shows up in review as a diff to a document a domain expert can read
without opening a Python module.

## What is still weak

**Scenario choice is the least guarded step.** Validation catches a bad
parameter; nothing catches a defensible choice of the wrong tutorial. A request
about 2D mesh refinement answered with a 1D scenario produces a perfectly valid
spec for the wrong problem. The interface mitigates this by showing which
scenarios were considered and letting you force one, and that is mitigation
rather than a fix.

**The model can still drop a requested value.** An early run of the acceptance
prompt produced six sweep points instead of seven, having quietly omitted one
restart interval. The prompt now names this explicitly and the run that follows
produces all seven, but nothing checks it mechanically. A deterministic check
that every literal value in the request appears in the spec would close it.

**Sweep size is unbounded by anything but the prompt.** A request that implies a
large grid produces one. The prompt asks for under ten runs; nothing enforces
it. A cap in `Spec`, with the interface asking for confirmation above it, would
be the obvious guard.
