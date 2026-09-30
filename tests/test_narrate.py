"""Tests for the two language-model stages.

Every test here uses ``ScriptedLLM``, so the suite never reaches a provider. A
stage whose only exercise is a live call is a stage nobody refactors, and a test
suite that costs money per run is one people stop running.

What is actually being tested is not the model. It is the machinery around the
model: that a bad spec is rejected rather than run, that the rejection comes back
with the error attached, that one repair round happens and not a loop, and that
the reviewer can never overturn validation.
"""

from __future__ import annotations

import pandas as pd
import pytest

from autoopensn.llm import LLMError, ScriptedLLM, extract_block, extract_json
from autoopensn.narrate import (
    NarrationError,
    PROMPT_DIR,
    declaration_lines,
    explain,
    load_prompt,
    parse_spec,
    request_to_spec,
    unsupported_numbers,
)
from autoopensn.narrate.explanation import table_for_prompt

GOOD_SPEC = """
name: tolerance_study
template: reed_1d
description: Effect of the inner tolerance.
template_parameters:
  emit_avg_flux: true
solver:
  l_abs_tol: 1.0e-6
sweep:
  - parameter: l_abs_tol
    values: [1.0e-4, 1.0e-6, 1.0e-8]
reference:
  l_abs_tol: 1.0e-10
observables: [avg_flux, sweeps, wall_time]
"""

OUT_OF_RANGE = """
name: bad_study
template: reed_1d
solver:
  gmres_restart_interval: 0
observables: [sweeps]
"""

UNKNOWN_PARAMETER = """
name: bad_study
template: reed_1d
solver:
  magic_accelerator: 3
observables: [sweeps]
"""


@pytest.fixture
def one_scenario(reed_template):
    """A catalog of exactly one scenario, so no real catalog is needed."""
    from autoopensn.tutorials.catalog import Scenario

    return [
        Scenario(
            name="reed_1d",
            title="Reed 1D problem",
            summary="A five-region slab benchmark.",
            notebook="foundations/reed.ipynb",
            section="foundations",
            parameters=list(reed_template.parameters),
            solver_parameters=list(reed_template.solver_parameters),
        )
    ]


# --- prompts ----------------------------------------------------------------


def test_both_prompts_exist_as_files():
    assert (PROMPT_DIR / "request_to_spec.md").exists()
    assert (PROMPT_DIR / "explain.md").exists()


def test_an_unknown_prompt_lists_what_exists():
    with pytest.raises(NarrationError) as excinfo:
        load_prompt("no_such_prompt")
    assert "explain" in str(excinfo.value)


def test_the_spec_prompt_forbids_inventing_parameters():
    text = load_prompt("request_to_spec")
    assert "declared" in text
    assert "range" in text.lower()


def test_the_explain_prompt_forbids_numbers_not_in_the_table():
    assert "not in the table" in load_prompt("explain")


# --- response handling ------------------------------------------------------


def test_fenced_output_is_unwrapped():
    assert extract_block("here you go:\n```yaml\nname: x\n```\nhope that helps") == "name: x"


def test_json_survives_surrounding_prose():
    assert extract_json('Sure! {"a": 1} Let me know.') == {"a": 1}


def test_unparseable_json_is_none_not_an_exception():
    assert extract_json("no object here") is None


def test_parse_spec_reports_bad_yaml():
    document, problem = parse_spec("name: [unclosed")
    assert document is None
    assert "YAML" in problem


def test_parse_spec_rejects_a_bare_list():
    document, problem = parse_spec("- one\n- two\n")
    assert document is None
    assert "mapping" in problem


# --- spec generation --------------------------------------------------------


def test_a_good_response_becomes_a_validated_spec(one_scenario):
    llm = ScriptedLLM([GOOD_SPEC])
    draft = request_to_spec("vary the tolerance", llm=llm, scenarios=one_scenario,
                            ground=False, review=False)
    assert draft.ok
    assert draft.spec.template == "reed_1d"
    assert len(draft.spec.expand()) == 4


def test_the_pinned_commit_is_imposed_not_taken_from_the_model(one_scenario):
    llm = ScriptedLLM([GOOD_SPEC + "\npack_commit: deadbeef\n"])
    draft = request_to_spec("vary the tolerance", llm=llm, scenarios=one_scenario,
                            ground=False, review=False)
    assert draft.ok
    assert draft.spec.pack_commit != "deadbeef"


def test_an_out_of_range_value_is_rejected_and_repaired_once(one_scenario):
    """The model gets the exact error back, and one chance to use it."""
    llm = ScriptedLLM([OUT_OF_RANGE, GOOD_SPEC])
    draft = request_to_spec("vary the restart", llm=llm, scenarios=one_scenario,
                            ground=False, review=False)
    assert draft.ok
    assert len(draft.attempts) == 2
    assert "minimum" in draft.errors[0]
    # The repair prompt must actually contain the rejection.
    assert "REJECTED" in llm.calls[1]["user"]
    assert "minimum" in llm.calls[1]["user"]


def test_repair_happens_once_not_in_a_loop(one_scenario):
    llm = ScriptedLLM([OUT_OF_RANGE, OUT_OF_RANGE, GOOD_SPEC])
    draft = request_to_spec("vary the restart", llm=llm, scenarios=one_scenario,
                            ground=False, review=False)
    assert not draft.ok
    assert len(draft.attempts) == 2
    assert len(llm.responses) == 1, "a third call would be a loop"


def test_an_unknown_parameter_is_rejected(one_scenario):
    llm = ScriptedLLM([UNKNOWN_PARAMETER, UNKNOWN_PARAMETER])
    draft = request_to_spec("do something", llm=llm, scenarios=one_scenario,
                            ground=False, review=False)
    assert not draft.ok
    assert "magic_accelerator" in draft.errors[0]


def test_garbage_output_is_reported_not_raised(one_scenario):
    llm = ScriptedLLM(["I'd be happy to help!", "Still thinking about it."])
    draft = request_to_spec("do something", llm=llm, scenarios=one_scenario,
                            ground=False, review=False)
    assert not draft.ok
    assert draft.errors


def test_a_provider_failure_is_reported_not_raised(one_scenario):
    class Broken(ScriptedLLM):
        def complete(self, *args, **kwargs):
            raise LLMError("the gateway is down")

    draft = request_to_spec("do something", llm=Broken([]), scenarios=one_scenario,
                            ground=False, review=False)
    assert not draft.ok
    assert "gateway is down" in draft.errors[0]


def test_an_empty_catalog_says_how_to_build_it():
    with pytest.raises(NarrationError) as excinfo:
        request_to_spec("anything", llm=ScriptedLLM([]), scenarios=[])
    assert "rebuild" in str(excinfo.value)


def test_forcing_a_scenario_skips_the_shortlist(one_scenario):
    llm = ScriptedLLM([GOOD_SPEC])
    draft = request_to_spec("anything", llm=llm, scenarios=one_scenario,
                            template="reed_1d", ground=False, review=False)
    assert [s.name for s in draft.candidates] == ["reed_1d"]


def test_forcing_an_unknown_scenario_is_an_error(one_scenario):
    with pytest.raises(NarrationError):
        request_to_spec("anything", llm=ScriptedLLM([GOOD_SPEC]), scenarios=one_scenario,
                        template="no_such_scenario")


def test_the_model_is_shown_the_declared_ranges(one_scenario, reed_template):
    llm = ScriptedLLM([GOOD_SPEC])
    request_to_spec("vary the tolerance", llm=llm, scenarios=one_scenario,
                    ground=False, review=False)
    sent = llm.calls[0]["user"]
    assert "gmres_restart_interval" in sent
    assert "petsc_gmres" in sent
    assert ">= 1" in sent


def test_declaration_lines_mark_an_inferred_range(reed_template):
    from dataclasses import replace

    inferred = replace(reed_template.parameters["n_polar"], provenance="inferred")
    template = replace(reed_template, parameters={**reed_template.parameters, "n_polar": inferred})
    assert "guard not a code limit" in declaration_lines(template, limit=200)


# --- review -----------------------------------------------------------------


class FakeFinding:
    def __init__(self, explanation, supported):
        self.explanation = explanation
        self.supported_by_evidence = supported


class FakeReviewPass:
    def __init__(self, answer_level=(), source_level=()):
        self.ok = True
        self.answer_level = list(answer_level)
        self.source_level = list(source_level)


class FakeReviewer:
    model = "fake-reviewer"

    def __init__(self, result):
        self.result = result

    def should_review(self, text, citations):
        return True

    def review(self, *args, **kwargs):
        return self.result


def _grounded_draft(llm, one_scenario, reviewer):
    from autoopensn.narrate.base import Citation

    draft = request_to_spec(
        "vary the tolerance", llm=llm, scenarios=one_scenario, ground=False, review=False
    )
    draft.citations = [Citation(1, "pid-1", "some evidence text", "docs/x.rst:1-2")]
    from autoopensn.narrate.spec_generation import _review

    _review(draft, llm, reviewer, load_prompt("request_to_spec"), draft.spec.pack_commit)
    return draft


def test_a_clean_review_leaves_the_spec_alone(one_scenario):
    llm = ScriptedLLM([GOOD_SPEC])
    draft = _grounded_draft(llm, one_scenario, FakeReviewer(FakeReviewPass()))
    assert draft.review_status == "clean"
    assert draft.revisions == 0


def test_a_source_level_finding_becomes_a_note_and_is_never_fed_back(one_scenario):
    """No rewrite can satisfy an objection to the evidence itself."""
    llm = ScriptedLLM([GOOD_SPEC])
    reviewer = FakeReviewer(
        FakeReviewPass(source_level=[FakeFinding("the docs overstate this", True)])
    )
    draft = _grounded_draft(llm, one_scenario, reviewer)
    assert draft.notes == ["the docs overstate this"]
    assert draft.revisions == 0
    assert len(llm.calls) == 1, "a source-level finding must not trigger a regeneration"


def test_an_answer_level_finding_triggers_exactly_one_revision(one_scenario):
    revised = GOOD_SPEC.replace("name: tolerance_study", "name: revised_study")
    llm = ScriptedLLM([GOOD_SPEC, revised])
    reviewer = FakeReviewer(
        FakeReviewPass(answer_level=[FakeFinding("the tolerance is too loose", False)])
    )
    draft = _grounded_draft(llm, one_scenario, reviewer)
    assert draft.review_status == "revised"
    assert draft.revisions == 1
    assert draft.spec.name == "revised_study"


def test_the_reviewer_cannot_overturn_validation(one_scenario):
    """The lesson the sister repository records: advisory never beats decisive.

    A revision that the reviewer asked for, but which violates a declared range,
    is discarded and the validated spec stands.
    """
    llm = ScriptedLLM([GOOD_SPEC, OUT_OF_RANGE])
    reviewer = FakeReviewer(
        FakeReviewPass(answer_level=[FakeFinding("use a smaller restart", False)])
    )
    draft = _grounded_draft(llm, one_scenario, reviewer)
    assert draft.spec.name == "tolerance_study", "the valid spec must survive"
    assert draft.revisions == 0
    assert any("keeping the original" in e for e in draft.errors)


def test_a_reviewer_that_fails_does_not_cost_the_spec(one_scenario):
    class Exploding(FakeReviewer):
        def review(self, *args, **kwargs):
            raise RuntimeError("reviewer timed out")

    llm = ScriptedLLM([GOOD_SPEC])
    draft = _grounded_draft(llm, one_scenario, Exploding(None))
    assert draft.ok
    assert draft.review_status == "unavailable"


# --- explanation ------------------------------------------------------------


@pytest.fixture
def table():
    return pd.DataFrame(
        {
            "case": ["a", "b"],
            "sweeps": [18, 27],
            "wall_time": [3.516, 4.074],
            "rel_flux_diff": [3.1e-05, 2.8e-07],
            "ok": [True, True],
            "sweeps_source": ["derived", "derived"],
        }
    )


def test_explain_returns_the_models_prose(table, gmres_spec):
    llm = ScriptedLLM(["A tighter tolerance costs more sweeps."])
    result = explain(table, gmres_spec, llm=llm)
    assert "tighter tolerance" in result.text
    assert not result.warnings


def test_explain_sends_the_table_and_the_purpose(table, gmres_spec):
    llm = ScriptedLLM(["fine"])
    explain(table, gmres_spec, llm=llm)
    sent = llm.calls[0]["user"]
    assert "3.516" in sent
    assert gmres_spec.description.split()[0] in sent


def test_explain_refuses_an_empty_table(gmres_spec):
    with pytest.raises(NarrationError):
        explain(pd.DataFrame(), gmres_spec, llm=ScriptedLLM(["x"]))


def test_a_number_not_in_the_table_is_flagged(table, gmres_spec):
    llm = ScriptedLLM(["The solve took 91.4 seconds and converged."])
    result = explain(table, gmres_spec, llm=llm)
    assert result.warnings
    assert "91.4" in result.warnings[0]


def test_numbers_that_are_in_the_table_are_not_flagged(table, gmres_spec):
    llm = ScriptedLLM(["Case b took 4.074 s and 27 sweeps."])
    assert not explain(table, gmres_spec, llm=llm).warnings


def test_small_integers_are_not_flagged(table, gmres_spec):
    """Prose counts cases and ranks results; those are not invented data."""
    llm = ScriptedLLM(["Both of the 2 cases converged, and the first is fastest."])
    assert not explain(table, gmres_spec, llm=llm).warnings


def test_unsupported_numbers_tolerates_rounding(table):
    assert not unsupported_numbers("about 3.52 seconds", table)


def test_bookkeeping_columns_are_not_sent(table):
    frame = table.assign(case_id=["x", "y"], is_reference=[False, True])
    sent = table_for_prompt(frame)
    assert "case_id" not in sent
    assert "is_reference" not in sent


def test_all_blank_columns_are_dropped(table):
    frame = table.assign(avg_flux=[None, None])
    assert "avg_flux" not in table_for_prompt(frame)


def test_the_replayed_column_reaches_the_model(table, gmres_spec):
    """The prompt requires saying when a number was replayed rather than measured.

    Withholding the column would make that rule unsatisfiable while still
    reading as compliance, which is worse than not having the rule.
    """
    llm = ScriptedLLM(["fine"])
    explain(table.assign(replayed=[True, True]), gmres_spec, llm=llm)
    assert "replayed" in llm.calls[0]["user"]


def test_the_spec_prompt_requires_every_requested_value():
    assert "Include every value the request names" in load_prompt("request_to_spec")
