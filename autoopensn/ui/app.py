"""AutoOpenSn, as a page.

One tab for the whole pipeline, and two for reference. Ask takes a question,
shows the spec it produced, runs it, and explains the result; editing the spec
and reading the generated script are expanders inside that flow rather than
places you have to navigate to. Scenarios and Knowledge pack answer questions
that are not studies: what can this thing do, and what does this parameter mean.

An earlier version had five tabs, two of which rendered the same panel. The
duplication showed up as widget-key collisions, which is the code pointing at a
structure that was wrong.

Three things this page is careful about, for the same reasons the rest of the
package is:

* It never shows a number that did not come out of the results table. The
  charts are drawn from the table, the table from the parser, the parser from
  recorded output.
* It says out loud when a value is replayed rather than measured, derived
  rather than logged, or guarded by an inferred range rather than a real one.
* It shows the spec before running it. The point of separating the stages is
  that a person reads what a model produced before a cluster acts on it.

Run it with ``autoopensn ui``, or ``streamlit run autoopensn/ui/app.py``.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Streamlit runs this file as a script, so the package root has to be importable
# before the first autoopensn import. This is the only place in the package that
# touches sys.path for its own sake, and it points at this repository, never at
# the sister one.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import altair as alt
import pandas as pd
import streamlit as st
import yaml

from autoopensn import PINNED_OPENSN_COMMIT, __version__, kp_bridge, llm as llm_module
from autoopensn.narrate import NarrationError, explain, load_prompt, request_to_spec
from autoopensn.parse.table import convergence_table
from autoopensn.runner import (
    FakeRunner,
    LocalMPIRunner,
    ReferenceRunner,
    RemoteRunner,
    RunnerError,
)
from autoopensn.runner import reference as reference_runner
from autoopensn.runner.remote import DEFAULT_HOST as DEFAULT_REMOTE_HOST
from autoopensn.spec import Spec, SpecError
from autoopensn.store import RunStore
from autoopensn.study import render_points, run_study
from autoopensn.templates import TemplateError, list_templates, load_template
from autoopensn.tutorials import catalog

REPO = _ROOT
FIXTURES_DIR = REPO / "tests" / "fixtures"
SPEC_DIR = REPO / "tests" / "data"
RUN_ROOT = Path(os.environ.get("AUTOOPENSN_RUN_ROOT", REPO / "runs"))
CACHE = Path(os.environ.get("AUTOOPENSN_CACHE", RUN_ROOT / "cache.db"))

st.set_page_config(page_title="AutoOpenSn", page_icon="⚛", layout="wide")

REFERENCE = "Reference solver"
FIXTURES = "Recorded fixtures"
LOCAL = "Local MPI"
REMOTE = "Cluster over SSH"



# --- cached resources -------------------------------------------------------


@st.cache_resource(show_spinner=False)
def cached_template(name: str):
    return load_template(name)


@st.cache_data(show_spinner=False)
def cached_catalog():
    return catalog.load()


@st.cache_resource(show_spinner=False)
def cached_llm():
    """The provider-backed client, built once per session."""
    return llm_module.default_llm()


@st.cache_data(show_spinner=False)
def pack_search(query: str, k: int):
    hits = kp_bridge.search(query, k=k)
    return [
        {
            "citation": hit.citation(),
            "kind": hit.kind,
            "title": hit.title or "",
            "text": hit.text,
            "commit": hit.commit,
            "url": hit.url,
        }
        for hit in hits
    ]


# --- helpers ----------------------------------------------------------------


def spec_from_text(text: str) -> Spec:
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise SpecError("expected a mapping at the top level")
    spec = Spec(**raw)
    spec.validate_against()
    return spec


def build_runner(kind: str, fixture_dir: Path, host: str = ""):
    if kind == REFERENCE:
        # Not strict: a study touching an unsupported scenario reports those
        # cases as unavailable instead of stopping.
        return ReferenceRunner(strict=False)
    if kind == FIXTURES:
        return FakeRunner(fixture_dir)
    if kind == REMOTE:
        remote = RemoteRunner(host=host) if host else RemoteRunner()
        remote.preflight()
        return remote
    runner = LocalMPIRunner()
    runner.preflight()
    return runner


def execute(spec: Spec, kind: str, fixture_dir: Path, use_cache: bool, host: str = "") -> bool:
    """Run a study and put the results in session state. True on success."""
    try:
        runner = build_runner(kind, fixture_dir, host)
    except RunnerError as exc:
        st.error(str(exc))
        return False

    total = max(len(spec.expand()), 1)
    progress = st.progress(0.0, text="starting")
    done: list[str] = []

    def report(message: str) -> None:
        done.append(message)
        progress.progress(min(len(done) / total, 1.0), text=message)

    try:
        with RunStore(CACHE) as store:
            outcome = run_study(
                spec, runner, run_root=RUN_ROOT, store=store,
                use_cache=use_cache, progress=report,
            )
    except Exception:
        st.error("The study failed.")
        st.code(traceback.format_exc())
        return False
    finally:
        progress.empty()

    st.session_state["outcome_table"] = outcome.table
    st.session_state["outcome_summary"] = outcome.summary()
    st.session_state["outcome_root"] = str(outcome.run_root)
    st.session_state["outcome_spec"] = spec
    st.session_state.pop("explanation", None)
    return True


def show_results(table: pd.DataFrame) -> None:
    """The results table, its caveats, and two charts."""
    st.caption(st.session_state.get("outcome_summary", ""))

    if "replayed" in table and bool(table["replayed"].fillna(False).any()):
        st.warning(
            "Some rows were replayed from recorded fixtures rather than run. Their "
            "wall times are the recorded ones, and the fixtures in this repository "
            "are hand-written, so treat those numbers as a shape rather than a "
            "measurement. See tests/fixtures/README.md."
        )

    missing = table[table.get("failure", pd.Series(dtype=object)).astype(str).str.contains("no recorded run", na=False)]
    if len(missing):
        st.info(
            f"{len(missing)} case(s) have no recording and were not run. OpenSn is "
            "not installed here, so a newly generated study can be rendered and "
            "inspected but not executed. Switch the runner to Local MPI on a "
            "machine with OpenSn built, or record fixtures for these cases."
        )

    st.dataframe(
        convergence_table(table),
        hide_index=True,
        width="stretch",
        column_config={
            "Relative flux difference": st.column_config.NumberColumn(format="%.3e"),
            "Wall time (s)": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    if "sweeps_source" in table and (table["sweeps_source"] == "derived").any():
        st.caption(
            "Sweeps is derived from the inner-iteration count. OpenSn does not "
            "print a sweep count for a steady-state source solve at the pinned commit."
        )

    if "gold_passed" in table and table["gold_passed"].notna().any():
        passed = table["gold_passed"].dropna()
        if bool(passed.all()):
            st.success("Every case that could be checked matched the value OpenSn's own test suite records.")
        else:
            st.error("A case did not match the value OpenSn's own test suite records. See the gold columns below.")

    plottable = table.dropna(subset=["sweeps"]) if "sweeps" in table else pd.DataFrame()
    if len(plottable):
        left, right = st.columns(2)
        with left:
            if "rel_flux_diff" in plottable:
                nonzero = plottable[plottable["rel_flux_diff"].fillna(0) > 0]
                if len(nonzero):
                    st.altair_chart(
                        alt.Chart(nonzero)
                        .mark_point(size=120, filled=True)
                        .encode(
                            x=alt.X("sweeps:Q", title="Sweeps (derived)"),
                            y=alt.Y("rel_flux_diff:Q", title="Relative flux difference",
                                    scale=alt.Scale(type="log")),
                            color=alt.Color("case:N", title="Case"),
                            tooltip=["case", "sweeps", "rel_flux_diff", "wall_time"],
                        )
                        .properties(height=300, title="Accuracy against cost"),
                        width="stretch",
                    )
                    st.caption("The reference case is omitted: its difference is exactly zero.")
        with right:
            st.altair_chart(
                alt.Chart(plottable)
                .mark_bar()
                .encode(
                    y=alt.Y("case:N", sort="-x", title=None),
                    x=alt.X("sweeps:Q", title="Sweeps (derived)"),
                    tooltip=["case", "sweeps", "iterations", "wall_time"],
                )
                .properties(height=300, title="Cost by case"),
                width="stretch",
            )

    with st.expander("Every column"):
        st.dataframe(table, hide_index=True, width="stretch")
        st.download_button("Download CSV", table.to_csv(index=False),
                           file_name="results.csv", mime="text/csv",
                           key="download_csv")

    with st.expander("Raw output for one case"):
        root = Path(st.session_state.get("outcome_root", RUN_ROOT))
        cases = sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
        if cases:
            case = st.selectbox("Case", cases, key="stdout_case")
            path = root / case / "stdout.txt"
            st.code(path.read_text() if path.exists() else "no stdout recorded", language=None)
        else:
            st.caption("no run directories yet")


def show_explanation(table: pd.DataFrame, spec) -> None:
    """The narration stage, with its own caveats."""
    if not llm_ready:
        st.warning(llm_reason)
        return

    question = st.text_input(
        "Ask something specific about these results (optional)",
        placeholder="Which restart interval gives the best accuracy per sweep?",
        key="explain_question",
    )
    if st.button("Explain these results", type="primary", key="explain_button"):
        if not table["ok"].any():
            st.error("No case produced a result, so there is nothing to explain.")
        else:
            with st.spinner("reading the table…"):
                try:
                    st.session_state["explanation"] = explain(
                        table, spec, llm=cached_llm(), question=question or None
                    )
                except NarrationError as exc:
                    st.error(str(exc))

    explanation = st.session_state.get("explanation")
    if explanation is not None:
        st.markdown(explanation.text)
        for warning in explanation.warnings:
            st.warning(warning)
        st.caption(
            f"{explanation.model or 'model'} · {explanation.elapsed:.1f}s · "
            "written from the table above and nothing else"
        )


# --- sidebar ----------------------------------------------------------------

with st.sidebar:
    st.title("AutoOpenSn")
    st.caption(f"version {__version__}")

    st.subheader("Runner")
    runner_kind = st.radio(
        "How to run each sweep point",
        [REFERENCE, REMOTE, FIXTURES, LOCAL],
        help=(
            "The reference solver computes 1D scenarios here, in numpy, in under a "
            "second. It is not OpenSn. The other three run, or replay, OpenSn itself. "
            "The cluster runner is the only one that runs every scenario."
        ),
        label_visibility="collapsed",
    )
    remote_host = ""

    if runner_kind == REFERENCE:
        st.info(
            "**Not OpenSn.** These numbers come from this package's own 1D "
            "discrete-ordinates solver, which reproduces the values OpenSn's "
            "regression suite records for the scenarios it supports. It handles "
            "one-dimensional, one-group problems with isotropic scattering; "
            "anything else is reported as unavailable."
        )
        st.caption("supports: " + ", ".join(sorted(reference_runner.ADAPTERS)))

    fixture_sets = sorted(p.name for p in FIXTURES_DIR.iterdir() if p.is_dir()) if FIXTURES_DIR.is_dir() else []
    fixture_choice = (
        st.selectbox("Fixture set", fixture_sets,
                     index=fixture_sets.index("reed_gmres") if "reed_gmres" in fixture_sets else 0)
        if fixture_sets and runner_kind == FIXTURES else None
    )
    fixture_dir = FIXTURES_DIR / fixture_choice if fixture_choice else FIXTURES_DIR

    if runner_kind == LOCAL:
        try:
            st.success(f"launcher: {LocalMPIRunner().resolve_launcher()}")
            if LocalMPIRunner.pyopensn_available():
                st.success("pyopensn importable")
            else:
                st.error("pyopensn not importable; runs will fail at import")
        except RunnerError as exc:
            st.error(str(exc))
    elif runner_kind == REMOTE:
        remote_host = st.text_input(
            "SSH host",
            value=DEFAULT_REMOTE_HOST,
            help=(
                "An entry in your ~/.ssh/config, not a bare hostname. Cluster "
                "compute nodes are usually reachable only through a front end, "
                "and that jump belongs in the SSH config."
            ),
        )
        if st.button("Check the connection"):
            try:
                RemoteRunner(host=remote_host).preflight()
                st.success(f"{remote_host} reachable, OpenSn environment loads")
            except RunnerError as exc:
                st.error(str(exc))
    elif runner_kind == FIXTURES:
        st.info("Replaying recorded output. Nothing is executed.")

    use_cache = st.checkbox("Use the run cache", value=True)

    st.divider()
    st.subheader("Model")
    llm_ready = llm_module.available()
    llm_reason = ""
    if llm_ready:
        st.success("provider configured")
    else:
        llm_reason = (
            "No language-model provider is configured, so the ask and explain "
            "stages are unavailable. Add a key to the sister repository's .env; "
            "see .env.example."
        )
        st.warning("no provider configured")

    st.divider()
    st.subheader("Provenance")
    st.code(PINNED_OPENSN_COMMIT, language=None)
    st.caption("pinned OpenSn commit")
    scenarios = cached_catalog()
    if scenarios:
        st.caption(f"{len(scenarios)} scenarios from the OpenSn tutorials")
    if kp_bridge.available():
        try:
            st.success(f"knowledge pack: {kp_bridge.pack_commit()[:12]}…")
        except Exception as exc:
            st.error(str(exc))
    else:
        st.warning("sister repository not found; pack search is unavailable")

    st.divider()
    st.caption(
        "LLM at the edges, deterministic in the middle. Nothing between the spec "
        "and the results table involves a language model."
    )


ask_tab, scenarios_tab, pack_tab = st.tabs(["Ask", "Scenarios", "Knowledge pack"])


# --- 1. ask -----------------------------------------------------------------

with ask_tab:
    st.subheader("Describe the study you want")
    st.caption(
        "Your request is matched against the OpenSn tutorials, and a model turns "
        "it into a spec using only the parameters that scenario actually declares. "
        "The spec is range-checked in code before you see it."
    )

    prompt = st.text_area(
        "Request",
        value=st.session_state.get(
            "ask_prompt",
            "I'm studying how tightly I need to converge the inner solve on a simple "
            "1D slab problem. Compare GMRES tolerances 1e-4, 1e-6 and 1e-8 against a "
            "1e-10 reference, and also try restart intervals 5, 20 and 50.",
        ),
        height=110,
        key="ask_prompt",
        label_visibility="collapsed",
    )

    columns = st.columns([1, 1, 2])
    forced = columns[0].selectbox(
        "Scenario", ["(choose for me)"] + sorted(s.name for s in scenarios if s.sweepable)
    )
    review = columns[1].checkbox("Domain review", value=True,
                                 help="Have the sister repository's reviewer check the spec.")

    if st.button("Generate the spec", type="primary", disabled=not llm_ready):
        with st.spinner("choosing a scenario and writing a spec…"):
            try:
                draft = request_to_spec(
                    prompt,
                    llm=cached_llm(),
                    scenarios=scenarios,
                    template=None if forced.startswith("(") else forced,
                    review=review,
                )
                st.session_state["draft"] = draft
                st.session_state.pop("outcome_table", None)
            except Exception as exc:
                st.error(str(exc))
    if not llm_ready:
        st.warning(llm_reason)

    draft = st.session_state.get("draft")
    if draft is not None:
        if draft.candidates:
            st.caption("considered: " + ", ".join(s.name for s in draft.candidates))

        for note in draft.notes:
            st.warning(f"Reviewer note, on the documentation rather than the spec: {note}")

        if not draft.ok:
            st.error("No valid spec was produced.")
            for error in draft.errors:
                st.code(error, language=None)
            with st.expander("What the model actually returned"):
                for attempt in draft.attempts:
                    st.code(attempt, language="yaml")
        else:
            st.success(draft.summary())
            for error in draft.errors:
                st.caption(f"recovered from: {error}")

            left, right = st.columns([3, 2])
            with left:
                st.code(draft.spec.to_yaml(), language="yaml")
            with right:
                st.markdown("**This expands to**")
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "case": point.case_id,
                                "varies": ", ".join(f"{k}={v}" for k, v in sorted(point.varied.items())) or "nothing",
                                "reference": point.is_reference,
                            }
                            for point in draft.spec.expand()
                        ]
                    ),
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    f"model: {draft.model or 'unknown'} · {draft.elapsed:.1f}s · "
                    f"review: {draft.review_status}"
                )

            if st.button("Run this study", type="primary"):
                if execute(draft.spec, runner_kind, fixture_dir, use_cache, remote_host):
                    st.rerun()

            if draft.citations:
                with st.expander("Evidence the model was given"):
                    for citation in draft.citations:
                        st.markdown(f"**[{citation.number}] {citation.location}**")
                        st.code(citation.text[:1200], language=None)

    # --- edit, render, run ---------------------------------------------

    active = None
    draft = st.session_state.get("draft")
    if draft is not None and draft.ok:
        active = draft.spec

    with st.expander("Edit the spec, or load one from a file", expanded=active is None):
        st.caption(
            "The escape hatch. Generation chooses a scenario from a shortlist and "
            "can choose the wrong one; this is where you fix it, or start from a "
            "spec somebody already wrote."
        )
        saved = sorted(SPEC_DIR.glob("*.yaml")) if SPEC_DIR.is_dir() else []
        columns = st.columns([3, 1])
        source = columns[0].selectbox(
            "Load", ["(the generated spec)"] + [p.name for p in saved], key="spec_source"
        )
        if columns[1].button("Load", key="load_spec"):
            if source.startswith("("):
                if active is not None:
                    st.session_state["spec_text"] = active.to_yaml()
            else:
                match = next((p for p in saved if p.name == source), None)
                if match:
                    st.session_state["spec_text"] = match.read_text()

        spec_text = st.text_area(
            "Spec (YAML)", value=st.session_state.get("spec_text", ""),
            height=320, key="spec_text", label_visibility="collapsed",
        )
        if spec_text.strip():
            try:
                edited = spec_from_text(spec_text)
                st.success(f"Valid. {len(edited.expand())} run(s).")
                active = edited
            except Exception as exc:
                st.error(f"**Spec rejected.** {exc}")

    if active is not None:
        left, right = st.columns([2, 3], gap="large")
        with left:
            st.markdown("**This expands to**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "case": point.case_id,
                            "varies": ", ".join(
                                f"{k}={v}" for k, v in sorted(point.varied.items())
                            ) or "nothing",
                            "reference": point.is_reference,
                        }
                        for point in active.expand()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            if st.button("Run this study", type="primary", key="run_study"):
                if execute(active, runner_kind, fixture_dir, use_cache, remote_host):
                    st.rerun()

        with right:
            with st.expander("The script that would actually run"):
                rendered = render_points(active)
                case = st.selectbox(
                    "Case", [item.case_id for item in rendered], key="script_case"
                )
                selected = next(item for item in rendered if item.case_id == case)
                st.code(selected.script, language="python")
                st.download_button(
                    "Download this script", selected.script,
                    file_name=f"{case}.py", mime="text/x-python", key="download_script",
                )

                import difflib

                from autoopensn.store.db import strip_provenance

                template = cached_template(active.template)
                diff = "\n".join(
                    difflib.unified_diff(
                        template.render().splitlines(),
                        strip_provenance(selected.script).splitlines(),
                        f"{template.name} at defaults",
                        case,
                        lineterm="",
                    )
                )
                st.markdown("**Difference from the tutorial it came from**")
                st.code(diff or "identical", language="diff")

    # --- results and narrative ------------------------------------------

    table = st.session_state.get("outcome_table")
    if table is not None:
        st.divider()
        show_results(table)
        st.divider()
        show_explanation(table, st.session_state.get("outcome_spec"))
        with st.expander("The prompt the explanation uses"):
            st.markdown(load_prompt("explain"))


# --- 2. scenarios -----------------------------------------------------------

with scenarios_tab:
    st.subheader("Scenario library")
    st.caption(
        "Every OpenSn tutorial at the pinned commit, with the parameters read out "
        "of its own script and, for nearly all of them, the correct answer recorded "
        "by OpenSn's regression suite."
    )

    query = st.text_input("Search", placeholder="quadrature, mesh refinement, eigenvalue…")
    listed = catalog.rank(query, scenarios, limit=40) if query else scenarios

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "scenario": s.name,
                    "title": s.title,
                    "area": s.section,
                    "knobs": len(s.parameters),
                    "solver knobs": len(s.solver_parameters),
                    "verified": s.has_gold,
                    "sweepable": s.sweepable,
                    "needs data": bool(s.data_files),
                }
                for s in listed
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    chosen = st.selectbox("Inspect", [s.name for s in listed])
    if chosen:
        scenario = catalog.by_name(chosen, scenarios)
        st.markdown(f"### {scenario.title}")
        st.write(scenario.summary)
        st.caption(f"from `doc/source/tutorials/{scenario.notebook}`")
        if scenario.gold:
            st.markdown("**Recorded correct answers**")
            st.dataframe(
                pd.DataFrame(
                    [{"key": g.key, "value": g.value, "abs_tol": g.abs_tol} for g in scenario.gold]
                ),
                hide_index=True,
                width="stretch",
            )
        try:
            template = cached_template(chosen)
        except TemplateError:
            template = None
        if template is not None:
            st.markdown("**Parameters**")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "parameter": name,
                            "group": d.group,
                            "type": d.type,
                            "default": str(d.default),
                            "range": (
                                "one of " + ", ".join(map(str, d.choices)) if d.choices
                                else " ".join(
                                    filter(None, [
                                        f"≥ {d.minimum:g}" if d.minimum is not None else "",
                                        f"≤ {d.maximum:g}" if d.maximum is not None else "",
                                    ])
                                ) or "any"
                            ),
                            "bound from": d.provenance,
                        }
                        for name, d in sorted(template.parameters.items())
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(
                "A bound marked `inferred` is a guard scaled from the tutorial's own "
                "value, not a limit OpenSn enforces. `opensn` and `physics` bounds are real."
            )


# --- 3. knowledge pack ------------------------------------------------------

with pack_tab:
    st.subheader("Ask the knowledge pack")
    st.caption(
        "Retrieval over the version-pinned OpenSn documentation and source, through "
        "the sister repository. This is what grounds spec generation. Every hit "
        "carries a citation; retrieval never returns text without a location."
    )

    if not kp_bridge.available():
        st.warning(
            "The sister repository was not found. Set AUTOOPENSN_KP_REPO to your "
            "Code_assistant_TAU checkout to enable search."
        )
    else:
        query = st.text_input("Query", value="GMRES restart interval", key="pack_query")
        count = st.slider("Hits", 1, 15, 5)
        if query:
            try:
                hits = pack_search(query, count)
            except Exception as exc:
                st.error(str(exc))
            else:
                if not hits:
                    st.caption("no hits")
                for index, hit in enumerate(hits, start=1):
                    with st.expander(f"{index}. {hit['title'] or hit['citation']}  ·  {hit['kind']}"):
                        st.caption(hit["citation"])
                        st.code(hit["text"][:4000], language=None)
                        if hit["url"]:
                            st.markdown(f"[source]({hit['url']})")
                        if not hit["commit"]:
                            st.warning(
                                "This passage is from scraped documentation whose "
                                "revision is unknown and may predate the pinned commit."
                            )
