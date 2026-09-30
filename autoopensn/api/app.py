"""The HTTP surface of the workflow engine.

Every stage of the pipeline is reachable here, in the same order and with the
same guarantees it has on the command line: a request becomes a spec, a spec
becomes scripts, scripts run, runs become a table, and the table becomes prose.

Three things this layer does *not* do, because they are the point of the design:

**It does not run anything a spec has not validated.** ``/render`` and ``/runs``
both build a ``Spec`` and call ``validate_against`` before doing anything. A
frontend that edits YAML by hand gets the same range checking a generated spec
gets, and the error comes back as a 422 with the exact message.

**It does not let a caller choose the script.** The only thing that becomes a
script is a template plus validated parameters. There is no endpoint that takes
Python and runs it, and there should never be one.

**It does not hold a connection open for a simulation.** ``/runs`` returns a job
id immediately, and ``/runs/{id}`` is polled. See ``jobs.py``.

The app is created by a factory so a test can build one with a scripted model
and no network, which is what the whole suite does.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from autoopensn import PINNED_OPENSN_COMMIT, __version__
from autoopensn.api.jobs import JobRegistry
from autoopensn.llm import LLMError, LLMUnavailable
from autoopensn.narrate import NarrationError, chat, explain, request_to_spec
from autoopensn.parse.table import convergence_table
from autoopensn.runner.remote import PROBE_TIMEOUT
from autoopensn.runner import (
    FakeRunner,
    LocalMPIRunner,
    ReferenceRunner,
    RemoteRunner,
    Runner,
    RunnerError,
)
from autoopensn.spec import Spec, SpecError
from autoopensn.store import RunStore
from autoopensn.study import render_points, run_study
from autoopensn.templates import TemplateError, load_template
from autoopensn.tutorials import catalog

DEFAULT_FIXTURES = Path(__file__).resolve().parent.parent.parent / "tests" / "fixtures" / "reed_gmres"

# Vite's dev server runs on its own port, so the browser's origin is not the
# API's. Restricted to loopback: this app is a local tool, and a wildcard here
# would let any page the user visits drive their cluster account.
DEV_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]


# --- wire formats -----------------------------------------------------------


class SpecRequest(BaseModel):
    prompt: str = Field(min_length=1)
    template: Optional[str] = None
    ground: bool = True
    review: bool = True


class SpecYaml(BaseModel):
    spec_yaml: str = Field(min_length=1)


class RunRequestBody(SpecYaml):
    runner: str = "auto"
    host: Optional[str] = None
    fixtures: Optional[str] = None
    use_cache: bool = True
    run_root: str = "runs"


class ExplainRequest(BaseModel):
    table: list[dict[str, Any]]
    spec_yaml: Optional[str] = None
    question: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    table: list[dict[str, Any]]
    messages: list[ChatMessage]
    spec_yaml: Optional[str] = None


# --- helpers ----------------------------------------------------------------


def _jsonable(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Table rows as JSON-safe records.

    NaN and infinity are not JSON, and a silent failure to serialise one row of
    a results table is exactly the kind of gap that gets read as "the run
    produced nothing" rather than "the value was missing".
    """
    records = frame.to_dict(orient="records")
    for row in records:
        for key, value in row.items():
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                row[key] = None
            elif hasattr(value, "item"):  # numpy scalars
                row[key] = value.item()
    return records


def _model_failure(exc: Exception) -> HTTPException:
    """Turn a model failure into a status a caller can act on.

    503, not 500. A missing provider is a configuration problem with a known
    fix, and the message already says what the fix is — but only if it reaches
    the caller. Left as an unhandled exception it becomes a bare "Internal
    Server Error" in the browser and a useful sentence in a log nobody reading
    the page will ever see. That is the first thing a new installation does,
    so it is the worst place to lose the explanation.
    """
    status = 503 if isinstance(exc, LLMUnavailable) else 502
    return HTTPException(status_code=status, detail=str(exc))


def _load_spec(spec_yaml: str) -> Spec:
    """Parse and validate, or raise the HTTP error that says why.

    422 rather than 400: the document parsed, and the values are what is wrong.
    A frontend distinguishes "you sent me nonsense" from "that tolerance is out
    of range", and only the second is worth showing beside the field.
    """
    try:
        document = yaml.safe_load(spec_yaml)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=400, detail=f"could not read the spec: {exc}") from exc
    if not isinstance(document, dict):
        raise HTTPException(
            status_code=400, detail="could not read the spec: expected a mapping at the top level"
        )
    try:
        spec = Spec(**document)
        spec.validate_against()
    except (SpecError, TemplateError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # a field of the wrong type, an unknown key
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return spec


def _choose_runner(host: Optional[str]) -> tuple[Runner, str]:
    """Pick the best runner available, and say which and why.

    The interface does not ask. A person describing a physics study should not
    have to know what a fixture replay is, and the right answer is almost always
    "the cluster if you can reach it, this machine's own solver if you cannot".

    What the interface must never do is run something and leave the person
    guessing which engine produced the numbers, so the reason is returned
    alongside and shown.
    """
    try:
        # A short probe: this decides whether to use the cluster, and the person
        # is looking at a spinner while it happens. The runner that actually
        # runs gets the ordinary, patient timeout.
        probe = RemoteRunner(host=host, connect_timeout=PROBE_TIMEOUT) if host else RemoteRunner(
            connect_timeout=PROBE_TIMEOUT
        )
        probe.preflight()
        return (
            RemoteRunner(host=host) if host else RemoteRunner()
        ), f"Running real OpenSn on {probe.host}."
    except RunnerError as exc:
        first_line = str(exc).strip().splitlines()[0]
        return (
            ReferenceRunner(strict=False),
            "The cluster is not reachable, so this ran in AutoOpenSn's own 1D "
            f"solver instead. These are not OpenSn's numbers. ({first_line})",
        )


def _build_runner(kind: str, host: Optional[str], fixtures: Optional[str]) -> tuple[Runner, str]:
    """The same runners the CLI offers, chosen by name, or 'auto'."""
    if kind == "auto":
        return _choose_runner(host)
    if kind == "reference":
        # Not strict: a study touching an unsupported scenario reports those
        # cases as unavailable rather than stopping the whole run.
        return ReferenceRunner(strict=False), "AutoOpenSn's own 1D solver. Not OpenSn."
    if kind == "remote":
        remote = RemoteRunner(host=host) if host else RemoteRunner()
        try:
            remote.preflight()
        except RunnerError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return remote, f"Running real OpenSn on {remote.host}."
    if kind == "fake":
        try:
            return (
                FakeRunner(Path(fixtures) if fixtures else DEFAULT_FIXTURES),
                "Replaying recorded output. Nothing was executed.",
            )
        except RunnerError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    if kind == "local":
        local = LocalMPIRunner()
        try:
            local.preflight()
        except RunnerError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return local, "Running real OpenSn on this machine."
    raise HTTPException(
        status_code=422,
        detail=f"unknown runner {kind!r}; expected 'auto', 'reference', 'remote', "
        "'fake' or 'local'",
    )


def _scenario_summary(scenario: Any) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "title": getattr(scenario, "title", "") or scenario.name,
        "summary": getattr(scenario, "summary", ""),
        "section": getattr(scenario, "section", ""),
        "parameters": list(getattr(scenario, "parameters", []) or []),
        "solver_parameters": list(getattr(scenario, "solver_parameters", []) or []),
        "sweepable": bool(getattr(scenario, "sweepable", False)),
        "has_gold": bool(getattr(scenario, "has_gold", False)),
        "num_procs": getattr(scenario, "num_procs", 1),
    }


# --- the app ----------------------------------------------------------------


def create_app(
    *,
    llm: Any = None,
    store: Optional[RunStore] = None,
    registry: Optional[JobRegistry] = None,
    run_root: Path = Path("runs"),
) -> FastAPI:
    """Build the API.

    ``llm`` is injected rather than looked up so that the test suite can pass a
    scripted model and never reach a provider. Left None, each narration stage
    resolves the configured client itself, exactly as the CLI does.
    """
    app = FastAPI(
        title="AutoOpenSn",
        version=__version__,
        summary="Parameter studies for the OpenSn neutron transport code.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=DEV_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    jobs = registry or JobRegistry()
    app.state.jobs = jobs
    app.state.store = store
    app.state.run_root = Path(run_root)

    # --- status ---

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        """What this installation can actually do, before anything is asked of it."""
        from autoopensn import kp_bridge, llm as llm_module

        return {
            "version": __version__,
            "pack_commit": PINNED_OPENSN_COMMIT,
            "llm_available": llm is not None or llm_module.available(),
            "knowledge_pack_available": kp_bridge.available(),
            "scenarios": len(catalog.load()),
        }

    # --- the scenario library ---

    @app.get("/api/scenarios")
    def scenarios(q: str = "", limit: int = 50) -> dict[str, Any]:
        """The catalogue, ranked against ``q`` when given.

        The ranking is the same deterministic one spec generation uses to build
        its shortlist, so what a person browses and what the model chooses from
        are the same ordering.
        """
        available = catalog.load()
        chosen = catalog.rank(q, available, limit=limit) if q else available[:limit]
        return {"total": len(available), "scenarios": [_scenario_summary(s) for s in chosen]}

    @app.get("/api/scenarios/{name}")
    def scenario(name: str) -> dict[str, Any]:
        found = catalog.by_name(name, catalog.load())
        if found is None:
            raise HTTPException(status_code=404, detail=f"no scenario named {name!r}")
        try:
            template = load_template(name)
        except TemplateError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        record = _scenario_summary(found)
        record["declarations"] = [
            {
                "name": declaration.name,
                "group": getattr(declaration, "group", ""),
                "type": getattr(declaration, "type", ""),
                "is_list": bool(getattr(declaration, "is_list", False)),
                "default": getattr(declaration, "default", None),
                "minimum": getattr(declaration, "minimum", None),
                "maximum": getattr(declaration, "maximum", None),
                "choices": list(getattr(declaration, "choices", []) or []),
                "description": getattr(declaration, "description", ""),
                "provenance": getattr(declaration, "provenance", ""),
            }
            for declaration in template.parameters.values()
        ]
        return record

    # --- prompt to spec ---

    @app.post("/api/spec")
    def generate_spec(body: SpecRequest) -> dict[str, Any]:
        """Turn a request into a validated spec, or explain why it could not be.

        A draft that failed still returns 200. The attempts and the validation
        errors are the useful part of that outcome, and an error status would
        push a frontend into an error path that shows none of them.
        """
        try:
            draft = request_to_spec(
                body.prompt,
                llm=llm,
                template=body.template,
                ground=body.ground,
                review=body.review,
            )
        except NarrationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMError as exc:
            raise _model_failure(exc) from exc

        return {
            "ok": draft.ok,
            "summary": draft.summary(),
            "spec_yaml": draft.spec.to_yaml() if draft.spec else None,
            # The same object serialised twice, rather than converted in the
            # browser. A YAML parser in the front end would be a second
            # implementation of what a spec means, and the two would eventually
            # disagree about something like an unsigned exponent.
            "spec_json": (
                json.dumps(draft.spec.model_dump(mode="json"), indent=2)
                if draft.spec
                else None
            ),
            "cases": len(draft.spec.expand()) if draft.spec else 0,
            "candidates": [_scenario_summary(s) for s in draft.candidates],
            "citations": [
                {
                    "number": c.number,
                    "location": c.location,
                    "title": c.title or "",
                    "text": c.as_evidence(400),
                }
                for c in draft.citations
            ],
            "notes": list(draft.notes),
            "errors": list(draft.errors),
            "attempts": list(draft.attempts),
            "revisions": draft.revisions,
            "review_status": draft.review_status,
            "model": draft.model,
            "elapsed": draft.elapsed,
        }

    # --- spec to scripts ---

    @app.post("/api/render")
    def render(body: SpecYaml) -> dict[str, Any]:
        """Every script this spec would run, without running any of them."""
        spec = _load_spec(body.spec_yaml)
        points = render_points(spec)
        return {
            "template": spec.template,
            "cases": [
                {
                    "case_id": point.case_id,
                    "script": point.script,
                    "parameters": point.point.parameters,
                }
                for point in points
            ],
        }

    # --- running ---

    @app.post("/api/runs", status_code=202)
    def start_run(body: RunRequestBody) -> dict[str, Any]:
        """Start a study and return its id. Does not wait."""
        spec = _load_spec(body.spec_yaml)
        runner, note = _build_runner(body.runner, body.host, body.fixtures)
        points = len(spec.expand())
        current_store = app.state.store

        def work(say):
            result = run_study(
                spec,
                runner,
                run_root=Path(body.run_root),
                store=current_store,
                use_cache=body.use_cache,
                progress=say,
            )
            table = convergence_table(result.table)
            return {
                "table": _jsonable(table),
                "summary": result.summary(),
                "run_root": str(result.run_root),
                "cache_hits": list(result.cache_hits),
            }

        job = jobs.start(
            work, total=points, label=f"{spec.name} ({spec.template})", note=note
        )
        return {
            "id": job.id,
            "status": job.status,
            "total": points,
            "runner": runner.name,
            "note": note,
        }

    @app.get("/api/runs")
    def list_runs() -> dict[str, Any]:
        return {"jobs": [job.as_dict(include_table=False) for job in jobs.list()]}

    @app.get("/api/runs/{job_id}")
    def run_status(job_id: str) -> dict[str, Any]:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no job {job_id!r}")
        return job.as_dict()

    # --- results to prose ---

    @app.post("/api/explain")
    def explain_table(body: ExplainRequest) -> dict[str, Any]:
        """Narrate a finished table.

        The table is sent back by the caller rather than read from the job, so
        that a person can narrate a table they have filtered or edited. The
        narration stage checks every figure in its own prose against this
        table, so a caller cannot get a number blessed that is not in it.
        """
        if not body.table:
            raise HTTPException(status_code=422, detail="there are no results to explain")
        spec = _load_spec(body.spec_yaml) if body.spec_yaml else None
        try:
            result = explain(
                pd.DataFrame(body.table), spec, llm=llm, question=body.question
            )
        except NarrationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMError as exc:
            raise _model_failure(exc) from exc
        return {
            "text": result.text,
            "unsupported": list(getattr(result, "unsupported", []) or []),
            "model": getattr(result, "model", None),
            "elapsed": getattr(result, "elapsed", 0.0),
        }

    @app.post("/api/chat")
    def chat_about(body: ChatRequest) -> dict[str, Any]:
        """Answer a follow-up about a finished table.

        The table is sent every turn rather than remembered, so a long
        conversation cannot drift onto numbers the model is recalling instead of
        reading.
        """
        if not body.table:
            raise HTTPException(status_code=422, detail="there are no results to discuss")
        spec = _load_spec(body.spec_yaml) if body.spec_yaml else None
        try:
            answer = chat(
                pd.DataFrame(body.table),
                [m.model_dump() for m in body.messages],
                spec,
                llm=llm,
            )
        except NarrationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LLMError as exc:
            raise _model_failure(exc) from exc
        return {
            "text": answer.text,
            "warnings": list(answer.warnings),
            "model": answer.model,
            "elapsed": answer.elapsed,
        }

    # --- the knowledge pack ---

    @app.get("/api/search")
    def search(q: str, limit: int = 8) -> dict[str, Any]:
        """Version-pinned OpenSn documentation and source, with a citation."""
        from autoopensn import kp_bridge

        if not kp_bridge.available():
            raise HTTPException(
                status_code=503,
                detail="the knowledge pack is not available; set AUTOOPENSN_KP_REPO",
            )
        if not q.strip():
            raise HTTPException(status_code=422, detail="a query is required")
        hits = kp_bridge.search(q, limit=limit)
        return {
            "hits": [
                {
                    "source": getattr(hit, "source", "") or getattr(hit, "path", ""),
                    "text": getattr(hit, "text", "")[:2000],
                    "score": getattr(hit, "score", None),
                }
                for hit in hits
            ]
        }

    return app
