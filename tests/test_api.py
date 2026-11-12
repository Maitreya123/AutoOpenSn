"""Tests for the HTTP surface.

Nothing here reaches a provider or a cluster. The language model is a
``ScriptedLLM`` and the runner is the reference solver, which computes the 1D
scenarios for real in well under a second — so these are genuine end-to-end
runs through the API, not mocks of one.

The properties worth protecting are the ones that make this layer safe to point
a browser at: a spec is validated before anything runs, an invalid one is
refused with the reason, and there is no route that takes Python and executes
it.
"""

from __future__ import annotations

import time

import pytest
import yaml
from fastapi.testclient import TestClient

from autoopensn.api import JobRegistry, create_app
from autoopensn.llm import ScriptedLLM

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
    values: [1.0e-4, 1.0e-6]
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
def client(tmp_path):
    app = create_app(llm=ScriptedLLM([GOOD_SPEC]), run_root=tmp_path / "runs")
    return TestClient(app)


def wait_for(client, job_id: str, timeout: float = 90.0) -> dict:
    """Poll a job to completion, the way a frontend would."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/runs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


# --- status -----------------------------------------------------------------


def test_health_reports_what_is_available(client):
    body = client.get("/api/health").json()
    assert body["pack_commit"]
    assert body["scenarios"] > 0
    assert set(body) >= {"version", "llm_available", "knowledge_pack_available"}


# --- the scenario library ---------------------------------------------------


def test_scenarios_are_listed(client):
    body = client.get("/api/scenarios").json()
    assert body["total"] > 0
    assert body["scenarios"]
    assert all("name" in s for s in body["scenarios"])


def test_scenarios_rank_against_a_query(client):
    """The browser and the model must see the same ordering."""
    body = client.get("/api/scenarios", params={"q": "reed", "limit": 5}).json()
    names = [s["name"] for s in body["scenarios"]]
    assert "reed_1d" in names


def test_one_scenario_carries_its_declared_ranges(client):
    body = client.get("/api/scenarios/reed_1d").json()
    assert body["name"] == "reed_1d"
    assert body["declarations"]
    named = {d["name"] for d in body["declarations"]}
    assert "l_abs_tol" in named


def test_unknown_scenario_is_a_404(client):
    assert client.get("/api/scenarios/not_a_scenario").status_code == 404


# --- validation, which is the point -----------------------------------------


def test_render_returns_one_script_per_point(client):
    body = client.post("/api/render", json={"spec_yaml": GOOD_SPEC}).json()
    assert body["template"] == "reed_1d"
    # Two swept values plus the reference.
    assert len(body["cases"]) == 3
    assert all(case["script"].strip() for case in body["cases"])


def test_render_refuses_a_value_outside_its_declared_range(client):
    response = client.post("/api/render", json={"spec_yaml": OUT_OF_RANGE})
    assert response.status_code == 422
    assert "gmres_restart_interval" in response.json()["detail"]


def test_render_refuses_a_parameter_that_does_not_exist(client):
    response = client.post("/api/render", json={"spec_yaml": UNKNOWN_PARAMETER})
    assert response.status_code == 422
    assert "magic_accelerator" in response.json()["detail"]


def test_a_run_is_refused_before_it_starts_if_the_spec_is_invalid(client):
    """Validation must happen in the request, not on the worker thread.

    A 202 followed by a failed job would mean the caller has to poll to learn
    that the tolerance it typed was out of range.
    """
    response = client.post(
        "/api/runs", json={"spec_yaml": OUT_OF_RANGE, "runner": "reference"}
    )
    assert response.status_code == 422


def test_malformed_yaml_is_a_400_not_a_422(client):
    """"Unreadable" and "read, but wrong" are different things to a frontend."""
    response = client.post("/api/render", json={"spec_yaml": "this: [is: not: yaml"})
    assert response.status_code == 400


def test_there_is_no_route_that_runs_arbitrary_code(client):
    """The one invariant the whole design rests on."""
    paths = {route.path for route in client.app.routes}
    assert not {p for p in paths if "exec" in p or "eval" in p or "script" in p}

    # And the run endpoint takes a spec, not a script.
    response = client.post(
        "/api/runs", json={"script": "import os; os.system('id')", "runner": "reference"}
    )
    assert response.status_code == 422


# --- running ----------------------------------------------------------------


def test_a_study_runs_end_to_end_and_produces_a_table(client):
    started = client.post(
        "/api/runs",
        json={"spec_yaml": GOOD_SPEC, "runner": "reference", "use_cache": False},
    )
    assert started.status_code == 202
    body = started.json()
    assert body["total"] == 3

    finished = wait_for(client, body["id"])
    assert finished["status"] == "succeeded", finished.get("error")
    assert finished["table"]
    assert len(finished["table"]) == 3
    assert finished["completed"] == 3


def test_progress_is_reported_while_running(client):
    started = client.post(
        "/api/runs",
        json={"spec_yaml": GOOD_SPEC, "runner": "reference", "use_cache": False},
    ).json()
    finished = wait_for(client, started["id"])
    assert finished["progress"], "a study that says nothing is one nobody can watch"


def test_table_values_are_json_safe(client):
    """NaN is not JSON; a row that fails to serialise reads as a missing run."""
    started = client.post(
        "/api/runs",
        json={"spec_yaml": GOOD_SPEC, "runner": "reference", "use_cache": False},
    ).json()
    finished = wait_for(client, started["id"])
    for row in finished["table"]:
        for value in row.values():
            assert value is None or isinstance(value, (str, int, float, bool, list, dict))


def test_a_failed_run_reports_why_rather_than_hanging(client, tmp_path):
    """A job stuck on "running" is the one outcome with no recovery."""
    registry = JobRegistry()

    def explode(_say):
        raise RuntimeError("the cluster caught fire")

    job = registry.start(explode, total=1)
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not registry.get(job.id).done:
        time.sleep(0.02)

    finished = registry.get(job.id)
    assert finished.status == "failed"
    assert "cluster caught fire" in finished.error


def test_unknown_job_is_a_404(client):
    assert client.get("/api/runs/deadbeef").status_code == 404


def test_jobs_are_listed_without_their_tables(client):
    """A listing must stay small; a sweep's table can be thousands of rows."""
    started = client.post(
        "/api/runs",
        json={"spec_yaml": GOOD_SPEC, "runner": "reference", "use_cache": False},
    ).json()
    wait_for(client, started["id"])
    listing = client.get("/api/runs").json()
    assert listing["jobs"]
    assert "table" not in listing["jobs"][0]
    assert listing["jobs"][0]["label"]


def test_an_unknown_runner_is_refused(client):
    response = client.post(
        "/api/runs", json={"spec_yaml": GOOD_SPEC, "runner": "quantum"}
    )
    assert response.status_code == 422
    assert "quantum" in response.json()["detail"]


# --- narration --------------------------------------------------------------


def test_explain_narrates_a_table(tmp_path):
    app = create_app(
        llm=ScriptedLLM(["The tighter tolerance converged in more sweeps."]),
        run_root=tmp_path / "runs",
    )
    client = TestClient(app)
    response = client.post(
        "/api/explain",
        json={
            "table": [
                {"case_id": "a", "sweeps": 12, "wall_time": 0.4},
                {"case_id": "b", "sweeps": 20, "wall_time": 0.6},
            ],
            "spec_yaml": GOOD_SPEC,
        },
    )
    assert response.status_code == 200
    assert response.json()["text"]


def test_explain_refuses_an_empty_table(client):
    response = client.post("/api/explain", json={"table": []})
    assert response.status_code == 422


# --- prompt to spec ---------------------------------------------------------


def test_a_prompt_becomes_a_validated_spec(client):
    response = client.post(
        "/api/spec", json={"prompt": "compare inner tolerances", "ground": False, "review": False}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["ok"]
    assert body["spec_yaml"]
    assert body["cases"] == 3
    assert body["candidates"]


def test_a_spec_the_model_could_not_produce_is_still_a_200(tmp_path):
    """The attempts and the errors are the useful part of that outcome."""
    app = create_app(llm=ScriptedLLM([OUT_OF_RANGE, OUT_OF_RANGE]), run_root=tmp_path)
    client = TestClient(app)
    body = client.post(
        "/api/spec", json={"prompt": "something impossible", "ground": False, "review": False}
    ).json()
    assert body["ok"] is False
    assert body["errors"]
    assert body["attempts"]


# --- choosing an engine for the user ----------------------------------------


def test_auto_falls_back_and_says_so(client):
    """The interface does not ask which engine; it must still say which ran.

    With no cluster reachable, `auto` has to land on the reference solver and
    state plainly that these are not OpenSn's numbers. A table that does not say
    where it came from is one nobody should quote.
    """
    started = client.post(
        "/api/runs",
        json={"spec_yaml": GOOD_SPEC, "runner": "auto", "use_cache": False},
    )
    assert started.status_code == 202
    body = started.json()
    assert body["runner"] == "reference"
    assert "not OpenSn" in body["note"]

    finished = wait_for(client, body["id"])
    assert finished["status"] == "succeeded"
    assert finished["note"] == body["note"]


def test_auto_is_the_default_runner(client):
    """The flow sends a prompt and a spec; it should not have to send a runner."""
    started = client.post("/api/runs", json={"spec_yaml": GOOD_SPEC, "use_cache": False})
    assert started.status_code == 202
    assert started.json()["note"]


# --- chatting about results -------------------------------------------------


def test_chat_answers_a_follow_up(tmp_path):
    app = create_app(
        llm=ScriptedLLM(["Restart 5 needed more sweeps because the interval truncates."]),
        run_root=tmp_path,
    )
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "table": [
                {"Case": "restart=5", "Sweeps": 11},
                {"Case": "restart=20", "Sweeps": 8},
            ],
            "messages": [{"role": "user", "content": "why did restart 5 take longer?"}],
        },
    )
    assert response.status_code == 200
    assert response.json()["text"]


def test_chat_refuses_when_the_last_word_is_its_own(tmp_path):
    """A turn must answer the reader, not continue talking to itself."""
    app = create_app(llm=ScriptedLLM(["unused"]), run_root=tmp_path)
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={
            "table": [{"Case": "a", "Sweeps": 1}],
            "messages": [{"role": "assistant", "content": "Here is what I think."}],
        },
    )
    assert response.status_code == 422


def test_chat_refuses_an_empty_table(tmp_path):
    app = create_app(llm=ScriptedLLM(["unused"]), run_root=tmp_path)
    client = TestClient(app)
    response = client.post(
        "/api/chat",
        json={"table": [], "messages": [{"role": "user", "content": "well?"}]},
    )
    assert response.status_code == 422


def test_the_spec_comes_back_in_both_formats(client):
    """Downloadable as YAML for the engine, as JSON for anything else.

    Serialised from one validated object rather than converted in the browser,
    so the two cannot come to disagree about what the spec says.
    """
    import json as json_module

    body = client.post(
        "/api/spec", json={"prompt": "compare tolerances", "ground": False, "review": False}
    ).json()
    assert body["ok"]
    assert body["spec_yaml"]
    assert body["spec_json"]

    from_json = json_module.loads(body["spec_json"])
    from_yaml = yaml.safe_load(body["spec_yaml"])
    assert from_json["template"] == from_yaml["template"]
    assert from_json["name"] == from_yaml["name"]
    assert from_json["sweep"] == from_yaml["sweep"]


def test_a_failed_draft_has_no_spec_in_either_format(tmp_path):
    app = create_app(llm=ScriptedLLM([OUT_OF_RANGE, OUT_OF_RANGE]), run_root=tmp_path)
    body = TestClient(app).post(
        "/api/spec", json={"prompt": "impossible", "ground": False, "review": False}
    ).json()
    assert body["spec_yaml"] is None
    assert body["spec_json"] is None


# --- a machine with no model configured -------------------------------------


def test_no_provider_explains_itself_instead_of_a_500(tmp_path, monkeypatch):
    """The first thing a fresh install does must not be an opaque error.

    With no sister repository there is no provider, and the exception already
    says exactly what to set. Left unhandled it became "Internal Server Error"
    in the browser and a useful sentence in a log nobody reading the page will
    ever open.
    """
    from autoopensn.llm import LLMUnavailable

    app = create_app(run_root=tmp_path)
    monkeypatch.setattr(
        "autoopensn.llm.default_llm",
        lambda *a, **k: (_ for _ in ()).throw(
            LLMUnavailable("the sister repository is not present, so no provider is configured.")
        ),
    )
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/spec", json={"prompt": "compare tolerances"}
    )
    assert response.status_code == 503
    assert "sister repository" in response.json()["detail"]


def test_a_provider_that_fails_mid_call_is_a_502(tmp_path, monkeypatch):
    """Distinct from 503: configured, but the provider did not answer."""
    from autoopensn.llm import LLMError

    app = create_app(run_root=tmp_path)
    monkeypatch.setattr(
        "autoopensn.llm.default_llm",
        lambda *a, **k: (_ for _ in ()).throw(LLMError("All LLM providers failed.")),
    )
    response = TestClient(app, raise_server_exceptions=False).post(
        "/api/explain", json={"table": [{"Case": "a", "Sweeps": 3}]}
    )
    assert response.status_code == 502


def test_health_reports_the_pack_not_just_the_sister_code(client, monkeypatch):
    """A fresh clone of the sister repository has its code and no pack."""
    from autoopensn import kp_bridge

    monkeypatch.setattr(kp_bridge, "available", lambda: True)
    monkeypatch.setattr(kp_bridge, "pack_available", lambda *a, **k: False)
    assert client.get("/api/health").json()["knowledge_pack_available"] is False


# --- the catalogue and the cache, through the page's API -------------------------


def _run_and_wait(client, spec_yaml, **extra):
    started = client.post("/api/runs", json={"spec_yaml": spec_yaml, "runner": "reference", **extra}).json()
    return wait_for(client, started["id"])


def test_the_page_now_uses_the_run_cache(client):
    """The app used to be built with no store: every study re-ran every case."""
    first = _run_and_wait(client, GOOD_SPEC)
    second = _run_and_wait(client, GOOD_SPEC)
    assert first["cache_hits"] == []
    assert len(second["cache_hits"]) == 3


def test_a_finished_study_is_in_the_catalogue(client):
    job = _run_and_wait(client, GOOD_SPEC)
    assert job["study_id"]
    listed = client.get("/api/studies").json()["studies"]
    assert any(s["id"] == job["study_id"] for s in listed)
    study = client.get(f"/api/studies/{job['study_id']}").json()
    assert len(study["cases"]) == 3
    assert study["engine_note"] == "AutoOpenSn's own 1D solver. Not OpenSn."


def test_groups_are_served(client):
    _run_and_wait(client, GOOD_SPEC)
    groups = client.get("/api/groups").json()["groups"]
    assert groups[0]["group_key"] == "reed_1d:l_abs_tol"


def test_an_unknown_study_is_a_404(client):
    assert client.get("/api/studies/nope").status_code == 404


def test_run_directories_stay_inside_the_apps_run_root(client, tmp_path):
    """They used to land in the repository's own runs/ folder."""
    _run_and_wait(client, GOOD_SPEC)
    assert (tmp_path / "runs" / "tolerance_study").is_dir()


def test_a_finished_study_comes_with_its_charts(client):
    job = _run_and_wait(client, GOOD_SPEC)
    assert job["charts"], "a tolerance sweep has something to chart"
    assert job["charts"][0]["title"]["text"] == "Varying l_abs_tol"


def test_a_catalogued_study_can_be_charted_later(client):
    job = _run_and_wait(client, GOOD_SPEC)
    charts = client.get(f"/api/studies/{job['study_id']}/charts").json()["charts"]
    assert charts and charts[0]["title"]["text"] == "Varying l_abs_tol"


def test_a_group_can_be_compared(client):
    _run_and_wait(client, GOOD_SPEC)
    _run_and_wait(client, GOOD_SPEC)
    body = client.get("/api/compare", params={"group": "reed_1d:l_abs_tol"}).json()
    assert len(body["studies"]) == 2 and body["charts"]


# --- field pictures --------------------------------------------------------------


def _study_with_a_field(client):
    meshio = pytest.importorskip("meshio")
    pytest.importorskip("matplotlib")
    import numpy as np

    job = _run_and_wait(client, GOOD_SPEC)
    study = client.get(f"/api/studies/{job['study_id']}").json()
    case = study["cases"][0]
    from pathlib import Path

    z = np.linspace(0, 1, 5)
    meshio.Mesh(np.array([[0, 0, v] for v in z]), [("line", np.array([[i, i + 1] for i in range(4)]))],
                point_data={"phi_g000_m00": z}).write(Path(case["directory"]) / "flux.vtu")
    return job["study_id"], case["case_id"]


def test_a_studys_field_files_are_listed(client):
    study_id, case_id = _study_with_a_field(client)
    listed = client.get(f"/api/studies/{study_id}/fields").json()["cases"]
    assert [c["case_id"] for c in listed] == [case_id]
    assert listed[0]["fields"][0]["path"] == "flux.vtu"


def test_a_field_is_drawn_as_a_png(client):
    study_id, case_id = _study_with_a_field(client)
    response = client.get(f"/api/studies/{study_id}/fields/image",
                          params={"case": case_id, "file": "flux.vtu"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_a_field_request_cannot_leave_the_case_directory(client):
    study_id, case_id = _study_with_a_field(client)
    response = client.get(f"/api/studies/{study_id}/fields/image",
                          params={"case": case_id, "file": "../../../../etc/hosts"})
    assert response.status_code == 404


def test_explain_describes_the_studys_charts_and_fields(tmp_path):
    """The narration is told what the page draws, computed from the catalogue."""
    pytest.importorskip("meshio")
    llm = ScriptedLLM(["The tighter tolerance needed more sweeps.", "Group 0 only."])
    client = TestClient(create_app(llm=llm, run_root=tmp_path / "runs"))
    study_id, _ = _study_with_a_field(client)
    table = [{"case": "a", "sweeps": 6}]
    response = client.post("/api/explain", json={"table": table, "study_id": study_id})
    assert response.status_code == 200
    assert "warnings" in response.json()
    user = llm.calls[-1]["user"]
    assert 'Chart "Varying l_abs_tol"' in user
    assert "Field file flux.vtu: a 1D mesh of 4 cells, z from 0 to 1" in user
    answer = client.post("/api/chat", json={
        "table": table, "study_id": study_id,
        "messages": [{"role": "user", "content": "What does the flux picture show?"}]})
    assert answer.status_code == 200
    assert "Field file flux.vtu" in llm.calls[-1]["user"]


def test_explain_reports_invented_numbers(tmp_path):
    """The endpoint used to look for an attribute that did not exist and drop them."""
    client = TestClient(create_app(llm=ScriptedLLM(["It took 987654 sweeps."]), run_root=tmp_path / "runs"))
    response = client.post("/api/explain", json={"table": [{"case": "a", "sweeps": 6}]})
    assert any("987654" in w for w in response.json()["warnings"])


# --- editing a spec by hand -----------------------------------------------------


def test_an_edited_spec_is_checked_without_running(client):
    edited = GOOD_SPEC.replace("values: [1.0e-4, 1.0e-6]", "values: [1.0e-4, 1.0e-6, 1.0e-8]")
    assert edited != GOOD_SPEC
    body = client.post("/api/spec/check", json={"spec_yaml": edited}).json()
    assert body["ok"] and body["cases"] == 3 + 1, "three swept values and the reference"
    assert '"template": "reed_1d"' in body["spec_json"]
    assert client.get("/api/runs").json()["jobs"] == []


@pytest.mark.parametrize("broken,reason", [
    ("name: x\ntemplate: [", "could not read"),
    (OUT_OF_RANGE, "gmres_restart_interval"),
    (GOOD_SPEC.replace("template: reed_1d", "template: no_such_scenario"), "no_such_scenario"),
])
def test_an_edited_spec_that_does_not_validate_says_why(client, broken, reason):
    body = client.post("/api/spec/check", json={"spec_yaml": broken}).json()
    assert body["ok"] is False and reason in body["error"]
