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
