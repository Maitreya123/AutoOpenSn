"""The MCP server, driven through a real MCP client session.

Every test here talks to the server the way URSA would: tools are listed and
called over the protocol, and results are read from what comes back. None of
them reaches the cluster; ``conftest.no_cluster`` still applies, and the runs
that do happen are fixture replays or the 1D reference solver.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
import yaml

pytest.importorskip("mcp", reason="the MCP SDK is not installed (pip install -e '.[mcp]')")

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402
from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from autoopensn.mcp_jobs import JobStore  # noqa: E402
from autoopensn.mcp_server import MAX_CASES, SPEC_RULES, build_server  # noqa: E402
from autoopensn.tutorials import catalog  # noqa: E402

TESTS = Path(__file__).resolve().parent
GMRES_SPEC = yaml.safe_load((TESTS / "data" / "gmres_convergence.yaml").read_text())


def call(server, tool: str, arguments: dict | None = None) -> dict:
    """Call one tool over an in-memory client session and return its result."""

    async def go() -> dict:
        async with create_connected_server_and_client_session(server) as session:
            result = await session.call_tool(tool, arguments or {})
            assert not result.isError, result.content
            return result.structuredContent

    return asyncio.run(go())


def wait_for(server, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = call(server, "get_results", {"job_id": job_id})
        if record["status"] in ("succeeded", "failed"):
            return record
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


@pytest.fixture
def server(tmp_path):
    return build_server(
        runner="fake",
        fixtures=str(TESTS / "fixtures" / "reed_gmres"),
        run_root=tmp_path / "runs",
        cache_path=tmp_path / "runs" / "cache.db",
        detach=False,
    )


# --- the surface ---


def test_the_five_tools_are_offered(server):
    async def go():
        async with create_connected_server_and_client_session(server) as session:
            listed = await session.list_tools()
            return {tool.name for tool in listed.tools}

    assert asyncio.run(go()) == {
        "list_scenarios",
        "describe_scenario",
        "validate_spec",
        "run_study",
        "get_results",
    }


def test_the_server_tells_the_agent_the_rules(server):
    # Sent to the client in the initialize result, before any tool is called.
    assert "validate_spec" in server.instructions
    assert "not OpenSn's numbers" in server.instructions


# --- reading the library ---


def test_every_scenario_is_listed(server):
    listed = call(server, "list_scenarios", {"limit": 100})
    assert listed["total"] == len(catalog.load())
    assert len(listed["scenarios"]) == listed["total"]


def test_the_listing_says_what_runs_without_the_cluster(server):
    listed = {s["name"]: s for s in call(server, "list_scenarios", {"limit": 100})["scenarios"]}
    assert listed["reed_1d"]["runs_without_cluster"] is True
    assert sum(s["runs_without_cluster"] for s in listed.values()) == 2


def test_a_query_ranks_the_listing(server):
    ranked = call(server, "list_scenarios", {"query": "reed slab", "limit": 3})
    assert len(ranked["scenarios"]) == 3
    assert ranked["total"] > 3


def test_a_description_says_where_each_parameter_goes(server):
    described = call(server, "describe_scenario", {"name": "reed_1d"})
    by_name = {p["name"]: p for p in described["parameters"]}
    assert by_name["l_abs_tol"]["goes_under"] == "solver"
    assert by_name["l_abs_tol"]["maximum"] is not None
    assert "avg_flux" in described["observables"]


@pytest.mark.parametrize("name", [s.name for s in catalog.load()])
def test_every_example_spec_validates(server, name):
    """The example is what an agent copies first, so it must always be valid."""
    described = call(server, "describe_scenario", {"name": name})
    assert described["ok"]
    assert described["example_spec"] is not None, f"{name} has no valid example"
    checked = call(server, "validate_spec", {"spec": described["example_spec"]})
    assert checked["ok"], checked["errors"]
    assert checked["case_count"] == 1


def test_an_unknown_scenario_is_an_answer_not_a_crash(server):
    described = call(server, "describe_scenario", {"name": "no_such_problem"})
    assert described["ok"] is False
    assert "no_such_problem" in described["error"]


# --- validation ---


def test_a_valid_spec_lists_its_runs(server):
    checked = call(server, "validate_spec", {"spec": GMRES_SPEC})
    assert checked["ok"]
    assert checked["case_count"] == 7
    assert [c for c in checked["cases"] if c["is_reference"]][0]["case_id"] == "reference"


def test_an_out_of_range_value_says_the_limit(server):
    spec = {**GMRES_SPEC, "solver": {**GMRES_SPEC["solver"], "l_abs_tol": 5.0}}
    checked = call(server, "validate_spec", {"spec": spec})
    assert checked["ok"] is False
    assert "maximum" in checked["errors"][0]


def test_a_parameter_in_the_wrong_section_says_where_it_belongs(server):
    spec = {
        "name": "misplaced",
        "template": "reed_1d",
        "template_parameters": {"emit_avg_flux": True, "l_abs_tol": 1e-6},
    }
    checked = call(server, "validate_spec", {"spec": spec})
    assert checked["ok"] is False
    assert "move it from 'template_parameters' to 'solver'" in checked["errors"][0]


def test_a_malformed_spec_names_the_field(server):
    checked = call(server, "validate_spec", {"spec": {"template": "reed_1d", "colour": "red"}})
    assert checked["ok"] is False
    joined = " ".join(checked["errors"])
    assert "name" in joined and "colour" in joined


def test_an_oversized_study_is_refused(server):
    values = [round(1e-3 * (i + 1), 6) for i in range(15)]
    spec = {
        "name": "too_big",
        "template": "reed_1d",
        "template_parameters": {"emit_avg_flux": True},
        "solver": {"inner_linear_method": "petsc_gmres"},
        "sweep_mode": "grid",
        "sweep": [
            {"parameter": "l_abs_tol", "values": [v / 10 for v in values[:10]]},
            {"parameter": "gmres_restart_interval", "values": list(range(5, 26))},
        ],
    }
    checked = call(server, "validate_spec", {"spec": spec})
    assert checked["case_count"] > MAX_CASES
    assert checked["ok"] is False
    assert call(server, "run_study", {"spec": spec})["ok"] is False


# --- running ---


def test_a_study_runs_in_the_background_and_reports_a_table(server):
    started = call(server, "run_study", {"spec": GMRES_SPEC})
    assert started["ok"] and started["cases"] == 7

    finished = wait_for(server, started["job_id"])
    assert finished["status"] == "succeeded", finished.get("error")
    assert finished["note"] == "Replaying recorded output. Nothing was executed."
    assert len(finished["table"]) == 7
    reference = [row for row in finished["table"] if row["case"] == "reference"][0]
    assert reference["rel_flux_diff"] == 0.0
    json.dumps(finished)  # every cell is JSON: no NaN, no numpy


def test_a_second_identical_study_comes_from_the_cache(server):
    first = wait_for(server, call(server, "run_study", {"spec": GMRES_SPEC})["job_id"])
    second = wait_for(server, call(server, "run_study", {"spec": GMRES_SPEC})["job_id"])
    assert first["cache_hits"] == []
    assert len(second["cache_hits"]) == 7


def test_an_invalid_spec_starts_nothing(server):
    spec = {**GMRES_SPEC, "template": "no_such_problem"}
    started = call(server, "run_study", {"spec": spec})
    assert started["ok"] is False
    assert "job_id" not in started


def test_the_reference_engine_can_be_asked_for(tmp_path):
    # A server configured for the cluster still answers a quick 1D question
    # locally when asked. conftest makes the cluster unreachable regardless.
    server = build_server(runner="remote", run_root=tmp_path, cache_path=None, detach=False)
    spec = {
        "name": "quick",
        "template": "reed_1d",
        "template_parameters": {"emit_avg_flux": True},
        "observables": ["avg_flux", "sweeps"],
    }
    finished = wait_for(server, call(server, "run_study", {"spec": spec, "engine": "reference"})["job_id"])
    assert finished["status"] == "succeeded", finished.get("error")
    assert "Not OpenSn" in finished["note"]
    assert finished["table"][0]["ok"] is True


def test_an_unreachable_cluster_fails_the_job_with_a_reason(tmp_path):
    server = build_server(runner="remote", run_root=tmp_path, cache_path=None, detach=False)
    started = call(server, "run_study", {"spec": GMRES_SPEC})
    finished = wait_for(server, started["job_id"])
    assert finished["status"] == "failed"
    assert finished["error"]


def test_an_unknown_job_is_an_answer_not_a_crash(server):
    assert call(server, "get_results", {"job_id": "0123456789ab"})["ok"] is False


@pytest.mark.parametrize("job_id", ["../../etc/passwd", "", "ABCDEF012345", "a" * 40])
def test_a_job_id_never_becomes_a_path(server, job_id):
    assert call(server, "get_results", {"job_id": job_id})["ok"] is False


def test_the_spec_rules_example_is_a_valid_spec(server):
    """The example in every tool description is what models copy."""
    start = SPEC_RULES.index("{\"name\"")
    example = json.loads(SPEC_RULES[start:].replace("\n", " "))
    checked = call(server, "validate_spec", {"spec": example})
    assert checked["ok"], checked["errors"]
    assert any(case["is_reference"] for case in checked["cases"])


def test_the_rules_are_in_the_tool_descriptions(server):
    """URSA gives the model tool descriptions, not the server's instructions."""

    async def go():
        async with create_connected_server_and_client_session(server) as session:
            return {t.name: t.description for t in (await session.list_tools()).tools}

    descriptions = asyncio.run(go())
    assert "reference" in descriptions["validate_spec"]
    assert "goes_under" in descriptions["validate_spec"]
    assert "spec_rules" in descriptions["describe_scenario"]


# --- jobs outlive the server process ---


def test_a_job_started_by_one_server_is_reported_by_another(tmp_path):
    """URSA's client starts a new server process for every tool call."""
    options = dict(
        runner="fake",
        fixtures=str(TESTS / "fixtures" / "reed_gmres"),
        run_root=tmp_path,
        cache_path=None,
        detach=False,
    )
    started = call(build_server(**options), "run_study", {"spec": GMRES_SPEC})
    finished = wait_for(build_server(**options), started["job_id"])
    assert finished["status"] == "succeeded", finished.get("error")
    assert len(finished["table"]) == 7


def test_a_detached_study_runs_in_its_own_process(tmp_path):
    server = build_server(
        runner="fake",
        fixtures=str(TESTS / "fixtures" / "reed_gmres"),
        run_root=tmp_path,
        cache_path=tmp_path / "cache.db",
    )
    started = call(server, "run_study", {"spec": GMRES_SPEC})
    finished = wait_for(server, started["job_id"], timeout=60)
    assert finished["status"] == "succeeded", finished.get("error")
    assert len(finished["table"]) == 7
    pid = int((JobStore(tmp_path).root / started["job_id"] / "pid").read_text())
    assert pid != os.getpid()


def test_a_worker_that_died_is_reported_as_failed(tmp_path):
    store = JobStore(tmp_path)
    job_id = store.create({"spec": {}}, total=1, label="doomed")
    gone = subprocess.Popen([sys.executable, "-c", "print('worker started'); raise SystemExit(3)"],
                            stdout=open(store.root / job_id / "worker.log", "w"))
    gone.wait()
    (store.root / job_id / "pid").write_text(str(gone.pid))
    state = store.read(job_id)
    assert state["status"] == "failed"
    assert "worker started" in state["error"]


# --- the boundary ---


def test_no_language_model_code_is_loaded():
    """The server promises no model runs in it. Hold it to that, in a clean
    interpreter, through a whole study: import, build, validate, run, poll."""
    script = textwrap.dedent(
        f"""
        import asyncio, sys, tempfile, time
        from autoopensn import kp_bridge
        for name in ("llm_client", "load_own_env", "llm_models", "reviewer"):
            setattr(kp_bridge, name, lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("the MCP server called kp_bridge." + name)))
        from mcp.shared.memory import create_connected_server_and_client_session
        from autoopensn.mcp_server import build_server
        server = build_server(runner="fake", fixtures={str(TESTS / "fixtures" / "reed_gmres")!r},
                              run_root=tempfile.mkdtemp(), cache_path=None, detach=False)
        spec = {GMRES_SPEC!r}
        async def go():
            async with create_connected_server_and_client_session(server) as s:
                await s.call_tool("list_scenarios", {{}})
                await s.call_tool("describe_scenario", {{"name": "reed_1d"}})
                await s.call_tool("validate_spec", {{"spec": spec}})
                job = (await s.call_tool("run_study", {{"spec": spec}})).structuredContent
                while True:
                    r = (await s.call_tool("get_results", {{"job_id": job["job_id"]}})).structuredContent
                    if r["status"] in ("succeeded", "failed"):
                        return r
                    time.sleep(0.05)
        result = asyncio.run(go())
        assert result["status"] == "succeeded", result
        loaded = [m for m in ("autoopensn.llm", "autoopensn.narrate", "fastapi") if m in sys.modules]
        print(",".join(loaded))
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "", f"loaded: {completed.stdout.strip()}"


def test_the_command_serves_over_stdio(tmp_path):
    """End to end the way URSA starts it: a subprocess speaking MCP on stdio."""

    async def go():
        params = StdioServerParameters(
            command=sys.executable,
            args=[
                "-m", "autoopensn.cli", "mcp",
                "--runner", "reference",
                "--run-root", str(tmp_path / "runs"),
                "--cache-path", str(tmp_path / "runs" / "cache.db"),
            ],
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                checked = await session.call_tool("validate_spec", {"spec": GMRES_SPEC})
                return {t.name for t in tools.tools}, checked.structuredContent

    names, checked = asyncio.run(go())
    assert "run_study" in names
    assert checked["ok"] and checked["case_count"] == 7


def test_a_study_survives_a_new_process_per_call_over_stdio(tmp_path):
    """Exactly what URSA does: every tool call is its own server process."""
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "autoopensn.cli", "mcp", "--runner", "reference",
              "--run-root", str(tmp_path / "runs"), "--cache-path", str(tmp_path / "runs" / "c.db")],
    )
    spec = {
        "name": "stateless",
        "template": "reed_1d",
        "template_parameters": {"emit_avg_flux": True},
        "solver": {"inner_linear_method": "petsc_gmres"},
        "sweep": [{"parameter": "l_abs_tol", "values": [1e-4]}],
        "reference": {"l_abs_tol": 1e-10},
        "observables": ["avg_flux", "sweeps"],
    }

    async def one_call(tool, arguments):
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return (await session.call_tool(tool, arguments)).structuredContent

    job_id = asyncio.run(one_call("run_study", {"spec": spec}))["job_id"]
    deadline = time.monotonic() + 60
    while True:
        state = asyncio.run(one_call("get_results", {"job_id": job_id}))
        if state["status"] in ("succeeded", "failed") or time.monotonic() > deadline:
            break
        time.sleep(0.5)
    assert state["status"] == "succeeded", state.get("error")
    assert [row["case"] for row in state["table"]] == ["l_abs_tol=1.0e-4", "reference"]
