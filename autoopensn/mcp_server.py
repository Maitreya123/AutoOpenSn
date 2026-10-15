"""The deterministic middle, offered to other agents as MCP tools.

An agent that speaks the Model Context Protocol, such as LANL's URSA or Claude
Code, starts this server and gets five tools:

    list_scenarios     what can be studied
    describe_scenario  one scenario's parameters, their ranges, a valid example
    validate_spec      check a spec and list the runs it would make, running nothing
    run_study          start a validated study; returns a job id at once
    get_results        poll that job; the results table when it is done

**No language model runs in this process.** The calling agent is the model: it
writes the spec and explains the table. That is this package's rule, "LLM at
the edges, deterministic in the middle", split across a process boundary, and
it is why the server needs no API key and no sister repository. Nothing here
imports ``autoopensn.llm``, ``autoopensn.kp_bridge`` or ``autoopensn.narrate``,
and a test holds that line.

**The agent cannot choose the script.** As with the HTTP API, the only thing
that becomes a script is a template plus parameters that passed
``validate_against``. A spec the validator refuses comes back as a list of
errors the agent can act on, never as a run.

**A study is started and polled, and outlives the server.** Runs on the cluster
take minutes, and a client may start a fresh server process for every tool
call, as URSA's does. So a study is a directory on disk run by a worker process
of its own (``mcp_jobs``), and any server process can report on it.

**The rules travel with the tools.** URSA hands the model the tool descriptions
but not the server's ``instructions``, so everything a model needs in order to
write a spec is in the descriptions and in ``describe_scenario``'s answer, not
only in ``INSTRUCTIONS``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from autoopensn import PINNED_OPENSN_COMMIT
from autoopensn.mcp_jobs import JobStore, _jsonable, start
from autoopensn.runner.reference import supported as runs_in_reference_solver
from autoopensn.spec import Spec, SpecError
from autoopensn.templates import TemplateError, load_template
from autoopensn.tutorials import catalog

MAX_CASES = 200
"""Studies larger than this are refused. A model that writes a grid sweep over
four parameters can ask for thousands of cluster runs in one sentence; a person
who wants that can run the CLI."""

PROGRESS_TAIL = 20
"""Progress lines returned per poll. The whole log is in the job directory."""

SPEC_RULES = f"""\
A spec is a JSON object:
  name                 short study name
  template             a scenario name from list_scenarios
  template_parameters  baseline values for parameters whose group is not "solver"
  solver               baseline values for parameters whose group is "solver"
  sweep                list of {{"parameter": <name>, "values": [<v1>, <v2>, ...]}}
  sweep_mode           "one_at_a_time" (default: each parameter varied alone
                       from the baseline) or "grid" (every combination)
  reference            optional overrides defining ONE extra comparison case,
                       e.g. {{"l_abs_tol": 1e-10}}. Put the reference here, not
                       in sweep values: every case's rel_flux_diff is measured
                       against it.
  observables          what to measure; must be among the scenario's observables
  num_procs            MPI ranks per run; leave it out to use the scenario's own
Each parameter goes under the section its "goes_under" field names. Values must
be within the declared minimum/maximum or choices. At most {MAX_CASES} runs.
Example sweep against a reference:
  {{"name": "tol_study", "template": "reed_1d",
    "template_parameters": {{"emit_avg_flux": true}},
    "solver": {{"inner_linear_method": "petsc_gmres"}},
    "sweep": [{{"parameter": "l_abs_tol", "values": [1e-4, 1e-6]}}],
    "reference": {{"l_abs_tol": 1e-10}},
    "observables": ["avg_flux", "sweeps", "wall_time"]}}
"""

INSTRUCTIONS = f"""\
AutoOpenSn runs parameter studies with OpenSn, a neutron transport code, from
templates verified against OpenSn's own tutorials and tests (OpenSn commit
{PINNED_OPENSN_COMMIT[:10]}). You never write simulation scripts; you write a
spec, and the server renders and runs verified scripts from it.

Workflow:
1. list_scenarios to find a scenario that fits the question.
2. describe_scenario for its parameters, allowed ranges, and a valid example spec.
3. validate_spec until it returns ok. Fix exactly what the errors say.
4. run_study, then poll get_results with the job id until status is
   "succeeded" or "failed". Cluster runs take minutes; poll every 20-60 s.
5. Explain the results using only numbers in the returned table, and always say
   which engine produced them (the "note" field). Numbers from AutoOpenSn's own
   1D solver are not OpenSn's numbers.

{SPEC_RULES}"""


def _parse_spec(spec: dict[str, Any]) -> tuple[Optional[Spec], list[str]]:
    """A validated spec, or the reasons it is not one, worded for a model to fix."""
    try:
        parsed = Spec(**spec)
    except ValidationError as exc:
        errors = []
        for error in exc.errors():
            where = ".".join(str(part) for part in error["loc"]) or "spec"
            errors.append(f"{where}: {error['msg']}")
        return None, errors
    except TypeError as exc:
        return None, [f"spec: {exc}"]
    try:
        template = parsed.validate_against()
    except (SpecError, TemplateError) as exc:
        return None, [str(exc)]
    if "num_procs" not in spec:
        # The scenario's own rank count, the one its gold values were recorded
        # with. Spec's default of 1 would run an 8-rank tutorial on one rank.
        parsed = parsed.model_copy(update={"num_procs": template.num_procs})
    return parsed, []


def _example_spec(name: str, template: Any) -> Optional[dict[str, Any]]:
    """The smallest spec for this scenario that validates: one baseline run."""
    template_parameters: dict[str, Any] = {}
    if "avg_flux" in template.observables and "emit_avg_flux" in template.parameters:
        template_parameters["emit_avg_flux"] = True
    example = {
        "name": f"{name}_study",
        "template": name,
        "description": "One run at the scenario's defaults.",
        "template_parameters": template_parameters,
        "solver": {},
        "sweep": [],
        "observables": list(template.observables),
        "num_procs": template.num_procs,
    }
    parsed, errors = _parse_spec(example)
    return example if parsed is not None and not errors else None


def build_server(
    *,
    runner: str = "auto",
    host: Optional[str] = None,
    fixtures: Optional[str] = None,
    run_root: Path = Path("runs"),
    cache_path: Optional[Path] = Path("runs/cache.db"),
    detach: bool = True,
) -> FastMCP:
    """The MCP server, configured with where and how studies run.

    ``runner`` is the operator's choice, made when the server starts: 'auto'
    uses the cluster when it answers and the 1D reference solver otherwise. The
    calling agent may ask for 'reference' to get a quick answer, but it cannot
    point the server at a different host. ``fixtures``, the 'fake' runner and
    ``detach=False``, which runs studies on a thread of this process, exist for
    the test suite.
    """
    run_root = Path(run_root).resolve()
    cache = Path(cache_path).resolve() if cache_path else None
    jobs = JobStore(run_root)
    # Warnings only: the default logs a line per request to stderr, which an
    # agent's terminal shows the person as noise.
    server = FastMCP("autoopensn", instructions=INSTRUCTIONS, log_level="WARNING")

    @server.tool()
    def list_scenarios(query: str = "", limit: int = 50) -> dict[str, Any]:
        """List the scenarios (OpenSn problems) that studies can be run on.

        Pass a few words in `query` to rank scenarios by relevance, e.g.
        "GMRES tolerance 1D" or "pincell eigenvalue". `has_gold` means OpenSn's
        test suite records the correct answer, so results can be checked against
        it. `runs_without_cluster` means AutoOpenSn's own 1D solver can also run
        it; every other scenario needs real OpenSn on the cluster.

        Next: describe_scenario on the one that fits.
        """
        available = catalog.load()
        chosen = (
            catalog.rank(query, available, limit=limit) if query.strip() else available[:limit]
        )
        return {
            "total": len(available),
            "scenarios": [
                {
                    "name": scenario.name,
                    "title": scenario.title or scenario.name,
                    "summary": scenario.summary,
                    "section": scenario.section,
                    "has_gold": bool(scenario.has_gold),
                    "sweepable": bool(scenario.sweepable),
                    "runs_without_cluster": runs_in_reference_solver(scenario.name),
                }
                for scenario in chosen
            ],
        }

    @server.tool()
    def describe_scenario(name: str) -> dict[str, Any]:
        """Describe one scenario: every parameter a spec may set, with its group,
        type, default, allowed range or choices, and meaning; the observables a
        study can measure; an example spec that already validates; and
        `spec_rules`, the format of a spec.

        Start every new spec from `example_spec` and follow `spec_rules`.
        """
        found = catalog.by_name(name, catalog.load())
        try:
            template = load_template(name)
        except TemplateError as exc:
            return {"ok": False, "error": str(exc)}
        parameters = [
            {
                "name": declaration.name,
                "group": declaration.group,
                "goes_under": "solver" if declaration.group == "solver" else "template_parameters",
                "type": declaration.type,
                "default": _jsonable(declaration.default),
                "minimum": declaration.minimum,
                "maximum": declaration.maximum,
                "choices": list(declaration.choices) if declaration.choices else None,
                "description": declaration.description,
            }
            for declaration in template.parameters.values()
        ]
        return {
            "ok": True,
            "name": template.name,
            "title": (found.title if found else "") or template.name,
            "description": template.description.strip(),
            "derived_from": template.source,
            "opensn_commit": template.opensn_commit,
            "observables": list(template.observables),
            "has_gold": bool(found.has_gold) if found else False,
            "runs_without_cluster": runs_in_reference_solver(template.name),
            "num_procs": template.num_procs,
            "parameters": parameters,
            "example_spec": _example_spec(template.name, template),
            "spec_rules": SPEC_RULES,
        }

    @server.tool(description=(
        "Check a study spec against its scenario and list the runs it would make. "
        "Runs nothing. If `ok` is false, `errors` says exactly what to change.\n\n"
        + SPEC_RULES
    ))
    def validate_spec(spec: dict[str, Any]) -> dict[str, Any]:
        parsed, errors = _parse_spec(spec)
        if parsed is None:
            return {"ok": False, "errors": errors}
        points = parsed.expand()
        result: dict[str, Any] = {
            "ok": len(points) <= MAX_CASES,
            "errors": [],
            "case_count": len(points),
            "num_procs": parsed.num_procs,
            "cases": [
                {
                    "case_id": point.case_id,
                    "label": point.label(),
                    "varied": {k: _jsonable(v) for k, v in point.varied.items()},
                    "is_reference": point.is_reference,
                }
                for point in points[:MAX_CASES]
            ],
        }
        if len(points) > MAX_CASES:
            result["errors"] = [
                f"this spec makes {len(points)} runs; the limit is {MAX_CASES}. "
                "Use fewer values, or sweep_mode one_at_a_time instead of grid."
            ]
        return result

    @server.tool()
    def run_study(
        spec: dict[str, Any],
        engine: Literal["default", "reference"] = "default",
    ) -> dict[str, Any]:
        """Start a validated study in the background and return its job id.

        Validates first, exactly as validate_spec does; an invalid spec starts
        nothing. `engine` "default" uses the server's configured runner (real
        OpenSn on the cluster when it is reachable). "reference" uses
        AutoOpenSn's own 1D solver, which is fast but only supports scenarios
        marked runs_without_cluster, and whose numbers are not OpenSn's.

        Next: poll get_results with the returned job_id until its status is
        "succeeded" or "failed". Cluster runs take minutes.
        """
        parsed, errors = _parse_spec(spec)
        if parsed is None:
            return {"ok": False, "errors": errors}
        points = len(parsed.expand())
        if points > MAX_CASES:
            return {
                "ok": False,
                "errors": [f"this spec makes {points} runs; the limit is {MAX_CASES}."],
            }
        job_id = jobs.create(
            {
                "spec": parsed.model_dump(mode="json"),
                "runner": "reference" if engine == "reference" else runner,
                "host": host,
                "fixtures": fixtures,
                "run_root": str(run_root),
                "cache_path": str(cache) if cache else None,
            },
            total=points,
            label=f"{parsed.name} ({parsed.template})",
        )
        start(jobs.root / job_id, detach=detach)
        return {"ok": True, "job_id": job_id, "status": "queued", "cases": points}

    @server.tool()
    def get_results(job_id: str) -> dict[str, Any]:
        """Report a study's progress, and its results table once it has finished.

        `status` is queued, running, succeeded or failed. When succeeded,
        `table` has one row per run: `case` names it, `ok` says whether it ran,
        the scenario's observables follow, `rel_flux_diff` compares each case
        with the reference case, and `gold_passed` says whether the run matched
        OpenSn's recorded answer. `note` says which engine produced the numbers;
        always report it.
        """
        state = jobs.read(job_id)
        if state is None:
            return {"ok": False, "error": f"no job {job_id!r}"}
        state["progress"] = state.get("progress", [])[-PROGRESS_TAIL:]
        state.pop("traceback", None)
        state.pop("pid", None)
        state["ok"] = True
        return state

    return server


def main(**options: Any) -> None:
    """Serve over stdio, which is how an agent launches a local MCP server."""
    build_server(**options).run()


if __name__ == "__main__":
    main()
