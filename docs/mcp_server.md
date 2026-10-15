# The MCP server

`autoopensn mcp` offers the deterministic stages of the pipeline as tools over
the [Model Context Protocol](https://modelcontextprotocol.io), so another agent
can run OpenSn studies without a person in between. It was built so that LANL's
[URSA](https://github.com/lanl/ursa) can use AutoOpenSn, but any MCP client
works, including Claude Code.

## The split

This package's rule is *LLM at the edges, deterministic in the middle*. The
server keeps the middle and hands the edges to the calling agent:

```
calling agent (a language model)          autoopensn mcp (no language model)
────────────────────────────────          ──────────────────────────────────
reads the question
                     list_scenarios   ─▶  the catalogue, ranked
                     describe_scenario ─▶ parameters, ranges, a valid example
writes a spec
                     validate_spec    ─▶  ok, or exactly what to change
                     run_study        ─▶  renders verified templates, runs them
                     get_results      ─▶  progress, then the results table
explains the table
```

So the server needs no API key, no `.env`, and no sister repository, and
`tests/test_mcp_server.py` checks that a whole study never loads
`autoopensn.llm` or `autoopensn.narrate` or calls the model client in
`kp_bridge`.

What it keeps from the HTTP API: **the agent cannot choose the script.** The
only thing that becomes a script is a template plus parameters that passed
`validate_against`, and a spec the validator refuses comes back as errors the
agent can act on, never as a run.

## The tools

| Tool | Does | Notes |
| --- | --- | --- |
| `list_scenarios(query, limit)` | The catalogue, ranked by `query` | `runs_without_cluster` marks the two scenarios the 1D reference solver supports; `has_gold` marks a recorded correct answer |
| `describe_scenario(name)` | Every parameter with its group, type, default, range or choices, and meaning; the observables; `num_procs`; `example_spec`; `spec_rules` | `goes_under` says whether a parameter belongs in `solver` or `template_parameters`. Every scenario's example is tested to validate |
| `validate_spec(spec)` | Validates and lists the runs, running nothing | Errors are the validator's own messages, e.g. `l_abs_tol: 5.0 is above the declared maximum 0.01` |
| `run_study(spec, engine)` | Validates, then starts the study in the background; returns `job_id` at once | `engine: "reference"` asks for the 1D solver regardless of the server's runner |
| `get_results(job_id)` | Status, the last 20 progress lines, and when finished the full results table | `note` names the engine; `gold_passed` says whether a run matched OpenSn's recorded answer |

A spec is the same object as a YAML spec file, as JSON. Studies are capped at
200 runs (`MAX_CASES`), so one sentence cannot queue a thousand cluster jobs.

A spec that leaves out `num_procs` runs on the scenario's own MPI rank count,
the one its gold values were recorded with, rather than on `Spec`'s default of
one; `glovebox`, `glovebox_adj`, `hdpe_example` and `pincell_example` are
written for eight. An
explicit `num_procs` is kept.

## Where studies run

The runner is chosen when the server starts, not by the agent:

```shell
autoopensn mcp                     # 'auto': the cluster if it answers, else the 1D solver
autoopensn mcp --runner remote     # the cluster or nothing
autoopensn mcp --runner reference  # never leaves this machine
autoopensn mcp --host my-node      # another ssh alias for the cluster
```

`--run-root` and `--cache-path` default to `runs/` relative to wherever the
agent starts the server. Remote runs need the same setup as the CLI: the VPN
off campus and `scripts/setup_cluster.sh` done once
([running_opensn.md](running_opensn.md)). Choosing the runner, including the
SSH probe, happens inside the background job, so `run_study` never blocks on
the network; a cluster that cannot be reached fails the job with the reason.

### Jobs live on disk, not in the server

A client may start a new server process for every tool call. URSA's does:
`langchain-mcp-adapters` is stateless by default, so `run_study` and the
`get_results` after it run in two different processes. A study therefore runs
in a worker process of its own, started in a new session so it outlives the
server, and keeps its state in `<run-root>/.mcp_jobs/<job_id>/`:
`request.json`, `state.json` (status, progress, and the table when done) and
`worker.log`. Any server process can answer `get_results`, and a worker that
died without finishing is reported as failed with the end of its log
([`mcp_jobs.py`](../autoopensn/mcp_jobs.py)).

The first version kept jobs in memory, and the first time URSA drove it every
`get_results` answered "no job". `tests/test_mcp_server.py` now starts a fresh
stdio server for each call and runs a study through them.

### The rules are in the tool descriptions

URSA passes the model the tool descriptions but not the server's
`instructions`. So the spec format, with an example of a sweep against a
reference, is in `validate_spec`'s description and in `describe_scenario`'s
`spec_rules`. Without it URSA's agent learned the format from validation errors
and put the reference among the sweep values, which leaves `rel_flux_diff`
empty.

## Connecting an agent

**URSA.** In URSA's config file:

```yaml
mcp_servers:
  autoopensn:
    transport: stdio
    command: autoopensn
    args: [mcp]
```

URSA starts the server and attaches the five tools to each agent that takes
tools, such as `#execute` and `#chat`.

**Claude Code.**

```shell
claude mcp add autoopensn -- autoopensn mcp
```

In both cases `autoopensn` must be on the agent's `PATH`, or give the full path
to the executable in the virtual environment it was installed into.

### URSA with TAMU AI Chat

TAMU's gateway is OpenAI-compatible with two differences that URSA 0.17.2
trips over:

1. **No Responses API.** URSA turns on `use_responses_api` for every OpenAI
   provider, and the gateway answers 404. Turn it off in the config:

   ```yaml
   inference_providers:
     tamu:
       base_url: https://chat-api.tamu.ai/openai
       api_key: {env: TAMU_assintant_key}
   llm_model:
     model: openai:protected.gpt-5.5
     inference_provider: tamu
     use_responses_api: false
   ```

2. **No `null` message content.** `langchain-openai` sends an assistant message
   that only calls tools with `"content": null`, which OpenAI accepts and the
   gateway rejects with a 422, so every tool-using agent fails on its second
   turn. There is no setting for this, so start URSA through
   [`scripts/ursa_tamu.py`](../scripts/ursa_tamu.py), which sends `""`
   instead and then runs URSA's own command line with every argument passed on:

   ```shell
   python scripts/ursa_tamu.py --config ursa_tamu.yaml exec "#execute ..."
   ```

   The patch lives only in that process; nothing installed is edited. A
   provider that follows OpenAI's API, which is what URSA is normally pointed
   at, does not need it.

Neither is specific to AutoOpenSn: any URSA agent with tools hits the second.
Plain `ursa exec` with this config makes one successful call and then fails
with 422 on every call after it. Through `scripts/ursa_tamu.py`, the same
command ran the Reed tolerance study described under "Verified" end to end on
14 October 2026, on real OpenSn on `class01`, with the same numbers.

## Verified

On 14 October 2026, driven over stdio exactly as an agent starts it, with
`--runner remote` against Orchard's `class01`:

| Study | Runs | Time | Result |
| --- | --- | --- | --- |
| `reed_1d`, `l_abs_tol` 1e-4 against a 1e-10 reference | 2 | 9 s | relative flux difference 2.7e-6 |
| `first_1d_fixed_source`, its example spec | 1 | 6 s | matches OpenSn's recorded answer exactly |
| `first_2d_fixed_source`, its example spec (cluster only) | 1 | 6 s | matches OpenSn's recorded answer exactly |

And driven by URSA 0.17.2's `ExecutionAgent` with the five tools attached by
URSA's own `add_mcp_tools`, the model being `protected.gpt-5.5` on TAMU AI Chat,
asked in plain language to compare GMRES tolerances 1e-4 and 1e-6 against a
1e-10 reference on the Reed problem "running real OpenSn on the cluster". It
called `list_scenarios`, `describe_scenario`, `validate_spec` (valid first
time, reference in the right place), `run_study`, and `get_results` four
times, then reported:

| Case | Avg flux | Relative difference from reference | Sweeps |
| --- | --- | --- | --- |
| `l_abs_tol` 1e-4 | 0.9588199124 | 2.69e-6 | 6 |
| `l_abs_tol` 1e-6 | 0.9588224823 | 1.42e-8 | 8 |
| reference, 1e-10 | 0.9588224959 | 0 | 11 |

with "Running real OpenSn on class01" as the engine. Its figures are the
table's, plus absolute flux differences it computed from them (2.58e-6 and
1.36e-8, both correct).

Later the same day, through `ursa exec` and `scripts/ursa_tamu.py`, with
`--runner remote`, three more plain-language questions:

| Question | What the agent ran | Result on `class01` |
| --- | --- | --- |
| How does the maximum flux change as 2D angular quadrature is refined? | `first_2d_fixed_source`, `n_azimuthal` 4, 8, 16 | 0.9670, 0.9300, 0.9331; the default, 8, matches gold exactly |
| Polyethylene moderation at default settings | `hdpe_example`, on 8 ranks | maximum flux 0.0782, matches gold (relative difference 1.6e-7) |
| The CMFD tutorial's k-eigenvalue | `cmfd` | k = 1.15663057, matches gold exactly |

### From a plain install

Every run above used an editable install, which reads data from the source
tree. A plain install from GitHub into a clean Python 3.11 environment showed
that `tutorials/catalog.json` was not packaged, so `list_scenarios` returned no
scenarios; `tests/test_packaging.py` now guards against that. From the fixed
install, outside the repository, with a new server process for every call and
`--runner remote`, every check passed:

| Check | Result |
| --- | --- |
| `list_scenarios` | all 41 scenarios |
| `describe_scenario("detector")` | 4 ranks, `has_gold`, an example spec with `num_procs: 4` |
| `validate_spec` with a misplaced parameter | "'n_azimuthal' is a quadrature parameter; move it from 'solver' to 'template_parameters'" |
| `run_study` with an unknown scenario | refused; no job |
| `get_results` with an unknown job id | `ok: false` |
| `first_2d_fixed_source`, `n_azimuthal` 4 and 8 | 10 s on `class01`; the default matches gold |
| `detector` with `num_procs` left out | ran on 4 ranks in 33 s; matches gold |
| `reed_1d` with `engine: "reference"` | ran locally, noted "Not OpenSn" |

And URSA, starting the server from that install, asked which tutorial covers a
point source, found `point_source` through `list_scenarios`, ran it on
`class01`, and reported an absorption rate of 1.0, matching gold exactly.
