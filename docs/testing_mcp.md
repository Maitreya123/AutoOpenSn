# Testing the MCP server

Four ways to check that `autoopensn mcp` works, from quickest to most
realistic. They all start from the same setup. What the server is and how it
works is in [mcp_server.md](mcp_server.md).

## Setup (once)

In a clone of this repository, with Python 3.11 or newer:

```shell
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api,llm,mcp]"
```

To run real OpenSn on the cluster, connect the TAMU VPN first, with Cloudflare
WARP turned off (it takes over the routes the VPN needs), and set up cluster
access once with `scripts/setup_cluster.sh`
([running_opensn.md](running_opensn.md)). Without the cluster everything still
works, but only the two 1D scenarios run, in AutoOpenSn's built-in solver.

## 1. The automated tests

No cluster, no network, about 20 seconds:

```shell
pytest tests/test_mcp_server.py
```

All of them should pass (73 at the time of writing). They call every tool
through a real MCP client session, including a new server process for every
call, which is how URSA's client behaves.

## 2. Call the tools by hand: MCP Inspector

[MCP Inspector](https://github.com/modelcontextprotocol/inspector) is the
official MCP testing tool. It opens a web page where you call each tool
yourself and see exactly what comes back, which is what an agent sees. It
needs Node.js.

```shell
npx @modelcontextprotocol/inspector "$(which autoopensn)" mcp
```

In the page, click **Connect**, open **Tools**, and go through them in order:

1. `list_scenarios` with no arguments: 41 scenarios.
2. `describe_scenario` with `name` = `reed_1d`: copy `example_spec` from the
   result.
3. `validate_spec` with that spec: `ok: true` and the list of runs.
4. `run_study` with the same spec: note the `job_id`.
5. `get_results` with that `job_id`: repeat until `status` is `succeeded`.

Then break something on purpose. Set a solver parameter out of range, or put
one under the wrong section, and `validate_spec` should say exactly what to
change.

## 3. From Claude Code

```shell
claude mcp add autoopensn -- "$(which autoopensn)" mcp
```

Then ask in plain language, for example: *"Using the autoopensn tools, compare
GMRES tolerances 1e-4 and 1e-6 against a 1e-10 reference on the Reed
problem."*

## 4. From URSA

This is the case the server was built for. Use the example on the URSA fork,
[`examples/opensn`](https://github.com/Maitreya123/ursa/tree/examples-opensn/examples/opensn),
from a clone of that fork:

```shell
cd examples/opensn
uv sync
```

With an OpenAI key, follow the example's README as written. To use TAMU AI Chat
as URSA's model instead, create `tamu.yaml` in that folder:

```yaml
inference_providers:
  tamu:
    base_url: https://chat-api.tamu.ai/openai
    api_key: {env: TAMU_assintant_key}
llm_model:
  model: openai:protected.gpt-5.5
  inference_provider: tamu
  use_responses_api: false
mcp_servers:
  autoopensn:
    transport: stdio
    command: autoopensn
    args: [mcp, --runner, auto, --run-root, runs, --cache-path, runs/cache.db]
```

and start URSA through `scripts/ursa_tamu.py` from this repository, never
plain `ursa`: TAMU's gateway rejects a message format URSA sends, and the
launcher works around it ([mcp_server.md](mcp_server.md), "URSA with TAMU AI
Chat"). The first line reads the key from this repository's `.env` without
printing it:

```shell
export TAMU_assintant_key="$(grep '^TAMU_assintant_key=' /path/to/AutoOpenSn/.env | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//')"
uv run python /path/to/AutoOpenSn/scripts/ursa_tamu.py --config tamu.yaml \
  exec "#execute Which OpenSn tutorial is about a point source? Run it at its default settings and say which engine produced the result."
```

For URSA's interactive interface instead, leave off `exec "..."`.

## What to check

Whichever way you test:

- **The engine is named.** Every result's `note` says what produced the
  numbers: "Running real OpenSn on class01." or "AutoOpenSn's own 1D solver.
  Not OpenSn."
- **Gold is checked.** Where OpenSn's test suite records the answer,
  `gold_passed` is `true`.
- **Multi-rank scenarios use their ranks.** `detector` should run on 4 ranks,
  and `hdpe_example` on 8, without the spec setting `num_procs`.
- **Failures explain themselves.** Each study has a folder under
  `runs/.mcp_jobs/<job_id>/`; if one fails, `get_results` gives the reason and
  `worker.log` there has the details.
