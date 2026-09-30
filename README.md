# AutoOpenSn

A workflow engine for parameter studies with the [OpenSn](https://github.com/Open-Sn/opensn)
neutron transport code.

You describe a study in plain language. AutoOpenSn turns that into a structured
scenario spec, generates PyOpenSn input scripts from templates, runs them,
parses the results into a table, and only then asks a language model to explain
what the numbers show.

## The principle

**LLM at the edges, deterministic in the middle.**

A language model translates the request into a spec, and narrates the results
table once the numbers exist. It never writes free-form simulation scripts, and
it never reads raw logs to decide what happened. Everything between the spec and
the results table is plain, testable Python: template rendering is a pure
function, running is a process invocation, and parsing is a set of regular
expressions with named observables.

This matters because the failure mode of an LLM-driven simulation tool is a
plausible-looking script that silently computes the wrong problem. Every template
here is derived from an OpenSn tutorial or regression script and reproduces it
character for character at default parameters, so a generated script differs from
validated upstream code by exactly the values the spec changed and nothing else.
Every generated script records the OpenSn commit it was generated against.

## Installation

```shell
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,api]"
cp .env.example .env    # only needed for the narration stages

cd web && npm install   # the interface
```

## Usage

### The interface

Two processes: the engine, and the page.

```shell
autoopensn serve        # the API, on :8000
cd web && npm run dev   # the interface, on :5173
```

Then open <http://localhost:5173>. Four steps, in order, with nothing to
configure:

1. **Describe the study** in plain language.
2. **Read the spec** it wrote, and download it as YAML or JSON.
3. **Run it**, watching each case complete.
4. **Read what the numbers show**, and ask follow-up questions.

The engine's options — which runner, whether to use the cache, where the
fixtures are — are not on the screen. They are in the API and the CLI, where
scripts and tests reach them. The interface picks the best runner available: the
cluster when it can be reached, this package's own solver when it cannot.

What it never hides is **which engine produced the numbers**, stated in words
above the table. A results table that does not say where it came from is a table
nobody should quote. It also flags any figure in its own prose that is not in
the table, and says when a value was replayed from a fixture rather than
measured, or derived rather than logged.

The built-in solver handles one-dimensional problems and **is not OpenSn** — it
is an independent implementation that reproduces OpenSn's own regression values
for the scenarios it supports ([docs/reference_solver.md](docs/reference_solver.md)).
Everything else needs real OpenSn, which is cheapest on a cluster where it is
already built ([docs/running_opensn.md](docs/running_opensn.md)).

### The command line

Every stage also runs on its own, which is what scripts and tests use, and what
lets you inspect and edit a spec before anything executes.

```shell
autoopensn scenarios "gmres restart"                           # browse the library
autoopensn spec "compare GMRES tolerances ..." -o study.yaml   # prompt -> spec (LLM)
autoopensn render study.yaml -o runs/render                    # spec -> scripts
autoopensn run study.yaml                                      # spec -> results table
autoopensn explain results.csv study.yaml                      # table -> narrative (LLM)
autoopensn study "compare GMRES tolerances ..."                # all of the above
autoopensn serve                                               # the same, over HTTP
```

`spec` and `explain` are the two stages that call a language model. Everything
between them is deterministic.

## The scenario library

The scenarios are the OpenSn tutorials themselves. Every notebook under
`doc/source/tutorials` at the pinned commit is read into a template with a
parameter sidecar, with the knobs found in the tutorial's own script and the
ranges taken from OpenSn's own constraints where they exist.

| | |
| --- | --- |
| Tutorials catalogued | 40 |
| With a recorded correct answer from OpenSn's test suite | 39 |
| Parameters found across them | 930 |

Rendering a generated template at its defaults reproduces the tutorial's script
character for character, which the test suite checks for all of them on every
run. See [docs/scenarios.md](docs/scenarios.md).

## The sister repository

`Code_assistant_TAU` is an OpenSn documentation assistant. AutoOpenSn uses it as
a library and as a data source, and never modifies it:

- its knowledge pack grounds spec generation in the real API for a pinned commit,
- its `SMEReviewer` checks generated specs for domain errors before anything runs,
- its OpenSn checkout supplies the template sources and the regression gold values.

All of that access goes through the single adapter module
`autoopensn/kp_bridge.py`. Nothing else in this codebase imports from the sister
repository. See [docs/kp_integration.md](docs/kp_integration.md).

The pinned OpenSn commit is `2fd4a19ceade4f8da581a5029c68f474d3333ccb`.

## Testing

```shell
pytest
```

No OpenSn, no MPI, and no network access is required.
