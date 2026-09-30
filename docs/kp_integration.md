# Integrating with the knowledge-pack assistant

AutoOpenSn depends on `Code_assistant_TAU`, an OpenSn documentation assistant,
and does not modify it. This document says how that dependency is arranged, what
it currently costs, and what would have to change upstream to make it cheaper.

## The arrangement

The sister repository has no `pyproject.toml`, so its `src.kp` package cannot be
installed. The only way to use it is to put its root on `sys.path`. That happens
in exactly one place, `autoopensn/kp_bridge.py`, and nowhere else.

This is enforced, not merely intended. `tests/test_kp_bridge.py` walks the
package with `ast` and fails if any other module imports `src` or touches
`sys.path`. A path mutation scattered across five modules is not a dependency;
it is a haunting, and the failure mode is an import that works on one developer's
machine for reasons nobody can reconstruct.

## What the bridge provides

| Function | What it wraps | Used by |
| --- | --- | --- |
| `searcher()`, `search()`, `ground()` | `src.kp.search.PackSearcher` | spec generation, to ground parameter choices in the API at the pinned commit |
| `reviewer(llm)` | `src.kp.reviewer.from_config` | checking a generated spec for domain errors before anything runs |
| `llm_client()` | `src.llm_client.LLMClient` | both narration stages |
| `checkout_path()`, `test_script()`, `tests_json_path()` | the pinned OpenSn checkout | template sources and regression gold values |
| `pack_commit()`, `pack_manifest()` | `knowledge-packs/<commit>/manifest.json` | the commit recorded in every generated script |

Where the sister repository is absent, the bridge raises `BridgeError` with a
message naming `AUTOOPENSN_KP_REPO`. `available()` returns False, and the tests
that need it skip rather than fail, so this repository stays testable alone.

## Two decisions worth stating

**The embedding model is off by default.** `searcher()` passes
`load_model=False`, so retrieval is BM25 over FTS5. The model is hundreds of
megabytes and seconds to load, which is the right trade for an interactive
documentation assistant and the wrong one for a command that wants to check
whether `gmres_restart_interval` exists. The queries this tool makes are
symbol-shaped, which is what lexical retrieval is best at. Pass
`load_model=True` for a genuinely natural-language query.

Measured on the pinned pack: `search("GMRES restart interval")` returns
`doc/source/userguide/iterative_methods.rst:358-385`, the section titled "GMRES
restart interval", as the first hit, in about a third of a second including
opening the pack.

**The pack commit is read, not assumed.** `pack_commit()` reads the manifest and
raises if it disagrees with the directory name. A pack whose manifest says one
commit while its directory says another would otherwise mislabel every generated
script in a study, and the label is the only defence that survives someone
copying a script out of `runs/`.

## What the sister repository would have to change

Nothing, for this to work. These are the changes that would make it cleaner,
recorded here rather than made there, per the brief.

1. **Add a `pyproject.toml`.** This is the whole reason `kp_bridge` exists. With
   `kp` installable, the bridge becomes a thin convenience module with no
   `sys.path` mutation, and the `ast`-walking test that enforces the constraint
   can be deleted. A minimal `setuptools` configuration exporting `src.kp` would
   do it; the package layout already supports it.

2. **Let `PackSearcher` take an opened connection or a read-only flag.** It opens
   SQLite with `check_same_thread=False` and holds the connection for its
   lifetime. Fine for a long-lived API process; awkward for a CLI that wants a short-lived
   read. Not blocking, and worked around here by caching the searcher.

3. **Separate `SMEReviewer` from `src.config`.** `reviewer.from_config` reads
   module-level constants for the model, timeout, and confidence floor. A
   consumer that wants a different confidence floor for spec review than for
   answer review has to construct `SMEReviewer` directly and duplicate the
   defaults. An optional overrides argument on `from_config` would fix it.

4. **Give `LLMClient` a quieter constructor.** It prints provider-selection
   banners to stdout on construction, which is noise in a CLI whose stdout is a
   results table. Worked around by keeping narration off the stdout path, but a
   `verbose=False` flag would be better.

5. **Fix the spelling of `TAMU_assintant_key`.** `LLMClient` reads
   `TAMU_assintant_key` before `TAMU_API_KEY`; the first is missing an `s`.
   The name exists for a real reason — an exported shell variable beats one
   set in `.env`, so a distinct name sidesteps the collision without touching
   anyone's shell configuration — but the reason argues for a correctly spelled
   distinct name, such as `TAMU_ASSISTANT_KEY`. As it stands, every consumer
   has to reproduce a typo to be understood, and anyone who spells it correctly
   gets silence rather than an error. Reading both, and preferring the correct
   spelling, would fix it without breaking existing setups.

None of these block a working pipeline.

## How the reviewer is used

Wired up in `autoopensn/narrate/spec_generation.py`. It follows the sister
repository's own routing, which is not reinvented here:

- A finding with `supported_by_evidence = False` says the spec asserts something
  the retrieved evidence does not support. A correct spec exists inside the
  evidence, so it is worth one regeneration.
- A finding with `supported_by_evidence = True` says the evidence itself is
  wrong. No rewrite can satisfy it, so it is attached as a note and never fed
  back.

That single boolean is what makes the loop terminate. AutoOpenSn adds one
deterministic check the reviewer cannot overrule, for the same reason the sister
repository added its "established facts" block: `Spec.validate_against` is an
exact comparison against the template's declared ranges, and a reviewer that
talked the generator out of a valid spec into an invalid one would undo the
guarantee the spec model exists to provide. A revision produced after review is
validated like any other, and discarded if it fails. See
[docs/llm_layer.md](llm_layer.md).

The client is also reused for both narration stages, through
`kp_bridge.llm_client(quiet=True)`. The `quiet` flag captures the provider
banners the client prints on construction, which are useful in a terminal
and are noise in a command whose stdout is a results table.

## Data flowing the other way

None. AutoOpenSn reads from the sister repository and writes nothing to it. The
run cache, the run directories, and the fixtures all live here.
