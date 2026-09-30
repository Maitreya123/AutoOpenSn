"""AutoOpenSn: a workflow engine for OpenSn parameter studies.

The pipeline has five stages and one rule: LLM at the edges, deterministic in
the middle.

    prompt --[LLM]--> spec --> rendered scripts --> runs --> table --[LLM]--> narrative
            narrate/         templates/            runner/   parse/    narrate/

Only the first and last arrows involve a language model. Everything between
them is a pure function, a subprocess invocation, or a regular expression.
"""

__version__ = "0.1.0"

PINNED_OPENSN_COMMIT = "2fd4a19ceade4f8da581a5029c68f474d3333ccb"
"""The OpenSn commit every template and gold value in this repository is tied to.

This is the commit recorded in the sister repository's knowledge pack manifest.
A template rendered against a different commit is not known to be valid, because
the PyOpenSn API has changed across versions.
"""
