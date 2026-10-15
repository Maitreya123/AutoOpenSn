#!/usr/bin/env python3
"""Run LANL's URSA with TAMU AI Chat as its model.

Use it exactly as you would the ``ursa`` command; every argument is passed on:

    python scripts/ursa_tamu.py --config ursa_tamu.yaml exec "#execute ..."

Why it exists. TAMU's gateway speaks OpenAI's chat API with one difference that
stops every URSA agent that uses tools: when the model's reply is only a tool
call, ``langchain-openai`` sends that message back on the next turn with
``"content": null``. OpenAI accepts null there; the gateway answers 422 and the
agent fails on its second turn. There is no setting for it in URSA or in
``langchain-openai``, so this script sends ``""`` instead, which means the same
thing, and then starts URSA's own command line unchanged.

The patch lives only in this process. Nothing installed is edited. It is a
workaround: the fix belongs in ``langchain-openai`` or the gateway, and against
any provider that follows OpenAI's API this script and ``ursa`` behave the same.

The config also needs ``use_responses_api: false`` under ``llm_model``, because
URSA turns on OpenAI's Responses API by default and the gateway does not serve
it. See docs/mcp_server.md, "URSA with TAMU AI Chat".
"""

from __future__ import annotations

import sys


def patch_null_tool_call_content() -> None:
    """Send "" rather than null as the content of a tool-call-only message."""
    import langchain_openai.chat_models.base as base

    original = base._convert_message_to_dict
    if getattr(original, "_tamu_patched", False):
        return

    def convert(message, *args, **kwargs):
        converted = original(message, *args, **kwargs)
        if converted.get("role") == "assistant" and converted.get("content") is None:
            converted["content"] = ""
        return converted

    convert._tamu_patched = True
    base._convert_message_to_dict = convert


def main() -> int:
    try:
        from ursa.cli import main as ursa_main
    except ImportError:
        sys.stderr.write(
            "URSA is not installed in this Python environment.\n"
            "Install it with: pip install ursa-ai\n"
        )
        return 1
    patch_null_tool_call_content()
    sys.argv[0] = "ursa"
    return ursa_main() or 0


if __name__ == "__main__":
    sys.exit(main())
