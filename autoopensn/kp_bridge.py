"""The single adapter between AutoOpenSn and the sister repository.

``Code_assistant_TAU`` is an OpenSn documentation assistant. It has no
``pyproject.toml``, so its ``src.kp`` package is not pip-installable, and the
only way to use it today is to put its root on ``sys.path``. That is done here,
in one clearly marked place, and nowhere else.

**Nothing else in this codebase may import from the sister repository.** Every
other module asks this one. The reason is replaceability: when ``kp`` becomes
installable, or is forked, or is swapped for something else, the blast radius is
this file. A ``sys.path`` mutation scattered across five modules is not a
dependency, it is a haunting.

The bridge exposes three things:

``searcher()``
    A ``PackSearcher`` over the version-pinned knowledge pack, used to ground
    spec generation in the API that actually exists at the pinned commit.

``reviewer(llm)``
    An ``SMEReviewer``, used to check a generated spec for domain errors before
    anything runs.

``checkout_path()``
    The pinned OpenSn checkout, which is where template sources and regression
    gold values come from.

Everything here degrades to a clear exception rather than a partial result. A
missing sister repository is a configuration problem with one obvious fix, and
the message says so.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from autoopensn import PINNED_OPENSN_COMMIT

# Default location of the sister repository: a sibling of this one. Overridable
# with AUTOOPENSN_KP_REPO, because a checkout is not always a sibling.
_DEFAULT_REPO = Path(__file__).resolve().parent.parent.parent / "Code_assistant_TAU"


class BridgeError(RuntimeError):
    """The sister repository is missing, incomplete, or at an unexpected commit."""


def repo_path() -> Path:
    """Root of the sister repository."""
    override = os.environ.get("AUTOOPENSN_KP_REPO")
    return Path(override).expanduser().resolve() if override else _DEFAULT_REPO


def available() -> bool:
    """True when the sister repository looks usable.

    Tests that need the real pack skip themselves on False rather than fail, so
    that this repository stays testable on a machine that has only checked out
    AutoOpenSn.
    """
    root = repo_path()
    return (root / "src" / "kp" / "search.py").exists()


def require() -> Path:
    """Root of the sister repository, or a BridgeError explaining what to do."""
    root = repo_path()
    if not (root / "src" / "kp" / "search.py").exists():
        raise BridgeError(
            f"sister repository not found at {root}. AutoOpenSn reads the knowledge "
            f"pack and the pinned OpenSn checkout from it. Set AUTOOPENSN_KP_REPO to "
            f"the path of your Code_assistant_TAU checkout."
        )
    return root


def _ensure_importable() -> Path:
    """Put the sister repository on sys.path.

    This is the whole reason the module exists. ``src.kp`` imports itself as
    ``src.kp.*``, so its *root* goes on the path, not its ``src`` directory.
    """
    root = require()
    entry = str(root)
    if entry not in sys.path:
        # Appended, not inserted: a package named ``src`` on the front of the
        # path would shadow anything similarly named in the caller's project.
        sys.path.append(entry)
    return root


# --- knowledge pack ---------------------------------------------------------


def pack_dir(commit: str = PINNED_OPENSN_COMMIT) -> Path:
    """Directory of the knowledge pack for one commit."""
    path = require() / "knowledge-packs" / commit
    if not path.is_dir():
        available_packs = sorted(
            p.name for p in (require() / "knowledge-packs").glob("*") if p.is_dir()
        )
        raise BridgeError(
            f"no knowledge pack for commit {commit} at {path}. "
            f"Packs present: {', '.join(available_packs) or 'none'}."
        )
    return path


def pack_manifest(commit: str = PINNED_OPENSN_COMMIT) -> dict[str, Any]:
    """The pack's manifest, which records the OpenSn commit it was built from."""
    manifest_path = pack_dir(commit) / "manifest.json"
    if not manifest_path.exists():
        raise BridgeError(f"pack at {manifest_path.parent} has no manifest.json")
    return json.loads(manifest_path.read_text())


def pack_commit(commit: str = PINNED_OPENSN_COMMIT) -> str:
    """The OpenSn commit the pack was built from, read from the manifest.

    Generated scripts record this value. It is read rather than assumed so that
    a pack whose manifest disagrees with its directory name is caught here
    instead of silently mislabelling every script in a study.
    """
    manifest = pack_manifest(commit)
    recorded = manifest.get("commit")
    if not recorded:
        raise BridgeError(f"pack manifest for {commit} records no commit")
    if recorded != commit:
        raise BridgeError(
            f"pack directory {commit} contains a manifest for commit {recorded}"
        )
    return recorded


@functools.lru_cache(maxsize=4)
def searcher(commit: str = PINNED_OPENSN_COMMIT, load_model: bool = False):
    """A ``PackSearcher`` over the pinned pack.

    ``load_model=False`` by default. The embedding model is a few hundred
    megabytes and takes seconds to load, which is the right trade for a live
    documentation assistant and the wrong one for a CLI that wants to check a
    handful of parameter names. With it off, retrieval is BM25 over FTS5, which
    is more than adequate for the symbol-shaped queries this tool makes
    ("gmres_restart_interval", "inner_linear_method"). Pass ``load_model=True``
    for a genuinely natural-language query.

    Cached because opening the pack costs a SQLite connection and, when the
    model is loaded, much more.
    """
    _ensure_importable()
    from src.kp.search import PackSearcher  # noqa: PLC0415  (deliberately late)

    return PackSearcher(pack_dir(commit), load_model=load_model)


def search(query: str, k: int = 8, commit: str = PINNED_OPENSN_COMMIT, **kwargs):
    """Convenience wrapper: hits for ``query`` against the pinned pack."""
    return searcher(commit).search(query, k=k, **kwargs)


# --- domain review ----------------------------------------------------------


def reviewer(llm, enabled: Optional[bool] = None):
    """An ``SMEReviewer`` built from the sister repository's configuration.

    Returns ``None`` when review is disabled there, matching that module's own
    contract, so callers test for ``None`` and skip the stage rather than
    branching on a flag of their own.
    """
    _ensure_importable()
    from src.kp import reviewer as kp_reviewer  # noqa: PLC0415

    return kp_reviewer.from_config(llm, enabled=enabled)


def llm_client(quiet: bool = True):
    """The sister repository's ``LLMClient``.

    Reused rather than rewritten: it already handles the TAMU gateway, the Groq
    fallback, and the fact that the TAMU client silently discards ``temperature``
    and ``max_tokens``. Keys come from ``.env``; see ``.env.example``.

    ``quiet`` captures the provider-selection banners the client prints to
    stdout on construction. They are useful in a Streamlit app and are noise in
    a command whose stdout is a results table. The captured text is returned on
    the client as ``startup_banner`` so a caller can still show it.
    """
    _ensure_importable()
    from src.llm_client import LLMClient  # noqa: PLC0415

    if not quiet:
        return LLMClient()

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        client = LLMClient()
    try:
        client.startup_banner = buffer.getvalue()
    except AttributeError:  # pragma: no cover - the client is a plain object
        pass
    return client


def llm_models() -> dict[str, Optional[str]]:
    """The model ids the sister repository is configured to use.

    Read rather than hardcoded. The gateway's model names carry a ``protected.``
    prefix and change as the gateway adds models, and there is no reason for
    this repository to hold a second opinion about which one is current.

    Returns ``fast`` and ``thorough``, either of which may be None when the
    sister repository is unavailable, in which case the client's own default
    stands.
    """
    try:
        _ensure_importable()
        from src import config  # noqa: PLC0415

        return {
            "fast": getattr(config, "LLM_MODEL", None),
            "thorough": getattr(config, "LLM_MODEL_THOROUGH", None),
        }
    except Exception:
        return {"fast": None, "thorough": None}


# --- the pinned OpenSn checkout ---------------------------------------------


def checkout_path() -> Path:
    """The full OpenSn checkout at the pinned commit."""
    path = require() / "data" / "repos" / "opensn"
    if not path.is_dir():
        raise BridgeError(
            f"no OpenSn checkout at {path}. The sister repository builds it with "
            f"scripts/build_pack.py; template sources and gold values are read from it."
        )
    return path


TRANSPORT_STEADY = Path("test/python/modules/linear_boltzmann_solvers/transport_steady")
"""Directory of small, valid PyOpenSn scripts that the templates derive from.

It also holds ``tests.json``, the regression gold values used by ``parse/gold.py``.
"""


def test_script(name: str) -> Path:
    """Path to one script in the steady-state transport regression directory."""
    path = checkout_path() / TRANSPORT_STEADY / name
    if not path.exists():
        raise BridgeError(f"no regression script named {name} at {path}")
    return path


def tests_json_path() -> Path:
    """Path to the gold values for the steady-state transport regression suite."""
    return checkout_path() / TRANSPORT_STEADY / "tests.json"


@dataclass(frozen=True)
class Grounding:
    """What retrieval found for one query, reduced to what a prompt needs."""

    query: str
    citations: tuple[str, ...]
    excerpts: tuple[str, ...]

    def as_prompt_block(self, max_chars: int = 2000) -> str:
        """The hits formatted for inclusion in an LLM prompt, with citations."""
        blocks = []
        for citation, excerpt in zip(self.citations, self.excerpts):
            blocks.append(f"[{citation}]\n{excerpt[:max_chars]}")
        return "\n\n".join(blocks)


def ground(query: str, k: int = 5, commit: str = PINNED_OPENSN_COMMIT) -> Grounding:
    """Retrieve evidence for ``query``, reduced to citations and text.

    This is the shape ``narrate.request_to_spec`` needs: it must not hand a
    prompt anything that lacks a location, because an ungrounded claim about
    the API is exactly the failure this project is built to avoid.
    """
    hits = search(query, k=k, commit=commit)
    return Grounding(
        query=query,
        citations=tuple(h.citation() for h in hits),
        excerpts=tuple(h.text for h in hits),
    )
