"""Tests for the adapter to the sister repository.

Two halves. The half that needs the sister repository skips itself when it is
absent, so this repository stays testable on its own. The half that does not,
the architectural constraint that nothing else imports ``src.kp``, always runs,
because that constraint is exactly the kind that erodes quietly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from autoopensn import PINNED_OPENSN_COMMIT, kp_bridge

PACKAGE = Path(__file__).resolve().parent.parent / "autoopensn"

needs_sister = pytest.mark.skipif(
    not kp_bridge.available(),
    reason="sister repository (Code_assistant_TAU) not present",
)


# --- the constraint ---------------------------------------------------------


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_only_the_bridge_imports_the_sister_repository():
    """The whole reason kp_bridge exists.

    A sys.path mutation scattered across five modules is not a dependency, it
    is a haunting. If this test ever fails, the fix is to add a function to
    kp_bridge, not to add an import.
    """
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name == "kp_bridge.py":
            continue
        for module in imported_modules(path):
            if module == "src" or module.startswith("src."):
                offenders.append(f"{path.relative_to(PACKAGE.parent)} imports {module}")
    assert not offenders, "\n".join(offenders)


# Nothing but the bridge. This used to carry an exemption for the Streamlit
# page, which ran as a script and so had to put *this* repository on the path
# before its first import. With the page gone the exemption goes with it, and
# the constraint is now exactly what it always claimed to be: one module
# touches sys.path.
SYS_PATH_ALLOWED = {"kp_bridge.py"}


def test_nothing_else_mutates_sys_path():
    offenders = []
    for path in sorted(PACKAGE.rglob("*.py")):
        if path.name in SYS_PATH_ALLOWED:
            continue
        text = path.read_text()
        if "sys.path.append" in text or "sys.path.insert" in text:
            offenders.append(str(path.relative_to(PACKAGE.parent)))
    assert not offenders, offenders


def test_the_pinned_commit_is_stated_once():
    """A commit hash repeated in three files drifts in two of them."""
    from autoopensn.templates import load_template

    assert load_template("reed_1d").opensn_commit == PINNED_OPENSN_COMMIT


# --- graceful absence -------------------------------------------------------


def test_a_missing_sister_repository_explains_the_fix(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOOPENSN_KP_REPO", str(tmp_path / "nowhere"))
    assert not kp_bridge.available()
    with pytest.raises(kp_bridge.BridgeError) as excinfo:
        kp_bridge.require()
    assert "AUTOOPENSN_KP_REPO" in str(excinfo.value)


def test_repo_path_is_overridable(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTOOPENSN_KP_REPO", str(tmp_path))
    assert kp_bridge.repo_path() == tmp_path.resolve()


# --- the real pack ----------------------------------------------------------


@needs_sister
def test_manifest_records_the_pinned_commit():
    assert kp_bridge.pack_commit() == PINNED_OPENSN_COMMIT


@needs_sister
def test_searcher_returns_hits_for_gmres_restart_interval():
    """The deliverable: a PackSearcher over the pinned pack, without the model.

    The embedding model is skipped, so this is BM25 over FTS5, which is fast
    and is entirely adequate for the symbol-shaped queries this tool makes.
    """
    hits = kp_bridge.search("GMRES restart interval", k=5)
    assert hits, "no hits for a term that is documented at this commit"
    assert all(hit.citation() for hit in hits), "every hit must carry a location"
    assert any("gmres" in (hit.text or "").lower() for hit in hits)


@needs_sister
def test_searcher_is_cached():
    assert kp_bridge.searcher() is kp_bridge.searcher()


@needs_sister
def test_grounding_pairs_citations_with_text():
    grounding = kp_bridge.ground("inner_linear_method", k=3)
    assert len(grounding.citations) == len(grounding.excerpts)
    assert grounding.as_prompt_block()


@needs_sister
def test_the_checkout_has_the_reed_regression_script():
    assert kp_bridge.test_script("reed_balance.py").exists()
    assert kp_bridge.tests_json_path().exists()


@needs_sister
def test_an_unknown_pack_commit_lists_what_is_present():
    with pytest.raises(kp_bridge.BridgeError) as excinfo:
        kp_bridge.pack_dir("0" * 40)
    assert PINNED_OPENSN_COMMIT in str(excinfo.value)
