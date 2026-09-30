"""Tests for the run cache.

The cache key is the whole design. Everything here is about what must change it
and what must not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autoopensn.runner import FakeRunner, RunRequest, RunResult
from autoopensn.store import RunStore, cache_key
from autoopensn.store.db import strip_provenance
from autoopensn.templates import render_run_script

SCRIPT = "print('reed')\n"
CONFIG = {"runner": "fake", "fixtures": "/somewhere"}


def result_for(tmp_path: Path, case_id: str = "case") -> RunResult:
    return RunResult(
        case_id=case_id,
        exit_code=0,
        stdout="Absorption=1.0e+02\n",
        stderr="",
        wall_time=4.2,
        directory=tmp_path,
        runner="fake",
        replayed=True,
        started_at="2026-01-01T00:00:00+00:00",
    )


# --- the key ----------------------------------------------------------------


def test_key_is_stable():
    assert cache_key(SCRIPT, "abc", CONFIG) == cache_key(SCRIPT, "abc", CONFIG)


def test_a_different_script_is_a_different_key():
    assert cache_key(SCRIPT, "abc", CONFIG) != cache_key("print('other')\n", "abc", CONFIG)


def test_a_different_pack_commit_is_a_different_key():
    """The same text against a different OpenSn is not the same computation."""
    assert cache_key(SCRIPT, "abc", CONFIG) != cache_key(SCRIPT, "def", CONFIG)


def test_a_different_runner_is_a_different_key():
    """A replayed fixture must not answer a question asked of the real solver."""
    other = {"runner": "local", "launcher": "mpiexec"}
    assert cache_key(SCRIPT, "abc", CONFIG) != cache_key(SCRIPT, "abc", other)


def test_key_ignores_the_provenance_header(reed_template):
    """Two studies that run the same simulation should share its result."""
    first = render_run_script(
        reed_template, {}, pack_commit="abc", spec_name="study_one", case_id="alpha"
    )
    second = render_run_script(
        reed_template, {}, pack_commit="abc", spec_name="study_two", case_id="beta"
    )
    assert first != second
    assert cache_key(first, "abc", CONFIG) == cache_key(second, "abc", CONFIG)


def test_key_does_not_ignore_a_changed_parameter(reed_template):
    first = render_run_script(
        reed_template, {}, pack_commit="abc", spec_name="s", case_id="a"
    )
    second = render_run_script(
        reed_template, {"l_abs_tol": 1.0e-6}, pack_commit="abc", spec_name="s", case_id="a"
    )
    assert cache_key(first, "abc", CONFIG) != cache_key(second, "abc", CONFIG)


def test_stripping_provenance_leaves_an_unheadered_script_alone():
    assert strip_provenance(SCRIPT) == SCRIPT


def test_stripping_provenance_removes_only_the_header(reed_template, vendored_reed):
    script = render_run_script(
        reed_template, {}, pack_commit="abc", spec_name="s", case_id="a"
    )
    assert strip_provenance(script) == vendored_reed


# --- storage ----------------------------------------------------------------


def test_put_then_get(tmp_path):
    with RunStore(tmp_path / "cache.db") as store:
        key = cache_key(SCRIPT, "abc", CONFIG)
        store.put(key, result_for(tmp_path), script=SCRIPT, pack_commit="abc", runner_config=CONFIG)
        cached = store.get(key)
    assert cached is not None
    assert cached.stdout == "Absorption=1.0e+02\n"
    assert cached.wall_time == pytest.approx(4.2)


def test_missing_key_returns_none(tmp_path):
    with RunStore(tmp_path / "cache.db") as store:
        assert store.get("nothing") is None
        assert "nothing" not in store


def test_cache_survives_reopening(tmp_path):
    path = tmp_path / "cache.db"
    key = cache_key(SCRIPT, "abc", CONFIG)
    with RunStore(path) as store:
        store.put(key, result_for(tmp_path), script=SCRIPT, pack_commit="abc", runner_config=CONFIG)
    with RunStore(path) as reopened:
        assert reopened.get(key) is not None


def test_failures_are_cached_but_remain_failures(tmp_path):
    """A deterministic crash should not be rediscovered on every invocation."""
    failure = RunResult(
        case_id="bad",
        exit_code=1,
        stdout="",
        stderr="RuntimeError\n",
        wall_time=1.0,
        directory=tmp_path,
        runner="fake",
    )
    with RunStore(tmp_path / "cache.db") as store:
        key = cache_key(SCRIPT, "abc", CONFIG)
        store.put(key, failure, script=SCRIPT, pack_commit="abc", runner_config=CONFIG)
        cached = store.get(key)
    assert cached is not None
    assert not cached.ok


def test_entries_can_be_listed_by_study(tmp_path):
    with RunStore(tmp_path / "cache.db") as store:
        for index, study in enumerate(("one", "one", "two")):
            store.put(
                f"key{index}",
                result_for(tmp_path, case_id=f"c{index}"),
                script=SCRIPT,
                pack_commit="abc",
                runner_config=CONFIG,
                study=study,
            )
        assert len(list(store.entries("one"))) == 2
        assert len(list(store.entries())) == 3


def test_forget_and_clear(tmp_path):
    with RunStore(tmp_path / "cache.db") as store:
        store.put("k", result_for(tmp_path), script=SCRIPT, pack_commit="abc", runner_config=CONFIG)
        assert store.forget("k")
        assert not store.forget("k")
        store.put("k", result_for(tmp_path), script=SCRIPT, pack_commit="abc", runner_config=CONFIG)
        assert store.clear() == 1


def test_rerunning_an_identical_point_hits_the_cache(
    tmp_path, gmres_spec, gmres_fixtures
):
    """The property the brief asks for, end to end."""
    from autoopensn.study import run_study

    runner = FakeRunner(gmres_fixtures)
    with RunStore(tmp_path / "cache.db") as store:
        first = run_study(gmres_spec, runner, run_root=tmp_path / "runs", store=store)
        second = run_study(gmres_spec, runner, run_root=tmp_path / "runs", store=store)

    assert first.cache_hits == []
    assert len(second.cache_hits) == 7
    assert second.table["sweeps"].tolist() == first.table["sweeps"].tolist()


def test_cache_can_be_bypassed(tmp_path, gmres_spec, gmres_fixtures):
    from autoopensn.study import run_study

    runner = FakeRunner(gmres_fixtures)
    with RunStore(tmp_path / "cache.db") as store:
        run_study(gmres_spec, runner, run_root=tmp_path / "runs", store=store)
        again = run_study(
            gmres_spec, runner, run_root=tmp_path / "runs", store=store, use_cache=False
        )
    assert again.cache_hits == []
