"""The catalogue of studies: what was run, and which studies compare."""

from __future__ import annotations

import math

import yaml

from autoopensn.runner import ReferenceRunner
from autoopensn.spec import Spec
from autoopensn.store import Catalog, RunStore, group_key
from autoopensn.study import run_study

TOLERANCES = """
name: tolerances
template: reed_1d
template_parameters: {emit_avg_flux: true}
solver: {l_abs_tol: 1.0e-6}
sweep:
  - parameter: l_abs_tol
    values: [1.0e-4, 1.0e-6]
reference: {l_abs_tol: 1.0e-10}
observables: [avg_flux, sweeps, wall_time]
"""


def spec(text=TOLERANCES, **changes):
    return Spec(**{**yaml.safe_load(text), **changes})


def run(tmp_path, s=None):
    with RunStore(tmp_path / "cache.db") as store:
        return run_study(s or spec(), ReferenceRunner(strict=False), run_root=tmp_path / "runs",
                         store=store, engine_note="built-in solver")


def test_group_key_is_scenario_and_varied_parameters():
    assert group_key("reed_1d", ["l_abs_tol"]) == "reed_1d:l_abs_tol"
    assert group_key("reed_1d", ["b", "a", "a"]) == "reed_1d:a,b"
    assert group_key("reed_1d", []) == "reed_1d:baseline"


def test_a_study_run_with_a_store_is_recorded(tmp_path):
    result = run(tmp_path)
    assert result.study_id
    with Catalog(tmp_path / "cache.db") as catalog:
        study = catalog.study(result.study_id)
    assert study["name"] == "tolerances"
    assert study["group_key"] == "reed_1d:l_abs_tol"
    assert study["engine_note"] == "built-in solver"
    assert len(study["cases"]) == 3
    assert study["cases"][0]["parameters"] == {"l_abs_tol": 1.0e-4}


def test_each_case_keeps_its_whole_row(tmp_path):
    """Observables, convergence and the runner name all come back."""
    result = run(tmp_path)
    with Catalog(tmp_path / "cache.db") as catalog:
        row = catalog.study(result.study_id)["cases"][0]["row"]
    for column in ("sweeps", "converged", "wall_time", "runner", "avg_flux"):
        assert column in row
    assert row["runner"] == "reference"


def test_rows_are_json_safe(tmp_path):
    """A NaN written as is makes a row that cannot be read back."""
    result = run(tmp_path)
    with Catalog(tmp_path / "cache.db") as catalog:
        for case in catalog.study(result.study_id)["cases"]:
            for value in case["row"].values():
                assert not (isinstance(value, float) and math.isnan(value))


def test_a_study_answered_from_the_cache_is_still_recorded(tmp_path):
    """Asking again is itself an event; the record says it was cached."""
    run(tmp_path)
    second = run(tmp_path)
    with Catalog(tmp_path / "cache.db") as catalog:
        studies = catalog.studies()
    assert len(studies) == 2
    assert next(s for s in studies if s["id"] == second.study_id)["cache_hits"] == 3


def test_studies_of_the_same_question_group_together(tmp_path):
    run(tmp_path)
    run(tmp_path, spec(name="tolerances_again"))
    run(tmp_path, spec(name="restarts", sweep=[{"parameter": "gmres_restart_interval", "values": [5, 20]}]))
    with Catalog(tmp_path / "cache.db") as catalog:
        groups = {g["group_key"]: g["studies"] for g in catalog.groups()}
        same = catalog.studies(group="reed_1d:l_abs_tol")
    assert groups == {"reed_1d:l_abs_tol": 2, "reed_1d:gmres_restart_interval": 1}
    assert {s["name"] for s in same} == {"tolerances", "tolerances_again"}


def test_without_a_store_nothing_is_recorded(tmp_path):
    result = run_study(spec(), ReferenceRunner(strict=False), run_root=tmp_path / "runs")
    assert result.study_id is None


def test_forgetting_a_study_leaves_the_cache_alone(tmp_path):
    result = run(tmp_path)
    with Catalog(tmp_path / "cache.db") as catalog:
        assert catalog.forget(result.study_id)
        assert catalog.study(result.study_id) is None
    with RunStore(tmp_path / "cache.db") as store:
        assert len(list(store.entries())) == 3


# --- rank counts ---------------------------------------------------------------


def test_a_spec_without_ranks_runs_on_the_scenarios_own():
    from autoopensn.study import ranks_for
    from autoopensn.templates import load_template

    template = load_template("hdpe_example")
    assert template.num_procs > 1
    unset = Spec(**{"name": "h", "template": "hdpe_example", "observables": ["wall_time"]})
    assert ranks_for(unset, template) == template.num_procs


def test_a_spec_that_asks_for_one_rank_gets_one():
    """Read from what the spec set, not from the value: 1 asked for is 1."""
    from autoopensn.study import ranks_for
    from autoopensn.templates import load_template

    asked = Spec(**{"name": "h", "template": "hdpe_example", "observables": ["wall_time"], "num_procs": 1})
    assert ranks_for(asked, load_template("hdpe_example")) == 1


def test_results_at_different_rank_counts_do_not_share_cache_entries(tmp_path):
    one = run(tmp_path, spec(num_procs=1))
    two = run(tmp_path, spec(num_procs=2))
    assert one.cache_hits == [] and two.cache_hits == [], "eight ranks is not one rank's evidence"


def test_an_unchosen_rank_count_survives_a_trip_through_yaml():
    """Every spec from the web page arrives as YAML. It used to arrive saying
    num_procs: 1, and HDPE ran on one rank instead of eight."""
    import yaml

    from autoopensn.spec.model import Spec
    from autoopensn.study import ranks_for
    from autoopensn.templates import load_template

    template = load_template("hdpe_example")
    unset = Spec(name="h", template="hdpe_example", observables=["wall_time"])
    assert "num_procs" not in unset.to_yaml()
    again = Spec(**yaml.safe_load(unset.to_yaml()))
    assert ranks_for(again, template) == template.num_procs > 1
    chosen = Spec(**yaml.safe_load(unset.model_copy(update={"num_procs": 1}).to_yaml()))
    assert ranks_for(chosen, template) == 1
