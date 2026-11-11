"""A catalogue of every study that was run, for finding and comparing them later.

The run cache answers "has this exact computation been done?" and is keyed by
what was computed, so one row can serve many studies. That is the wrong shape
for "what did I run last Tuesday, and how did it compare with the week
before?", which is a question about studies, not computations. So this keeps
its own record, in the same SQLite file:

``studies``
    One row per study that was run: its spec, scenario, which engine ran it and
    a sentence saying so, the pinned commit, when, and a few totals.

``study_cases``
    One row per case of a study: its parameters and its whole results row,
    observables, convergence, performance and the runner's metadata included,
    stored as JSON so that a column added later does not need a migration.

**Groups.** Two studies belong to the same group when they ran the same scenario
and varied the same parameters — every ``reed_1d`` study sweeping
``gmres_restart_interval``, say. Those are the studies whose tables line up
column for column, which is what makes them quick to compare. A study that
varied nothing groups with the other baselines of its scenario.

Every study is recorded, including one answered entirely from the cache: asking
the question again is itself something that happened, and the record says how
many of its cases were cached.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS studies (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    template        TEXT NOT NULL,
    group_key       TEXT NOT NULL,
    swept           TEXT NOT NULL,
    spec_yaml       TEXT NOT NULL,
    runner          TEXT NOT NULL,
    runner_config   TEXT NOT NULL,
    engine_note     TEXT,
    pack_commit     TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    cases           INTEGER NOT NULL,
    ok_cases        INTEGER NOT NULL,
    converged_cases INTEGER,
    cache_hits      INTEGER NOT NULL,
    wall_time_total REAL
);
CREATE INDEX IF NOT EXISTS studies_by_group ON studies (group_key, created_at);
CREATE INDEX IF NOT EXISTS studies_by_time ON studies (created_at);

CREATE TABLE IF NOT EXISTS study_cases (
    study_id     TEXT NOT NULL REFERENCES studies(id),
    case_id      TEXT NOT NULL,
    position     INTEGER NOT NULL,
    label        TEXT,
    is_reference INTEGER NOT NULL DEFAULT 0,
    run_key      TEXT,
    directory    TEXT,
    parameters   TEXT NOT NULL,
    row          TEXT NOT NULL,
    PRIMARY KEY (study_id, case_id)
);
"""

BASELINE = "baseline"


def group_key(template: str, swept: Iterable[str]) -> str:
    """The comparison group: the scenario and the parameters it varies."""
    names = sorted(set(swept))
    return f"{template}:{','.join(names) if names else BASELINE}"


def _plain(value: Any) -> Any:
    """A JSON-safe copy of one table cell.

    NaN is not JSON, and a numpy scalar is not a float. Either, written as is,
    makes a row that cannot be read back, which is worse than a blank.
    """
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    try:
        import pandas as pd  # noqa: PLC0415

        if value is pd.NA or value is pd.NaT:
            return None
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


class Catalog:
    """Records studies and reads them back.

    Opens its own connection, with a generous lock timeout, because studies
    finish on worker threads while requests read on others, and SQLite will not
    share one connection between threads.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.path), timeout=30)
        self.connection.row_factory = sqlite3.Row
        with closing(self.connection.cursor()) as cursor:
            cursor.executescript(SCHEMA)
        self.connection.commit()

    def __enter__(self) -> "Catalog":
        return self

    def __exit__(self, *exception) -> None:
        self.close()

    def close(self) -> None:
        self.connection.close()

    # --- writing ---

    def record(
        self,
        *,
        spec: Any,
        rows: list[dict[str, Any]],
        runner: str,
        runner_config: dict[str, Any],
        engine_note: Optional[str] = None,
        run_keys: Optional[dict[str, str]] = None,
        directories: Optional[dict[str, str]] = None,
        cache_hits: int = 0,
    ) -> str:
        """Record one study and its cases. Returns the study's id."""
        study_id = uuid.uuid4().hex[:12]
        swept = [dimension.parameter for dimension in spec.sweep]
        clean_rows = [{key: _plain(value) for key, value in row.items()} for row in rows]
        converged = [row.get("converged") for row in clean_rows]
        walls = [row.get("wall_time") for row in clean_rows if isinstance(row.get("wall_time"), (int, float))]

        with self.connection:
            self.connection.execute(
                """
                INSERT INTO studies (id, name, template, group_key, swept, spec_yaml,
                    runner, runner_config, engine_note, pack_commit, created_at, cases,
                    ok_cases, converged_cases, cache_hits, wall_time_total)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    study_id,
                    spec.name,
                    spec.template,
                    group_key(spec.template, swept),
                    json.dumps(swept),
                    spec.to_yaml(),
                    runner,
                    json.dumps(runner_config, sort_keys=True, default=str),
                    engine_note,
                    spec.pack_commit,
                    datetime.now(timezone.utc).isoformat(),
                    len(clean_rows),
                    sum(1 for row in clean_rows if row.get("ok")),
                    sum(1 for value in converged if value is True)
                    if any(value is not None for value in converged)
                    else None,
                    cache_hits,
                    round(sum(walls), 6) if walls else None,
                ),
            )
            for position, row in enumerate(clean_rows):
                case_id = str(row.get("case_id"))
                parameters = {
                    key[len("param_"):]: value
                    for key, value in row.items()
                    if key.startswith("param_")
                }
                self.connection.execute(
                    """
                    INSERT INTO study_cases (study_id, case_id, position, label,
                        is_reference, run_key, directory, parameters, row)
                    VALUES (?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        study_id,
                        case_id,
                        position,
                        row.get("case"),
                        int(bool(row.get("is_reference"))),
                        (run_keys or {}).get(case_id),
                        (directories or {}).get(case_id),
                        json.dumps(parameters, sort_keys=True),
                        json.dumps(row, sort_keys=True),
                    ),
                )
        return study_id

    # --- reading ---

    @staticmethod
    def _summary(row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["swept"] = json.loads(record["swept"])
        record.pop("spec_yaml", None)
        record.pop("runner_config", None)
        return record

    def studies(
        self,
        *,
        group: Optional[str] = None,
        template: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Most recent first, optionally only one group or one scenario."""
        clauses, values = [], []
        if group:
            clauses.append("group_key = ?")
            values.append(group)
        if template:
            clauses.append("template = ?")
            values.append(template)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"SELECT * FROM studies {where} ORDER BY created_at DESC LIMIT ?",
            (*values, max(1, min(int(limit), 500))),
        )
        return [self._summary(row) for row in rows]

    def study(self, study_id: str) -> Optional[dict[str, Any]]:
        """One study with every case's full row, or None."""
        row = self.connection.execute("SELECT * FROM studies WHERE id = ?", (study_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        record["swept"] = json.loads(record["swept"])
        record["runner_config"] = json.loads(record["runner_config"])
        record["cases"] = [
            {
                "case_id": case["case_id"],
                "label": case["label"],
                "is_reference": bool(case["is_reference"]),
                "run_key": case["run_key"],
                "directory": case["directory"],
                "parameters": json.loads(case["parameters"]),
                "row": json.loads(case["row"]),
            }
            for case in self.connection.execute(
                "SELECT * FROM study_cases WHERE study_id = ? ORDER BY position", (study_id,)
            )
        ]
        return record

    def groups(self) -> list[dict[str, Any]]:
        """Every comparison group, with how many studies it holds."""
        rows = self.connection.execute(
            """
            SELECT group_key, template, swept, COUNT(*) AS studies,
                   MAX(created_at) AS latest
            FROM studies GROUP BY group_key ORDER BY latest DESC
            """
        )
        return [
            {**dict(row), "swept": json.loads(row["swept"])}
            for row in rows
        ]

    def forget(self, study_id: str) -> bool:
        """Remove one study from the catalogue. Cached runs are left alone."""
        with self.connection:
            self.connection.execute("DELETE FROM study_cases WHERE study_id = ?", (study_id,))
            cursor = self.connection.execute("DELETE FROM studies WHERE id = ?", (study_id,))
        return cursor.rowcount > 0
