"""Studies started over MCP, kept on disk and run in their own process.

The MCP server cannot keep a study in memory, because it may not live long
enough to finish one. URSA's client, like any client built on
``langchain-mcp-adapters`` defaults, is stateless: every tool call starts a new
``autoopensn mcp`` process and stops it when the call returns. A study held in
a thread of that process dies with it, and the next call, in a new process, has
never heard of the job. That is how this module came to exist: the first time
URSA drove the server, ``get_results`` answered "no job" to an id that
``run_study`` had returned one call earlier.

So a job is a directory, and a study runs in a worker process of its own:

    <run_root>/.mcp_jobs/<job_id>/
        request.json   the spec and where to run it; written once
        state.json     status, progress, and the table when done; replaced
                       atomically, so a reader never sees half a file
        pid            the worker's process id, written by the server
        worker.log     the worker's stdout and stderr

``start`` writes the request and launches ``python -m autoopensn.mcp_jobs
<dir>`` in a new session, so stopping the server does not stop the study. Any
later server process, or the same one, reads ``state.json``. If the worker died
without saying so, ``read`` notices that its pid is gone and reports the job as
failed with the tail of its log.

Like the server, the worker runs no language model.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import threading
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

JOBS_DIRNAME = ".mcp_jobs"
MAX_PROGRESS_LINES = 2000
_JOB_ID = re.compile(r"^[0-9a-f]{12}$")
_FINISHED = ("succeeded", "failed")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    """A table cell as JSON: NaN, NA and infinity become null, numpy becomes Python."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isinf(value):
        return None
    return value


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {key: _jsonable(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Replace a file in one step, so a concurrent reader sees old or new, never half."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(data, indent=1))
    os.replace(temporary, path)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class JobStore:
    """The jobs under one run root."""

    def __init__(self, run_root: Path):
        self.root = Path(run_root) / JOBS_DIRNAME

    def directory(self, job_id: str) -> Optional[Path]:
        """The job's directory, or None for an id that is not one of ours.

        The id arrives from a model, so it is checked before it touches a path.
        """
        if not _JOB_ID.match(job_id or ""):
            return None
        path = self.root / job_id
        return path if (path / "state.json").is_file() else None

    def create(self, request: dict[str, Any], *, total: int, label: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        path = self.root / job_id
        path.mkdir(parents=True)
        _write_json(path / "request.json", request)
        _write_json(
            path / "state.json",
            {
                "id": job_id,
                "status": "queued",
                "created_at": _now(),
                "finished_at": None,
                "label": label,
                "total": total,
                "completed": 0,
                "progress": [],
                "note": "",
                "error": None,
                "pid": None,
            },
        )
        return job_id

    def read(self, job_id: str) -> Optional[dict[str, Any]]:
        path = self.directory(job_id)
        if path is None:
            return None
        state = json.loads((path / "state.json").read_text())
        pid_file = path / "pid"
        pid = state.get("pid") or (int(pid_file.read_text()) if pid_file.exists() else None)
        if state["status"] not in _FINISHED and pid and not _alive(pid):
            # Read once more: the worker may have finished between the two reads.
            state = json.loads((path / "state.json").read_text())
            if state["status"] not in _FINISHED:
                log = path / "worker.log"
                tail = log.read_text(errors="replace")[-1500:] if log.exists() else ""
                state.update(
                    status="failed",
                    finished_at=_now(),
                    error="the study's worker process exited without finishing"
                    + (f". Its log ends:\n{tail}" if tail else ""),
                )
                _write_json(path / "state.json", state)
        return state


def start(path: Path, *, detach: bool = True) -> None:
    """Run the job in ``path``: in its own process by default, or on a thread.

    The thread is for the test suite, whose guard against reaching the cluster
    patches this process and would not reach a worker.
    """
    if not detach:
        threading.Thread(target=execute, args=(path,), name=f"study-{path.name}", daemon=True).start()
        return
    log = open(path / "worker.log", "ab")
    try:
        process = subprocess.Popen(
            [sys.executable, "-m", "autoopensn.mcp_jobs", str(path)],
            stdin=subprocess.DEVNULL,
            # Never the server's stdout: on stdio that is the protocol.
            stdout=log,
            stderr=log,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log.close()
    # A file of its own, not a field in state.json: the worker is writing
    # state.json already, and two writers of one file lose each other's updates.
    (path / "pid").write_text(str(process.pid))


def execute(path: Path) -> None:
    """Run one study and record everything it does in ``state.json``."""
    from autoopensn.runner.select import build_runner  # noqa: PLC0415
    from autoopensn.spec import Spec  # noqa: PLC0415
    from autoopensn.store import RunStore  # noqa: PLC0415
    from autoopensn.study import run_study  # noqa: PLC0415

    state_path = path / "state.json"
    lock = threading.Lock()

    def update(**fields: Any) -> None:
        with lock:
            state = json.loads(state_path.read_text())
            state.update(fields)
            _write_json(state_path, state)

    def say(message: str) -> None:
        with lock:
            state = json.loads(state_path.read_text())
            state["progress"] = (state["progress"] + [message])[-MAX_PROGRESS_LINES:]
            # run_study emits one "[i/n] case: ..." line per point.
            if message.startswith("["):
                state["completed"] = min(state["completed"] + 1, state["total"] or state["completed"] + 1)
            _write_json(state_path, state)

    update(status="running", pid=os.getpid())
    try:
        request = json.loads((path / "request.json").read_text())
        spec = Spec(**request["spec"])
        runner, note = build_runner(
            request["runner"], request.get("host"), request.get("fixtures")
        )
        update(note=note)
        say(note)
        run_root = Path(request["run_root"])
        cache_path = request.get("cache_path")
        if cache_path:
            with RunStore(Path(cache_path)) as store:
                result = run_study(spec, runner, run_root=run_root, store=store, progress=say)
        else:
            result = run_study(spec, runner, run_root=run_root, progress=say)
        update(
            status="succeeded",
            finished_at=_now(),
            table=records(result.table),
            summary=result.summary(),
            run_root=str(result.run_root),
            cache_hits=list(result.cache_hits),
        )
    except BaseException as exc:  # noqa: BLE001
        # Whatever stopped the study, the next poll must be told why.
        update(
            status="failed",
            finished_at=_now(),
            error=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(limit=12),
        )
        if not isinstance(exc, Exception):
            raise


if __name__ == "__main__":
    execute(Path(sys.argv[1]))
