"""Background study runs, and the record of one in flight.

This module exists because of the cluster. A study against the reference solver
finishes in under a second and could be run inside a request handler; a study
against real OpenSn on a shared node takes minutes to hours, and an HTTP request
that waits that long is one a proxy will cut, a browser will abandon, and a user
will assume has hung.

So a study is started, given an id, and polled. ``run_study`` already takes a
``progress`` callback, so nothing in the engine changes: the callback appends to
a list the poller reads.

**Deliberately in-process, and deliberately not durable.** Jobs live in a dict
guarded by a lock, and a restart loses them. That is the honest scope of this
layer: a single person running studies from a page on their own machine. A
queue that survives restarts and coordinates several users is a different
system, and pretending a dict is one would be worse than being clear that it is
not. What *is* durable is the run directory and the SQLite cache, which the
engine writes as it goes and which outlive the process either way — a lost job
record costs the progress log, not the results.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"

MAX_PROGRESS_LINES = 2000
"""A sweep of thousands of points must not grow a list without bound. The tail
is what matters when watching, and the whole log is in the run directory."""

MAX_RETAINED = 64
"""Finished jobs kept for polling after the fact. Oldest are dropped first."""


@dataclass
class Job:
    """One study, in flight or finished."""

    id: str
    status: str = QUEUED
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    finished_at: Optional[str] = None
    progress: list[str] = field(default_factory=list)
    total: int = 0
    """Number of sweep points, known before anything runs."""
    completed: int = 0
    error: Optional[str] = None
    traceback: Optional[str] = None
    table: Optional[list[dict[str, Any]]] = None
    summary: Optional[str] = None
    run_root: Optional[str] = None
    cache_hits: list[str] = field(default_factory=list)
    label: str = ""
    """What the study was, for a UI listing jobs without re-reading the spec."""
    note: str = ""
    """Which engine produced these numbers, in words. Shown beside the results:
    a table that does not say whether it came from OpenSn or from this package's
    own solver is a table nobody should quote."""

    @property
    def done(self) -> bool:
        return self.status in (SUCCEEDED, FAILED)

    def as_dict(self, *, include_table: bool = True) -> dict[str, Any]:
        """A JSON-safe view. The table is large, so a listing can omit it."""
        record: dict[str, Any] = {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "finished_at": self.finished_at,
            "progress": list(self.progress),
            "total": self.total,
            "completed": self.completed,
            "error": self.error,
            "summary": self.summary,
            "run_root": self.run_root,
            "cache_hits": list(self.cache_hits),
            "label": self.label,
            "note": self.note,
        }
        if include_table:
            record["table"] = self.table
        return record


class JobRegistry:
    """Starts studies on threads and hands back their progress.

    Every method takes the lock. The contention is trivial — a poll every second
    against a dict — and the alternative, reasoning about which fields a worker
    thread may touch while a request handler reads them, is how this kind of
    code grows races that only appear under a real run.
    """

    def __init__(self, *, max_retained: int = MAX_RETAINED):
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._max_retained = max_retained

    # --- reading ---

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> list[Job]:
        """Newest first."""
        with self._lock:
            return [self._jobs[i] for i in reversed(self._order) if i in self._jobs]

    # --- writing, from the worker thread ---

    def _say(self, job_id: str, message: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.progress.append(message)
            if len(job.progress) > MAX_PROGRESS_LINES:
                # Keep the tail; the head is the least interesting part of a
                # long sweep and the run directory has all of it.
                del job.progress[: len(job.progress) - MAX_PROGRESS_LINES]
            # run_study emits one line per point as it completes or hits cache.
            # Counting them is cruder than instrumenting the engine, and it
            # keeps the engine unaware that an API exists.
            if message.startswith("["):
                job.completed = min(job.completed + 1, job.total or job.completed + 1)

    def _finish(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in fields.items():
                setattr(job, key, value)
            job.finished_at = datetime.now(timezone.utc).isoformat()

    def _evict(self) -> None:
        """Drop the oldest finished jobs. Caller holds the lock."""
        finished = [i for i in self._order if i in self._jobs and self._jobs[i].done]
        excess = len(finished) - self._max_retained
        for job_id in finished[:excess] if excess > 0 else []:
            self._jobs.pop(job_id, None)
            self._order.remove(job_id)

    # --- starting ---

    def start(
        self,
        work: Callable[[Callable[[str], None]], dict[str, Any]],
        *,
        total: int = 0,
        label: str = "",
        note: str = "",
    ) -> Job:
        """Run ``work`` on a thread, passing it a progress callback.

        ``work`` returns the fields to record on success. An exception becomes a
        failed job rather than a lost one: a study that dies must leave a
        readable reason behind, because the person waiting on it has no other
        window into what happened.
        """
        job = Job(id=uuid.uuid4().hex[:12], total=total, label=label, note=note)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._evict()

        def target() -> None:
            with self._lock:
                current = self._jobs.get(job.id)
                if current is not None:
                    current.status = RUNNING
            try:
                fields = work(lambda message: self._say(job.id, message))
                self._finish(job.id, status=SUCCEEDED, **fields)
            except BaseException as exc:  # noqa: BLE001
                # Broad on purpose. Whatever killed the study, the poller must
                # be told; a job stuck on "running" forever is the one outcome
                # with no recovery from the outside.
                self._finish(
                    job.id,
                    status=FAILED,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(limit=12),
                )

        thread = threading.Thread(target=target, name=f"study-{job.id}", daemon=True)
        thread.start()
        return job
