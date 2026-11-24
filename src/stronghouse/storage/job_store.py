"""SQLite storage for background jobs (jobs.sqlite3)."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

from stronghouse.core.jobs import Job, JobState
from stronghouse.core.models import KvCacheType, PlanSettings
from stronghouse.storage.sqlite_db import open_database

_MIGRATIONS_PACKAGE = "stronghouse.storage.job_migrations"
JOB_HISTORY_LIMIT = 100
INTERRUPTED_ERROR = "The app stopped while this job was running. Press Retry to run it again."


class JobStore:
    def __init__(self, database_path: Path) -> None:
        self._lock = threading.Lock()
        self._connection = open_database(database_path, _MIGRATIONS_PACKAGE)
        self._fail_interrupted_jobs()

    def add(self, job: Job) -> Job:
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """INSERT INTO jobs (title, prompt, model_id, plan_settings, think, max_tokens,
                                     state, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job.title, job.prompt, job.model_id, json.dumps(asdict(job.plan_settings)),
                    int(job.think), job.max_tokens, job.state.value, job.created_at,
                ),
            )  # fmt: skip
        return replace(job, id=cursor.lastrowid)

    def save(self, job: Job) -> None:
        """Persist a job's progress fields (state, timing, output, error)."""
        with self._lock, self._connection:
            self._connection.execute(
                """UPDATE jobs SET state = ?, started_at = ?, finished_at = ?, output_path = ?,
                       generated_tokens = ?, tokens_per_second = ?, error = ?
                   WHERE id = ?""",
                (
                    job.state.value, job.started_at, job.finished_at, job.output_path,
                    job.generated_tokens, job.tokens_per_second, job.error, job.id,
                ),
            )  # fmt: skip

    def get(self, job_id: int) -> Job | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _to_job(row) if row else None

    def list(self, limit: int = JOB_HISTORY_LIMIT) -> list[Job]:
        """Newest first."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_to_job(row) for row in rows]

    def next_queued(self) -> Job | None:
        """The oldest job still waiting to run."""
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM jobs WHERE state = ? ORDER BY id LIMIT 1", (JobState.QUEUED.value,)
            ).fetchone()
        return _to_job(row) if row else None

    def close(self) -> None:
        self._connection.close()

    def _fail_interrupted_jobs(self) -> None:
        # The worker thread does not survive a restart. Re-running automatically could
        # loop forever on a job that crashes the app, so ask the person instead.
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE jobs SET state = ?, error = ?, finished_at = ? WHERE state IN (?, ?)",
                (
                    JobState.FAILED.value, INTERRUPTED_ERROR, now(),
                    JobState.LOADING.value, JobState.RUNNING.value,
                ),
            )  # fmt: skip


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _to_job(row) -> Job:
    settings = json.loads(row["plan_settings"])
    settings["kv_cache_type"] = KvCacheType(settings["kv_cache_type"])
    return Job(
        id=row["id"],
        title=row["title"],
        prompt=row["prompt"],
        model_id=row["model_id"],
        plan_settings=PlanSettings(**settings),
        think=bool(row["think"]),
        max_tokens=row["max_tokens"],
        state=JobState(row["state"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        output_path=row["output_path"],
        generated_tokens=row["generated_tokens"],
        tokens_per_second=row["tokens_per_second"],
        error=row["error"],
    )
