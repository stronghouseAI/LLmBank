"""Run model downloads in the background and archive them when they finish."""

from __future__ import annotations

import itertools
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from stronghouse.services.gguf_reader import split_shard_paths
from stronghouse.services.huggingface import DownloadError, HuggingFaceClient, RemoteFile
from stronghouse.services.library_indexer import LibraryIndexer
from stronghouse.storage import archive

log = logging.getLogger(__name__)


class JobState(StrEnum):
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    DONE = "done"
    FAILED = "failed"


_ACTIVE_STATES = (JobState.QUEUED, JobState.DOWNLOADING, JobState.VERIFYING)


@dataclass(frozen=True)
class DownloadJob:
    id: int
    repo_id: str
    filename: str
    state: JobState
    bytes_done: int = 0
    bytes_total: int = 0
    error: str | None = None


class DownloadManager:
    def __init__(
        self,
        client: HuggingFaceClient,
        indexer: LibraryIndexer,
        archive_dir: Path,
        engine_version: Callable[[], str | None],
    ) -> None:
        self._client = client
        self._indexer = indexer
        self._archive_dir = archive_dir
        self._engine_version = engine_version
        self._jobs: dict[int, DownloadJob] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count(1)

    def start(self, repo_id: str, filename: str) -> DownloadJob:
        """Queue a download. For a split model, pass any part; all parts are fetched.

        Raises:
            archive.UnsafePathError: if the repo id or filename is not safe to store.
        """
        archive.file_path(self._archive_dir, repo_id, filename)  # validate early
        with self._lock:
            # Two jobs writing the same .part file corrupt each other, so a second
            # click on the same file joins the job that is already running.
            for existing in self._jobs.values():
                same_file = (existing.repo_id, existing.filename) == (repo_id, filename)
                if same_file and existing.state in _ACTIVE_STATES:
                    return existing
        job = DownloadJob(
            id=next(self._ids), repo_id=repo_id, filename=filename, state=JobState.QUEUED
        )
        self._save(job)
        threading.Thread(
            target=self._run, args=(job.id,), daemon=True, name=f"download-{job.id}"
        ).start()
        return job

    def jobs(self) -> list[DownloadJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.id, reverse=True)

    def _run(self, job_id: int) -> None:
        job = self._jobs[job_id]
        try:
            self._download_and_archive(job)
        except (DownloadError, archive.UnsafePathError, OSError, ValueError) as error:
            log.warning("download failed", extra={"repo_id": job.repo_id, "file": job.filename})
            self._update(job_id, state=JobState.FAILED, error=str(error))
        except Exception as error:
            # A background thread has no caller to raise to; record it loudly instead.
            log.exception("unexpected download failure", extra={"repo_id": job.repo_id})
            self._update(job_id, state=JobState.FAILED, error=f"Unexpected error: {error}")

    def _download_and_archive(self, job: DownloadJob) -> None:
        archive_dir = self._archive_dir
        parts = self._remote_parts(job.repo_id, job.filename)
        total = sum(part.size_bytes for part in parts)
        self._update(job.id, state=JobState.DOWNLOADING, bytes_total=total)

        folder = archive.repo_dir(archive_dir, job.repo_id)
        sidecar = archive.read_sidecar(folder, job.repo_id)
        sidecar.license = self._client.model_license(job.repo_id)
        sidecar.engine_version = self._engine_version() or sidecar.engine_version
        card = self._client.fetch_model_card(job.repo_id)
        if card is not None:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "README.md").write_text(card, encoding="utf-8")

        finished_bytes = 0
        first_sha = None
        for part in parts:
            destination = archive.file_path(archive_dir, job.repo_id, part.filename)

            def progress(done: int, _total: int, offset: int = finished_bytes) -> None:
                self._update(job.id, bytes_done=offset + done)

            sha = self._client.download(job.repo_id, part, destination, progress)
            first_sha = first_sha or sha
            finished_bytes += part.size_bytes
            sidecar.files[part.filename] = archive.ArchivedFile(
                sha256=sha, size_bytes=part.size_bytes, downloaded_at=_now()
            )
            archive.write_sidecar(folder, sidecar)

        self._update(job.id, state=JobState.VERIFYING, bytes_done=total)
        first_path = archive.file_path(archive_dir, job.repo_id, parts[0].filename)
        self._indexer.index_file(
            job.repo_id, parts[0].filename, first_path, first_sha, sidecar.license
        )
        self._update(job.id, state=JobState.DONE)

    def _remote_parts(self, repo_id: str, filename: str) -> list[RemoteFile]:
        available = {remote.filename: remote for remote in self._client.list_gguf_files(repo_id)}
        wanted = [path.as_posix() for path in split_shard_paths(Path(filename))]
        missing = [name for name in wanted if name not in available]
        if missing:
            raise DownloadError(f"Not found in {repo_id}: {', '.join(missing)}")
        return [available[name] for name in wanted]

    def _save(self, job: DownloadJob) -> None:
        with self._lock:
            self._jobs[job.id] = job

    def _update(self, job_id: int, **changes: object) -> None:
        with self._lock:
            self._jobs[job_id] = replace(self._jobs[job_id], **changes)  # type: ignore[arg-type]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
