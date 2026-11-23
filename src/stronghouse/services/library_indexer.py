"""Keep the SQLite library index in step with the files in the archive folder."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from stronghouse.core.model_profiler import profile_model
from stronghouse.core.models import LibraryEntry, ModelProfile
from stronghouse.services.gguf_reader import GgufFormatError, read_gguf, split_shard_paths
from stronghouse.storage import archive
from stronghouse.storage.library_db import LibraryStore

log = logging.getLogger(__name__)


class LibraryIndexer:
    def __init__(self, store: LibraryStore, archive_dir: Path) -> None:
        self._store = store
        self._archive_dir = archive_dir

    def index_file(
        self, repo_id: str, filename: str, path: Path, sha256: str | None, license_name: str | None
    ) -> LibraryEntry:
        """Add or refresh one model in the index. Unreadable files are still listed."""
        architecture, is_moe = None, False
        try:
            profile = self.profile(path)
            architecture, is_moe = profile.architecture, profile.is_moe
        except (GgufFormatError, ValueError, OSError) as error:
            log.warning(
                "could not read model metadata", extra={"path": str(path), "error": str(error)}
            )

        size = sum(part.stat().st_size for part in split_shard_paths(path) if part.exists())
        return self._store.upsert(
            LibraryEntry(
                path=str(path),
                repo_id=repo_id,
                filename=filename,
                size_bytes=size,
                sha256=sha256,
                license=license_name,
                architecture=architecture,
                is_moe=is_moe,
                tags=[],
                added_at=datetime.now(UTC).isoformat(timespec="seconds"),
            )
        )

    def rescan(self) -> int:
        """Index every model found in the archive folder. Returns how many were indexed.

        Useful after copying an archive onto a new drive or deleting the database.
        """
        count = 0
        for found in archive.scan_archive(self._archive_dir):
            if split_shard_paths(found.path)[0] != found.path:
                continue  # only the first part of a split model gets an entry
            folder = archive.repo_dir(self._archive_dir, found.repo_id)
            sidecar = archive.read_sidecar(folder, found.repo_id)
            recorded = sidecar.files.get(found.filename)
            self.index_file(
                found.repo_id,
                found.filename,
                found.path,
                recorded.sha256 if recorded else None,
                sidecar.license,
            )
            count += 1
        return count

    @staticmethod
    def profile(path: Path) -> ModelProfile:
        contents = read_gguf(path)
        return profile_model(contents.metadata, contents.tensors)
