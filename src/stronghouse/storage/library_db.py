"""SQLite index of the local model library. Only this module writes SQL."""

from __future__ import annotations

import sqlite3
import threading
from importlib import resources
from pathlib import Path

from stronghouse.core.models import LibraryEntry

_MIGRATIONS_PACKAGE = "stronghouse.storage.migrations"
_TAG_SEPARATOR = ","


class LibraryStore:
    """Read and write library entries. Thread-safe for the app's small workload."""

    def __init__(self, database_path: Path) -> None:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(database_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._apply_migrations()

    def upsert(self, entry: LibraryEntry) -> LibraryEntry:
        """Add a model, or update it if a model with the same path exists."""
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                INSERT INTO models (path, repo_id, filename, size_bytes, sha256, license,
                                    architecture, is_moe, tags, added_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    repo_id = excluded.repo_id, filename = excluded.filename,
                    size_bytes = excluded.size_bytes, sha256 = excluded.sha256,
                    license = excluded.license, architecture = excluded.architecture,
                    is_moe = excluded.is_moe
                RETURNING id
                """,
                (
                    entry.path, entry.repo_id, entry.filename, entry.size_bytes, entry.sha256,
                    entry.license, entry.architecture, int(entry.is_moe),
                    _TAG_SEPARATOR.join(entry.tags), entry.added_at,
                ),
            )  # fmt: skip
            (entry_id,) = cursor.fetchone()
        return self.get(entry_id)  # type: ignore[return-value]

    def get(self, entry_id: int) -> LibraryEntry | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM models WHERE id = ?", (entry_id,)
            ).fetchone()
        return _to_entry(row) if row else None

    def list_entries(self, search: str = "") -> list[LibraryEntry]:
        """All entries, newest first, optionally filtered by text in name, repo or tags."""
        pattern = f"%{search}%"
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT * FROM models
                WHERE filename LIKE ? OR repo_id LIKE ? OR tags LIKE ?
                ORDER BY added_at DESC, id DESC
                """,
                (pattern, pattern, pattern),
            ).fetchall()
        return [_to_entry(row) for row in rows]

    def set_tags(self, entry_id: int, tags: list[str]) -> LibraryEntry | None:
        clean = sorted({tag.strip().lower() for tag in tags if tag.strip()})
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE models SET tags = ? WHERE id = ?", (_TAG_SEPARATOR.join(clean), entry_id)
            )
        return self.get(entry_id)

    def remove(self, entry_id: int) -> None:
        """Forget an entry. The file on disk is left alone on purpose."""
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM models WHERE id = ?", (entry_id,))

    def close(self) -> None:
        self._connection.close()

    def _apply_migrations(self) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)"
            )
            applied = {
                row[0] for row in self._connection.execute("SELECT version FROM schema_migrations")
            }
        for version, sql in _migration_scripts():
            if version in applied:
                continue
            with self._lock:
                # executescript commits on its own, so record the version in the same script.
                self._connection.executescript(
                    f"BEGIN;\n{sql}\nINSERT INTO schema_migrations VALUES ({version});\nCOMMIT;"
                )


def _migration_scripts() -> list[tuple[int, str]]:
    scripts = []
    for item in resources.files(_MIGRATIONS_PACKAGE).iterdir():
        if item.name.endswith(".sql"):
            scripts.append((int(item.name.split("_", 1)[0]), item.read_text(encoding="utf-8")))
    return sorted(scripts)


def _to_entry(row: sqlite3.Row) -> LibraryEntry:
    return LibraryEntry(
        id=row["id"],
        path=row["path"],
        repo_id=row["repo_id"],
        filename=row["filename"],
        size_bytes=row["size_bytes"],
        sha256=row["sha256"],
        license=row["license"],
        architecture=row["architecture"],
        is_moe=bool(row["is_moe"]),
        tags=[tag for tag in row["tags"].split(_TAG_SEPARATOR) if tag],
        added_at=row["added_at"],
    )
