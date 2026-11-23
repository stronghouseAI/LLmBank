"""Open a SQLite database and bring its schema up to date with versioned migrations.

Each database has its own package of migration scripts named NNN_description.sql.
Applied versions are recorded in a schema_migrations table, so every script runs
exactly once per database.
"""

from __future__ import annotations

import sqlite3
from importlib import resources
from pathlib import Path


def open_database(path: Path, migrations_package: str) -> sqlite3.Connection:
    """Connect (creating the file if needed) and apply any new migrations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    _apply_migrations(connection, migrations_package)
    return connection


def _apply_migrations(connection: sqlite3.Connection, package: str) -> None:
    with connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY)"
        )
    applied = {row[0] for row in connection.execute("SELECT version FROM schema_migrations")}
    for version, sql in _migration_scripts(package):
        if version in applied:
            continue
        # executescript commits on its own, so record the version in the same script.
        connection.executescript(
            f"BEGIN;\n{sql}\nINSERT INTO schema_migrations VALUES ({version});\nCOMMIT;"
        )


def _migration_scripts(package: str) -> list[tuple[int, str]]:
    scripts = []
    for item in resources.files(package).iterdir():
        if item.name.endswith(".sql"):
            scripts.append((int(item.name.split("_", 1)[0]), item.read_text(encoding="utf-8")))
    return sorted(scripts)
