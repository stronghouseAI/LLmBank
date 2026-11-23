"""File skills: read, list, search, write and move. All need the user's approval."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from stronghouse.services.skills.common import SkillError, resolve_path

log = logging.getLogger(__name__)

MAX_READ_BYTES = 64 * 1024
MAX_LISTED_ENTRIES = 200
MAX_SEARCH_RESULTS = 50
MAX_SEARCH_FILE_BYTES = 2 * 1024 * 1024
_SKIPPED_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".cache"}


def read_file(args: dict) -> str:
    path = resolve_path(args["path"])
    with path.open("rb") as stream:
        data = stream.read(MAX_READ_BYTES + 1)
    if path.suffix.lower() == ".pdf" or data.startswith(b"%PDF"):
        raise SkillError("This is a PDF. Use the Read PDF skill instead.")
    text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
    if len(data) > MAX_READ_BYTES:
        text += f"\n[... file truncated after {MAX_READ_BYTES} bytes]"
    return text


def list_folder(args: dict) -> str:
    folder = resolve_path(args["path"])
    entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    lines = [f"{p.name}/" if p.is_dir() else p.name for p in entries[:MAX_LISTED_ENTRIES]]
    if len(entries) > MAX_LISTED_ENTRIES:
        lines.append(f"[... and {len(entries) - MAX_LISTED_ENTRIES} more]")
    return "\n".join(lines) or "(empty folder)"


def search_files(args: dict) -> str:
    folder = resolve_path(args["folder"])
    query = str(args["query"]).lower()
    if not query:
        raise SkillError("The search query is empty.")
    matches: list[str] = []
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in _SKIPPED_DIRS]
        for name in files:
            path = Path(root) / name
            if query in name.lower() or _file_contains(path, query):
                matches.append(str(path))
                if len(matches) >= MAX_SEARCH_RESULTS:
                    return "\n".join(matches) + "\n[... more results not shown]"
    return "\n".join(matches) or "No matches."


def write_file(args: dict) -> str:
    path = resolve_path(args["path"])
    content = str(args["content"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Wrote {len(content)} characters to {path}"


def move_file(args: dict) -> str:
    source = resolve_path(args["source"])
    destination = resolve_path(args["destination"])
    if not source.exists():
        raise SkillError(f"{source} does not exist.")
    if destination.is_dir():
        destination = destination / source.name
    if destination.exists():
        # Never overwrite: an agent mistake here would silently lose a file.
        raise SkillError(f"{destination} already exists. Choose another name.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return f"Moved {source} to {destination}"


def _file_contains(path: Path, query: str) -> bool:
    try:
        if path.stat().st_size > MAX_SEARCH_FILE_BYTES:
            return False
        return query in path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False
