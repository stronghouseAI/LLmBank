"""Carry out agent skills on the user's computer.

The quest runner only calls this after the user has approved the exact action
(except take_note, which writes to the agent's own journal). Results are
truncated so one big file or command output cannot flood a small model's context.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable
from pathlib import Path

log = logging.getLogger(__name__)

MAX_READ_BYTES = 64 * 1024
MAX_OUTPUT_CHARS = 8000
MAX_LISTED_ENTRIES = 200
MAX_SEARCH_RESULTS = 50
MAX_SEARCH_FILE_BYTES = 2 * 1024 * 1024
COMMAND_TIMEOUT_SECONDS = 120
_SKIPPED_DIRS = {".git", "node_modules", ".venv", "__pycache__", ".cache"}


class SkillError(RuntimeError):
    """A skill could not run; the message is shown to the agent and the user."""


class SkillExecutor:
    def __init__(self, add_note: Callable[[int, str], None]) -> None:
        self._add_note = add_note

    def execute(self, agent_id: int, skill: str, arguments: dict) -> str:
        """Run one skill and return its text result.

        Raises:
            SkillError: for bad arguments or a failed action.
        """
        handlers: dict[str, Callable[[dict], str]] = {
            "read_file": self._read_file,
            "list_folder": self._list_folder,
            "search_files": self._search_files,
            "write_file": self._write_file,
            "run_command": self._run_command,
            "take_note": lambda args: self._take_note(agent_id, args),
        }
        handler = handlers.get(skill)
        if handler is None:
            raise SkillError(f"Unknown skill: {skill}")
        try:
            return _truncate(handler(arguments))
        except KeyError as error:
            raise SkillError(f"Missing argument: {error.args[0]}") from error
        except OSError as error:
            raise SkillError(str(error)) from error

    @staticmethod
    def _read_file(args: dict) -> str:
        path = resolve_path(args["path"])
        with path.open("rb") as stream:
            data = stream.read(MAX_READ_BYTES + 1)
        text = data[:MAX_READ_BYTES].decode("utf-8", errors="replace")
        if len(data) > MAX_READ_BYTES:
            text += f"\n[... file truncated after {MAX_READ_BYTES} bytes]"
        return text

    @staticmethod
    def _list_folder(args: dict) -> str:
        folder = resolve_path(args["path"])
        entries = sorted(folder.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        lines = [f"{p.name}/" if p.is_dir() else p.name for p in entries[:MAX_LISTED_ENTRIES]]
        if len(entries) > MAX_LISTED_ENTRIES:
            lines.append(f"[... and {len(entries) - MAX_LISTED_ENTRIES} more]")
        return "\n".join(lines) or "(empty folder)"

    @staticmethod
    def _search_files(args: dict) -> str:
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

    @staticmethod
    def _write_file(args: dict) -> str:
        path = resolve_path(args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(args["content"]), encoding="utf-8")
        return f"Wrote {len(str(args['content']))} characters to {path}"

    @staticmethod
    def _run_command(args: dict) -> str:
        command = str(args["command"])
        log.info("running approved agent command", extra={"command": command})
        try:
            result = subprocess.run(
                command,
                shell=True,  # the user approved this exact command line
                capture_output=True,
                text=True,
                timeout=COMMAND_TIMEOUT_SECONDS,
                cwd=Path.home(),
            )
        except subprocess.TimeoutExpired as error:
            raise SkillError(f"Command timed out after {COMMAND_TIMEOUT_SECONDS}s") from error
        output = (result.stdout + result.stderr).strip()
        return f"exit code {result.returncode}\n{output}"

    def _take_note(self, agent_id: int, args: dict) -> str:
        note = str(args["note"]).strip()
        if not note:
            raise SkillError("The note is empty.")
        self._add_note(agent_id, note)
        return "Noted."


def resolve_path(raw: object) -> Path:
    """Expand ~ and make the path absolute (relative paths start at the home folder)."""
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else Path.home() / path


def _file_contains(path: Path, query: str) -> bool:
    try:
        if path.stat().st_size > MAX_SEARCH_FILE_BYTES:
            return False
        return query in path.read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n[... output truncated at {MAX_OUTPUT_CHARS} characters]"
