"""Shared pieces for skill modules: the error type, path handling and limits."""

from __future__ import annotations

import subprocess
from pathlib import Path

MAX_OUTPUT_CHARS = 8000
PROGRAM_TIMEOUT_SECONDS = 120


class SkillError(RuntimeError):
    """A skill could not run; the message is shown to the agent and the user."""


def resolve_path(raw: object) -> Path:
    """Expand ~ and make the path absolute (relative paths start at the home folder)."""
    path = Path(str(raw)).expanduser()
    return path if path.is_absolute() else Path.home() / path


def truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + f"\n[... output truncated at {MAX_OUTPUT_CHARS} characters]"


def run_program(argv: list[str], input_text: str | None = None) -> str:
    """Run a helper program (pdftotext, wl-copy, xdg-open) and return its output.

    Raises:
        SkillError: if the program is missing, times out or fails.
    """
    try:
        result = subprocess.run(
            argv,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=PROGRAM_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        raise SkillError(f"'{argv[0]}' is not installed.") from error
    except subprocess.TimeoutExpired as error:
        raise SkillError(f"'{argv[0]}' took longer than {PROGRAM_TIMEOUT_SECONDS}s.") from error
    if result.returncode != 0:
        raise SkillError(f"'{argv[0]}' failed: {result.stderr.strip() or result.returncode}")
    return result.stdout
