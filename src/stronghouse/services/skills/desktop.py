"""Desktop skills: read PDFs, open things in the default app, use the clipboard, run commands.

These call standard Linux tools: pdftotext (poppler), xdg-open and wl-copy
(wl-clipboard, for Wayland desktops such as Hyprland/Omarchy).
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlparse

from stronghouse.services.skills.common import (
    PROGRAM_TIMEOUT_SECONDS,
    SkillError,
    resolve_path,
    run_program,
)

log = logging.getLogger(__name__)

MAX_PDF_PAGES = 60
ProgramRunner = Callable[..., str]


class DesktopSkills:
    def __init__(self, run: ProgramRunner = run_program) -> None:
        self._run = run

    def read_pdf(self, args: dict) -> str:
        path = resolve_path(args["path"])
        if not path.is_file():
            raise SkillError(f"{path} does not exist.")
        try:
            text = self._run(["pdftotext", "-layout", "-l", str(MAX_PDF_PAGES), str(path), "-"])
        except SkillError as error:
            if "not installed" in str(error):
                raise SkillError("Reading PDFs needs poppler: sudo pacman -S poppler") from error
            raise
        return text.strip() or "The PDF has no text layer (it may be scanned images)."

    def open_with_app(self, args: dict) -> str:
        target = str(args["target"]).strip()
        if urlparse(target).scheme not in ("http", "https"):
            path = resolve_path(target)
            if not path.exists():
                raise SkillError(f"{path} does not exist.")
            target = str(path)
        self._run(["xdg-open", target])
        return f"Opened {target}"

    def copy_to_clipboard(self, args: dict) -> str:
        text = str(args["text"])
        try:
            self._run(["wl-copy"], input_text=text)
        except SkillError as error:
            if "not installed" in str(error):
                raise SkillError(
                    "Copying needs wl-clipboard: sudo pacman -S wl-clipboard"
                ) from error
            raise
        return f"Copied {len(text)} characters to the clipboard."


def run_command(args: dict) -> str:
    command = str(args["command"])
    log.info("running approved agent command", extra={"command": command})
    try:
        result = subprocess.run(
            command,
            shell=True,  # the user approved this exact command line
            capture_output=True,
            text=True,
            timeout=PROGRAM_TIMEOUT_SECONDS,
            cwd=Path.home(),
        )
    except subprocess.TimeoutExpired as error:
        raise SkillError(f"Command timed out after {PROGRAM_TIMEOUT_SECONDS}s") from error
    output = (result.stdout + result.stderr).strip()
    return f"exit code {result.returncode}\n{output}"
