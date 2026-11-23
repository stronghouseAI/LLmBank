"""Carry out agent skills. The quest runner calls this only after the user approved
the exact action, except for skills marked needs_approval=False in the catalog.

Results are truncated so one big file or page cannot flood a small model's context.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx

from stronghouse.core.skill_catalog import SKILLS
from stronghouse.services.skills import files, utility
from stronghouse.services.skills.common import SkillError, resolve_path, run_program, truncate
from stronghouse.services.skills.desktop import DesktopSkills, run_command
from stronghouse.services.skills.web import WebSkills

__all__ = ["SkillError", "SkillExecutor", "resolve_path"]

WEB_TIMEOUT = httpx.Timeout(20.0)
# The app names itself honestly first; WebSkills retries with browser headers only
# when a site refuses (see ADR 0006).
_APP_HEADERS = {"User-Agent": "StronghouseAI/0.1", "Accept": "*/*"}


class SkillExecutor:
    def __init__(
        self,
        add_note: Callable[[int, str], None],
        http: httpx.Client | None = None,
        run: Callable[..., str] = run_program,
    ) -> None:
        self._add_note = add_note
        web = WebSkills(
            http or httpx.Client(timeout=WEB_TIMEOUT, follow_redirects=True, headers=_APP_HEADERS)
        )
        desktop = DesktopSkills(run)
        self._handlers: dict[str, Callable[[dict], str]] = {
            "read_file": files.read_file,
            "list_folder": files.list_folder,
            "search_files": files.search_files,
            "write_file": files.write_file,
            "move_file": files.move_file,
            "fetch_web_page": web.fetch_web_page,
            "read_news_feed": web.read_news_feed,
            "web_search": web.web_search,
            "read_pdf": desktop.read_pdf,
            "open_with_app": desktop.open_with_app,
            "copy_to_clipboard": desktop.copy_to_clipboard,
            "run_command": run_command,
            "calculator": utility.calculator,
            "current_datetime": utility.current_datetime,
        }
        missing = set(SKILLS) - set(self._handlers) - {"take_note"}
        if missing:  # a catalog entry without code would only fail mid-quest
            raise RuntimeError(f"Skills without a handler: {sorted(missing)}")

    def execute(self, agent_id: int, skill: str, arguments: dict) -> str:
        """Run one skill and return its text result.

        Raises:
            SkillError: for bad arguments or a failed action.
        """
        if skill == "take_note":
            return self._take_note(agent_id, arguments)
        handler = self._handlers.get(skill)
        if handler is None:
            raise SkillError(f"Unknown skill: {skill}")
        try:
            return truncate(handler(arguments))
        except KeyError as error:
            raise SkillError(f"Missing argument: {error.args[0]}") from error
        except OSError as error:
            raise SkillError(str(error)) from error

    def _take_note(self, agent_id: int, args: dict) -> str:
        note = str(args.get("note", "")).strip()
        if not note:
            raise SkillError("The note is empty.")
        self._add_note(agent_id, note)
        return "Noted."
