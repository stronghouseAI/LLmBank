"""Desktop notifications through notify-send (libnotify).

On Omarchy and most Linux desktops a notification daemon (mako, dunst, swaync)
shows these. Without notify-send nothing happens: notifications are a convenience,
never a reason for a job to fail.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from collections.abc import Callable

log = logging.getLogger(__name__)
NOTIFY_TIMEOUT_SECONDS = 5
APP_NAME = "StronghouseAI"

CommandRunner = Callable[[list[str]], None]


def notification_command(summary: str, body: str) -> list[str]:
    return ["notify-send", "--app-name", APP_NAME, summary, body]


class DesktopNotifier:
    def __init__(self, enabled: bool = True, run_command: CommandRunner | None = None) -> None:
        self._enabled = enabled
        self._run = run_command or _run

    def notify(self, summary: str, body: str) -> None:
        if not self._enabled:
            return
        if self._run is _run and shutil.which("notify-send") is None:
            log.info("notify-send not found; skipping desktop notification")
            return
        try:
            self._run(notification_command(summary, body))
        except (OSError, subprocess.SubprocessError) as error:
            log.warning("desktop notification failed", extra={"error": str(error)})


def _run(command: list[str]) -> None:
    subprocess.run(command, check=False, capture_output=True, timeout=NOTIFY_TIMEOUT_SECONDS)
