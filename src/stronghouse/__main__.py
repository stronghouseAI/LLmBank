"""Command-line entry points.

`stronghouse` with no command starts the app and opens it in a browser. If the app is
already running (for example started in the background by `shai`), it just opens it.
`stronghouse <command>` and `shai <command>` run the terminal version (cli/commands.py).
"""

from __future__ import annotations

import logging
import sys
import threading
import webbrowser

import uvicorn

from stronghouse import __version__
from stronghouse.app import build_services, create_app
from stronghouse.cli.client import app_is_running
from stronghouse.config import load_settings
from stronghouse.services.log_files import configure_logging

BROWSER_OPEN_DELAY_SECONDS = 1.5


def serve(no_browser: bool = False) -> None:
    """Run the app (web UI and API) in this process until Ctrl+C, or open the running one."""
    settings = load_settings()
    url = f"http://{settings.host}:{settings.port}"
    if app_is_running(settings):  # e.g. shai started it in the background: reuse it
        print(f"StronghouseAI is already running at {url}  (stop it with: shai stop)")
        if not no_browser:
            webbrowser.open(url)
        return
    configure_logging(settings)
    logging.getLogger(__name__).info(
        "StronghouseAI %s starting",
        __version__,
        extra={"url": url, "data_dir": str(settings.data_dir)},
    )
    app = create_app(build_services(settings))

    if not no_browser:
        threading.Timer(BROWSER_OPEN_DELAY_SECONDS, webbrowser.open, args=(url,)).start()
    print(f"StronghouseAI running at {url}  (archive: {settings.archive_dir})")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


def main() -> None:
    argv = sys.argv[1:]
    if not argv or argv == ["--no-browser"]:
        serve(no_browser=bool(argv))
        return
    from stronghouse.cli.commands import run

    raise SystemExit(run(argv, load_settings()))


if __name__ == "__main__":
    main()
