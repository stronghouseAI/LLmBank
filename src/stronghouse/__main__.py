"""Command-line entry points.

`stronghouse` with no command starts the app and opens it in a browser, as before.
`stronghouse <command>` and `shai <command>` run the terminal version (cli/commands.py).
"""

from __future__ import annotations

import logging
import sys
import threading
import webbrowser

import uvicorn

from stronghouse.app import build_services, create_app
from stronghouse.config import load_settings

BROWSER_OPEN_DELAY_SECONDS = 1.5


def serve(no_browser: bool = False) -> None:
    """Run the app (web UI and API) in this process until Ctrl+C."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = load_settings()
    app = create_app(build_services(settings))

    url = f"http://{settings.host}:{settings.port}"
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
