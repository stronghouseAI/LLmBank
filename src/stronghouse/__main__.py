"""Command-line entry point: `stronghouse` starts the app and opens it in a browser."""

from __future__ import annotations

import argparse
import logging
import threading
import webbrowser

import uvicorn

from stronghouse.app import build_services, create_app
from stronghouse.config import load_settings

BROWSER_OPEN_DELAY_SECONDS = 1.5


def main() -> None:
    parser = argparse.ArgumentParser(prog="stronghouse", description="StronghouseAI model studio")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    settings = load_settings()
    app = create_app(build_services(settings))

    url = f"http://{settings.host}:{settings.port}"
    if not args.no_browser:
        threading.Timer(BROWSER_OPEN_DELAY_SECONDS, webbrowser.open, args=(url,)).start()
    print(f"StronghouseAI running at {url}  (archive: {settings.archive_dir})")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
