"""Talk to the StronghouseAI app over its local HTTP API, starting it if needed.

The terminal client is a thin front end: the app process owns the engine, the
downloads and running quests, so a loaded model keeps running after a command ends.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx

from stronghouse.config import Settings

START_TIMEOUT_SECONDS = 20.0
API_TIMEOUT = httpx.Timeout(30.0, read=600.0)  # loading and quests can take a while


class ApiError(RuntimeError):
    """The app answered with an error; the message is the app's own explanation."""


class ApiClient:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def get(self, path: str, **params) -> object:
        return self._check(self._http.get(f"/api{path}", params=params or None))

    def post(self, path: str, body: dict | None = None) -> object:
        return self._check(self._http.post(f"/api{path}", json=body or {}))

    def stream_chat(self, body: dict) -> Iterator[dict]:
        """Yield each streamed delta ({"content": ...} or {"reasoning_content": ...})."""
        with self._http.stream("POST", "/api/chat", json=body) as response:
            if response.status_code >= 400:
                response.read()
                self._check(response)
            for line in response.iter_lines():
                if not line.startswith("data: ") or "[DONE]" in line:
                    continue
                choices = json.loads(line[6:]).get("choices") or [{}]
                yield choices[0].get("delta") or {}

    @staticmethod
    def _check(response: httpx.Response) -> object:
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise ApiError(detail if isinstance(detail, str) else json.dumps(detail))
        return response.json()


def connect(settings: Settings, start_if_needed: bool = True) -> ApiClient:
    """A client for the running app, starting it in the background if it is not up."""
    base_url = f"http://{settings.host}:{settings.port}"
    http = httpx.Client(base_url=base_url, timeout=API_TIMEOUT)
    if _is_up(http):
        return ApiClient(http)
    if not start_if_needed:
        raise ApiError("StronghouseAI is not running. Start it with: shai serve")
    _start_background_server(settings)
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _is_up(http):
            return ApiClient(http)
        time.sleep(0.3)
    raise ApiError(f"StronghouseAI did not start. See {log_path(settings)}")


def stop_background_server(settings: Settings) -> bool:
    """Stop an app the terminal client started. Returns False if none was running."""
    pid_file = pid_path(settings)
    if not pid_file.exists():
        return False
    pid = int(pid_file.read_text().strip())
    pid_file.unlink()
    try:
        os.kill(pid, signal.SIGTERM)  # the app unloads the model on shutdown
    except ProcessLookupError:
        return False
    return True


def pid_path(settings: Settings) -> Path:
    return settings.data_dir / "server.pid"


def log_path(settings: Settings) -> Path:
    return settings.data_dir / "server.log"


def _is_up(http: httpx.Client) -> bool:
    try:
        return http.get("/api/hardware", timeout=2.0).status_code == httpx.codes.OK
    except httpx.HTTPError:
        return False


def _start_background_server(settings: Settings) -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    with log_path(settings).open("ab") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "stronghouse", "serve", "--no-browser"],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # keeps running after this terminal command ends
        )
    pid_path(settings).write_text(str(process.pid))
