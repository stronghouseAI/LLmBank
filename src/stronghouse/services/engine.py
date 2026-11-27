"""Start, stop and talk to one llama-server process (the inference engine).

StronghouseAI does not do inference itself; llama.cpp does. See
docs/adr/0001-wrap-llama-cpp.md for why.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx

from stronghouse.core.agents import ModelTurn, ToolCall
from stronghouse.core.launch_args import build_server_args
from stronghouse.core.models import LaunchPlan

log = logging.getLogger(__name__)

VERSION_TIMEOUT_SECONDS = 10
# Starting each GPU backend to list devices can take a few seconds on first run.
LIST_DEVICES_TIMEOUT_SECONDS = 20
STOP_TIMEOUT_SECONDS = 15
LOG_TAIL_LINES = 40
BENCHMARK_PROMPT = "Write a short paragraph explaining how a mixture-of-experts model works."
BENCHMARK_TOKENS = 128


class EngineState(StrEnum):
    STOPPED = "stopped"
    LOADING = "loading"
    READY = "ready"
    CRASHED = "crashed"


class EngineError(RuntimeError):
    """The engine is not running, not ready, or could not be started."""


@dataclass(frozen=True)
class EngineStatus:
    state: EngineState
    model_path: str | None
    plan: LaunchPlan | None
    command: list[str]
    log_tail: list[str]


@dataclass(frozen=True)
class BenchmarkResult:
    prompt_tokens_per_second: float
    generated_tokens_per_second: float
    generated_tokens: int


class EngineController:
    def __init__(
        self,
        executable: str,
        host: str,
        port: int,
        log_path: Path,
        http: httpx.Client | None = None,
    ) -> None:
        self._executable = executable
        self._host = host
        self._port = port
        self._log_path = log_path
        self._http = http or httpx.Client(timeout=httpx.Timeout(10.0, read=600.0))
        self._process: subprocess.Popen[bytes] | None = None
        self._model_path: str | None = None
        self._plan: LaunchPlan | None = None
        self._command: list[str] = []
        self._lock = threading.Lock()
        self._cached_version: str | None = None
        self._cached_devices: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self._host}:{self._port}"

    def version(self) -> str | None:
        """llama-server's version line, or None if it is not installed."""
        if self._cached_version is None:
            try:
                result = subprocess.run(
                    [self._executable, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=VERSION_TIMEOUT_SECONDS,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            lines = (result.stdout + result.stderr).strip().splitlines()
            version_lines = [line for line in lines if "version" in line.lower()]
            self._cached_version = (version_lines or lines or [""])[0].strip() or None
        return self._cached_version

    def list_devices(self) -> str | None:
        """`llama-server --list-devices` output (which GPUs and backend it can use).

        Asked once and then remembered, because the build does not change while the
        app runs. Returns None if llama-server is missing or does not know the flag.
        """
        if self._cached_devices is None:
            try:
                result = subprocess.run(
                    [self._executable, "--list-devices"],
                    capture_output=True,
                    text=True,
                    timeout=LIST_DEVICES_TIMEOUT_SECONDS,
                )
            except (OSError, subprocess.SubprocessError):
                return None
            if result.returncode != 0:
                log.info("llama-server --list-devices failed", extra={"code": result.returncode})
                return None
            self._cached_devices = result.stdout + result.stderr
        return self._cached_devices

    def start(self, model_path: Path, plan: LaunchPlan) -> EngineStatus:
        """Launch llama-server with the plan, replacing any running model.

        Returns immediately; poll status() until READY. Large models take a while.

        Raises:
            EngineError: if the executable cannot be started.
        """
        self.stop()
        command = build_server_args(self._executable, model_path, plan, self._host, self._port)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        log.info("starting engine", extra={"model": str(model_path), "command": command})
        with self._lock:
            try:
                with self._log_path.open("wb") as log_file:
                    self._process = subprocess.Popen(
                        command, stdout=log_file, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL
                    )
            except OSError as error:
                raise EngineError(
                    f"Could not start {self._executable}. Is llama.cpp installed? ({error})"
                ) from error
            self._model_path, self._plan, self._command = str(model_path), plan, command
        return self.status()

    def stop(self) -> None:
        with self._lock:
            process, self._process = self._process, None
            self._model_path, self._plan, self._command = None, None, []
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            log.warning("engine did not stop in time; killing it")
            process.kill()
            process.wait()

    def status(self) -> EngineStatus:
        with self._lock:
            process = self._process
            snapshot = (self._model_path, self._plan, list(self._command))
        return EngineStatus(self._state(process), *snapshot, log_tail=self._log_tail())

    def benchmark(self) -> BenchmarkResult:
        """Generate a fixed number of tokens and report llama-server's own timings."""
        self._require_ready()
        response = self._http.post(
            f"{self.base_url}/completion",
            json={"prompt": BENCHMARK_PROMPT, "n_predict": BENCHMARK_TOKENS, "cache_prompt": False},
        )
        response.raise_for_status()
        timings = response.json().get("timings", {})
        return BenchmarkResult(
            prompt_tokens_per_second=round(timings.get("prompt_per_second", 0.0), 2),
            generated_tokens_per_second=round(timings.get("predicted_per_second", 0.0), 2),
            generated_tokens=timings.get("predicted_n", 0),
        )

    def complete_chat(
        self, messages: list[dict], tools: list[dict], temperature: float
    ) -> ModelTurn:
        """One non-streaming chat turn with tool definitions, for agents.

        Raises:
            EngineError: if no model is ready.
            httpx.HTTPError: if the engine fails mid-request.
        """
        self._require_ready()
        payload: dict = {"messages": messages, "temperature": temperature}
        if tools:
            payload["tools"] = tools
        response = self._http.post(f"{self.base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        calls = [
            ToolCall(
                id=call.get("id") or f"call_{index}",
                name=call["function"]["name"],
                arguments=call["function"].get("arguments") or "{}",
            )
            for index, call in enumerate(message.get("tool_calls") or [])
        ]
        return ModelTurn(content=message.get("content") or "", tool_calls=calls)

    def render_prompt(self, payload: dict) -> str:
        """The exact text the model reads for this request, from llama-server's own
        chat template (its /apply-template endpoint). For `shai chat --debug`.

        Raises:
            EngineError: if no model is ready.
            httpx.HTTPError: if the engine refuses or fails.
        """
        self._require_ready()
        response = self._http.post(f"{self.base_url}/apply-template", json=payload)
        response.raise_for_status()
        return str(response.json().get("prompt", ""))

    def stream_chat(self, payload: dict) -> Iterator[bytes]:
        """Forward an OpenAI-style chat request and stream the raw SSE bytes back."""
        self._require_ready()
        with self._http.stream(
            "POST", f"{self.base_url}/v1/chat/completions", json={**payload, "stream": True}
        ) as response:
            response.raise_for_status()
            yield from response.iter_bytes()

    def _require_ready(self) -> None:
        if self.status().state is not EngineState.READY:
            raise EngineError("No model is loaded and ready yet.")

    def _state(self, process: subprocess.Popen[bytes] | None) -> EngineState:
        if process is None:
            return EngineState.STOPPED
        if process.poll() is not None:
            return EngineState.CRASHED
        try:
            health = self._http.get(f"{self.base_url}/health", timeout=2.0)
        except httpx.HTTPError:
            return EngineState.LOADING
        return EngineState.READY if health.status_code == httpx.codes.OK else EngineState.LOADING

    def _log_tail(self) -> list[str]:
        if not self._log_path.exists():
            return []
        lines = self._log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-LOG_TAIL_LINES:]
