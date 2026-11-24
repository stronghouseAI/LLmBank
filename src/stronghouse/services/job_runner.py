"""Run background jobs one at a time on a worker thread.

For each job: make sure its model is loaded (loading it with Auto-fit if needed),
stream the answer into a Markdown file as it is generated, then send a desktop
notification. The app keeps running in the background (`shai` starts it that way),
so jobs carry on after the browser or terminal is closed.
See docs/adr/0011-big-models-on-small-machines.md.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import httpx

from stronghouse.core.jobs import (
    Job,
    JobState,
    job_title,
    notification_text,
    output_filename,
    output_header,
    split_sse,
    thinking_section,
)
from stronghouse.core.models import PlanSettings
from stronghouse.services.engine import EngineError, EngineState, EngineStatus
from stronghouse.storage.job_store import JobStore, now

log = logging.getLogger(__name__)

IDLE_WAIT_SECONDS = 5.0
LOAD_POLL_SECONDS = 2.0
MAX_PROMPT_LENGTH = 200_000


class JobEngine(Protocol):
    def status(self) -> EngineStatus: ...
    def stream_chat(self, payload: dict) -> Iterator[bytes]: ...


class JobError(ValueError):
    """The job request or action is not valid, or the job cannot run."""


class _Cancelled(Exception):
    pass


# Called with a job; starts loading its model if another one is loaded and returns
# the model's file name. Raises JobError if the model cannot be found or planned.
ModelLoader = Callable[[Job], str]
Notify = Callable[[str, str], None]


class JobRunner:
    def __init__(
        self,
        store: JobStore,
        engine: JobEngine,
        load_model: ModelLoader,
        notify: Notify,
        output_dir: Path,
        load_timeout_seconds: float = 3600.0,
    ) -> None:
        self._store = store
        self._engine = engine
        self._load_model = load_model
        self._notify = notify
        self._output_dir = output_dir
        self._load_timeout = load_timeout_seconds
        self._wake = threading.Event()
        self._stopping = threading.Event()
        self._cancel_lock = threading.Lock()
        self._cancel_ids: set[int] = set()
        self._thread: threading.Thread | None = None

    @property
    def store(self) -> JobStore:
        return self._store

    # ----- control -----

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stopping.clear()
            self._thread = threading.Thread(target=self._loop, name="job-runner", daemon=True)
            self._thread.start()

    def stop(self, wait_seconds: float = 0.0) -> None:
        """Stop taking new jobs. A job already generating finishes in the background."""
        self._stopping.set()
        self._wake.set()
        if wait_seconds and self._thread is not None:
            self._thread.join(wait_seconds)

    def submit(
        self,
        prompt: str,
        model_id: int | None,
        plan_settings: PlanSettings,
        think: bool = True,
        max_tokens: int | None = None,
    ) -> Job:
        """Queue a prompt. It starts as soon as earlier jobs are finished.

        Raises:
            JobError: if the prompt is empty or too long.
        """
        prompt = prompt.strip()
        if not prompt:
            raise JobError("Write a prompt for the job.")
        if len(prompt) > MAX_PROMPT_LENGTH:
            raise JobError(f"The prompt is longer than {MAX_PROMPT_LENGTH:,} characters.")
        job = self._store.add(
            Job(
                prompt=prompt,
                model_id=model_id,
                plan_settings=plan_settings,
                think=think,
                max_tokens=max_tokens,
                state=JobState.QUEUED,
                created_at=now(),
                title=job_title(prompt),
            )
        )
        log.info("job queued", extra={"job_id": job.id, "model_id": model_id})
        self._wake.set()
        return job

    def cancel(self, job_id: int) -> Job:
        """Stop a running job, or take a queued one out of the queue.

        Raises:
            JobError: if there is no such job or it has already finished.
        """
        job = self._require(job_id)
        if job.state.finished:
            raise JobError("This job has already finished.")
        if job.state is JobState.QUEUED:
            job = replace(job, state=JobState.CANCELLED, finished_at=now())
            self._store.save(job)
            return job
        with self._cancel_lock:
            self._cancel_ids.add(job_id)
        return job

    def retry(self, job_id: int) -> Job:
        """Queue a finished job's prompt again, as a new job.

        Raises:
            JobError: if there is no such job or it is still queued or running.
        """
        job = self._require(job_id)
        if not job.state.finished:
            raise JobError("This job has not finished yet.")
        return self.submit(job.prompt, job.model_id, job.plan_settings, job.think, job.max_tokens)

    def read_output(self, job: Job) -> str:
        """The answer file's text so far, or "" if nothing has been written yet."""
        if not job.output_path:
            return ""
        try:
            return Path(job.output_path).read_text(encoding="utf-8")
        except OSError:
            return ""

    # ----- worker -----

    def run_next(self) -> bool:
        """Run the oldest queued job to the end. Returns False when the queue is empty."""
        job = self._store.next_queued()
        if job is None:
            return False
        finished = self._run(job)
        summary, body = notification_text(finished)
        self._notify(summary, body)
        return True

    def _loop(self) -> None:
        while not self._stopping.is_set():
            try:
                worked = self.run_next()
            except Exception:  # the worker must survive any one bad job
                log.exception("job runner error")
                worked = False
            if not worked:
                self._wake.wait(IDLE_WAIT_SECONDS)
                self._wake.clear()

    def _run(self, job: Job) -> Job:
        job = replace(job, state=JobState.LOADING, started_at=now())
        self._store.save(job)
        try:
            model_name = self._load_model(job)
            self._wait_until_ready(job)
            job = self._generate(replace(job, state=JobState.RUNNING), model_name)
            job = replace(job, state=JobState.DONE)
        except _Cancelled:
            job = replace(job, state=JobState.CANCELLED)
        except (JobError, EngineError) as error:
            job = replace(job, state=JobState.FAILED, error=str(error))
        except httpx.HTTPError as error:
            message = f"The engine stopped answering: {error}"
            job = replace(job, state=JobState.FAILED, error=message)
        except OSError as error:
            job = replace(job, state=JobState.FAILED, error=f"Could not write the answer: {error}")
        job = replace(job, finished_at=now())
        self._store.save(job)
        with self._cancel_lock:
            self._cancel_ids.discard(job.id)  # type: ignore[arg-type]
        log.info("job finished", extra={"job_id": job.id, "state": job.state.value})
        return job

    def _wait_until_ready(self, job: Job) -> None:
        deadline = time.monotonic() + self._load_timeout
        while True:
            self._check_cancel(job)
            status = self._engine.status()
            if status.state is EngineState.READY:
                return
            if status.state in (EngineState.CRASHED, EngineState.STOPPED):
                last = status.log_tail[-1] if status.log_tail else "no log"
                raise JobError(f"The model did not load (engine {status.state.value}): {last}")
            if time.monotonic() > deadline:
                raise JobError("The model was still loading after the time limit.")
            time.sleep(LOAD_POLL_SECONDS)

    def _generate(self, job: Job, model_name: str) -> Job:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        path = self._output_dir / output_filename(job.id or 0, job.title)
        job = replace(job, output_path=str(path))
        self._store.save(job)
        payload: dict = {"messages": [{"role": "user", "content": job.prompt}]}
        if job.max_tokens:
            payload["max_tokens"] = job.max_tokens
        if not job.think:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        thinking: list[str] = []
        tokens, speed, buffer = 0, None, ""
        stream = self._engine.stream_chat(payload)
        with path.open("w", encoding="utf-8") as out, closing(stream):  # type: ignore[type-var]
            out.write(output_header(job, model_name))
            out.flush()
            for chunk in stream:
                self._check_cancel(job)
                events, buffer = split_sse(buffer + chunk.decode("utf-8", errors="replace"))
                for event in events:
                    if "error" in event:
                        raise JobError(str(event["error"].get("message", event["error"])))
                    delta = (event.get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content"):
                        out.write(delta["content"])
                        out.flush()  # so `tail -f` and the Jobs page see progress
                    if delta.get("reasoning_content"):
                        thinking.append(delta["reasoning_content"])
                    timings = event.get("timings") or {}
                    tokens = timings.get("predicted_n", tokens)
                    speed = timings.get("predicted_per_second", speed)
            out.write(thinking_section("".join(thinking)))
        return replace(
            job,
            generated_tokens=int(tokens or 0),
            tokens_per_second=round(speed, 2) if speed else None,
        )

    def _check_cancel(self, job: Job) -> None:
        with self._cancel_lock:
            if job.id in self._cancel_ids:
                raise _Cancelled

    def _require(self, job_id: int) -> Job:
        job = self._store.get(job_id)
        if job is None:
            raise JobError(f"No job with id {job_id}.")
        return job
