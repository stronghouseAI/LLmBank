"""Background jobs: queue a prompt, walk away, find the answer in a file."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stronghouse.app import build_services, create_app
from stronghouse.config import load_settings
from stronghouse.core.jobs import (
    Job,
    JobState,
    job_title,
    notification_text,
    output_filename,
    split_sse,
)
from stronghouse.core.models import HardwareProfile, PlanSettings
from stronghouse.services.engine import EngineState, EngineStatus
from stronghouse.services.job_runner import JobError, JobRunner
from stronghouse.services.notifier import DesktopNotifier, notification_command
from stronghouse.storage.job_store import INTERRUPTED_ERROR, JobStore
from tests.gguf_builder import moe_model_spec, write_gguf

GIB = 1024**3


def sse(event: dict) -> bytes:
    return f"data: {json.dumps(event)}\n\n".encode()


class FakeEngine:
    """Ready at once; streams thinking, an answer split across chunks, and timings."""

    def __init__(self, state: EngineState = EngineState.READY, model_path: str | None = "/m.gguf"):
        self.state = state
        self.model_path = model_path
        self.payloads: list[dict] = []
        self.started = None

    def status(self):
        return EngineStatus(self.state, self.model_path, None, [], ["last log line"])

    def start(self, model_path, plan):
        self.started = (model_path, plan)
        self.model_path, self.state = str(model_path), EngineState.READY
        return self.status()

    def stop(self):
        self.state = EngineState.STOPPED

    def version(self):
        return "test"

    def stream_chat(self, payload):
        self.payloads.append(payload)
        yield sse({"choices": [{"delta": {"reasoning_content": "Let me think."}}]})
        answer = sse({"choices": [{"delta": {"content": "Slow power "}}]})
        yield answer[:20]  # an event split across two network chunks
        yield answer[20:]
        yield sse({"choices": [{"delta": {"content": "is better than no power."}}]})
        timings = {"predicted_n": 9, "predicted_per_second": 0.42}
        yield sse({"choices": [{"delta": {}}], "timings": timings})
        yield b"data: [DONE]\n\n"


def make_runner(tmp_path: Path, engine=None, loader=None):
    notes: list[tuple[str, str]] = []
    runner = JobRunner(
        JobStore(tmp_path / "jobs.sqlite3"),
        engine or FakeEngine(),
        load_model=loader or (lambda job: "glm-4.5-air-Q2_K.gguf"),
        notify=lambda summary, body: notes.append((summary, body)),
        output_dir=tmp_path / "out",
        load_timeout_seconds=0,
    )
    return runner, notes


# ----- core -----


def test_split_sse_keeps_unfinished_lines_for_later():
    events, rest = split_sse('data: {"a": 1}\n\ndata: [DONE]\n\ndata: {"b"')
    assert events == [{"a": 1}]
    assert rest == 'data: {"b"'


def test_titles_and_file_names_are_short_and_safe():
    title = job_title("Write   a\nplan " + "x" * 100)
    assert title.startswith("Write a plan") and len(title) <= 60
    assert output_filename(7, "What's 2/3? ../etc") == "0007-what-s-2-3-etc.md"
    assert output_filename(1, "???") == "0001-job.md"


def test_notification_text_by_state():
    job = Job("p", None, PlanSettings(), True, None, JobState.DONE, "t", title="T",
              generated_tokens=5, tokens_per_second=0.5)  # fmt: skip
    assert notification_text(job) == ("StronghouseAI job done", "T\n5 tokens at 0.5 tokens/s")
    failed = Job("p", None, PlanSettings(), True, None, JobState.FAILED, "t", title="T", error="x")
    assert notification_text(failed)[0] == "StronghouseAI job failed"


# ----- runner -----


def test_job_writes_answer_file_and_notifies(tmp_path):
    runner, notes = make_runner(tmp_path)
    job = runner.submit("Is slow power better than no power?", None, PlanSettings(), think=False)
    assert runner.run_next() is True
    done = runner.store.get(job.id)
    assert done.state is JobState.DONE
    assert done.generated_tokens == 9 and done.tokens_per_second == 0.42
    text = Path(done.output_path).read_text()
    assert "Slow power is better than no power." in text
    assert "> Is slow power better than no power?" in text
    assert "<summary>Thinking</summary>" in text and "Let me think." in text
    assert notes == [("StronghouseAI job done", f"{done.title}\n9 tokens at 0.42 tokens/s")]
    assert runner.run_next() is False  # queue is empty


def test_think_off_and_max_tokens_reach_the_engine(tmp_path):
    engine = FakeEngine()
    runner, _ = make_runner(tmp_path, engine)
    runner.submit("Hi", None, PlanSettings(), think=False, max_tokens=50)
    runner.run_next()
    assert engine.payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert engine.payloads[0]["max_tokens"] == 50


def test_jobs_run_in_the_order_queued(tmp_path):
    runner, _ = make_runner(tmp_path)
    first = runner.submit("first", None, PlanSettings())
    second = runner.submit("second", None, PlanSettings())
    runner.run_next()
    assert runner.store.get(first.id).state is JobState.DONE
    assert runner.store.get(second.id).state is JobState.QUEUED


def test_model_that_fails_to_load_fails_the_job_with_the_log(tmp_path):
    runner, notes = make_runner(tmp_path, FakeEngine(state=EngineState.CRASHED))
    job = runner.submit("Hi", None, PlanSettings())
    runner.run_next()
    failed = runner.store.get(job.id)
    assert failed.state is JobState.FAILED and "last log line" in failed.error
    assert notes[0][0] == "StronghouseAI job failed"


def test_loader_errors_fail_the_job(tmp_path):
    def no_model(_job):
        raise JobError("No model is loaded.")

    runner, _ = make_runner(tmp_path, loader=no_model)
    job = runner.submit("Hi", None, PlanSettings())
    runner.run_next()
    assert runner.store.get(job.id).error == "No model is loaded."


def test_cancel_queued_and_retry(tmp_path):
    runner, _ = make_runner(tmp_path)
    job = runner.submit("Hi", None, PlanSettings())
    assert runner.cancel(job.id).state is JobState.CANCELLED
    assert runner.run_next() is False
    with pytest.raises(JobError, match="already finished"):
        runner.cancel(job.id)
    again = runner.retry(job.id)
    assert again.id != job.id and again.state is JobState.QUEUED


def test_cancel_while_running_stops_the_job(tmp_path):
    engine = FakeEngine()
    runner, _ = make_runner(tmp_path, engine)
    job = runner.submit("Hi", None, PlanSettings())
    original = engine.stream_chat

    def cancel_midway(payload):
        for index, chunk in enumerate(original(payload)):
            if index == 1:
                runner.cancel(job.id)
            yield chunk

    engine.stream_chat = cancel_midway
    runner.run_next()
    assert runner.store.get(job.id).state is JobState.CANCELLED


def test_empty_prompt_is_refused(tmp_path):
    runner, _ = make_runner(tmp_path)
    with pytest.raises(JobError):
        runner.submit("   ", None, PlanSettings())


def test_jobs_running_when_the_app_stopped_are_marked_failed(tmp_path):
    runner, _ = make_runner(tmp_path)
    job = runner.submit("Hi", None, PlanSettings(context=4096))
    runner.store.save(replace(job, state=JobState.RUNNING))
    reopened = JobStore(tmp_path / "jobs.sqlite3").get(job.id)
    assert reopened.state is JobState.FAILED and reopened.error == INTERRUPTED_ERROR
    assert reopened.plan_settings == PlanSettings(context=4096)


def test_notifier_calls_notify_send_and_can_be_turned_off():
    calls = []
    DesktopNotifier(True, run_command=calls.append).notify("Done", "Body")
    assert calls == [notification_command("Done", "Body")]
    DesktopNotifier(False, run_command=calls.append).notify("Done", "Body")
    assert len(calls) == 1


def test_notifier_never_raises():
    def broken(_command):
        raise OSError("no notification daemon")

    DesktopNotifier(True, run_command=broken).notify("Done", "Body")


# ----- API, with the real model loader -----


@pytest.fixture
def app(tmp_path: Path):
    settings = load_settings({"STRONGHOUSE_DATA_DIR": str(tmp_path), "STRONGHOUSE_NOTIFY": "0"})
    services = build_services(settings)
    services.engine = FakeEngine(state=EngineState.STOPPED, model_path=None)
    services.probe_hardware = lambda: HardwareProfile("GPU", 8 * GIB, 8 * GIB, 16 * GIB,
                                                      12 * GIB, 20)  # fmt: skip
    services.jobs = JobRunner(
        services.jobs.store, services.engine, services.load_model_for_job,
        notify=lambda *_: None, output_dir=settings.jobs_output_dir,
    )  # fmt: skip
    folder = settings.archive_dir / "zai-org" / "GLM-4.5-Air-GGUF"
    folder.mkdir(parents=True)
    write_gguf(folder / "GLM-4.5-Air-Q4_K_M.gguf", *moe_model_spec(blocks=4))
    services.indexer.rescan()
    with TestClient(create_app(services)) as client:
        services.jobs.stop(wait_seconds=5)  # the tests drive the queue themselves with run_next()
        yield client, services


def test_api_job_loads_its_model_then_answers(app):
    client, services = app
    model_id = client.get("/api/library").json()[0]["id"]
    response = client.post(
        "/api/jobs", json={"prompt": "Plan my week", "model_id": model_id, "context": 4096}
    )
    assert response.status_code == 200
    job_id = response.json()["id"]
    assert client.get("/api/jobs").json()[0]["state"] == "queued"

    services.jobs.run_next()
    assert "GLM-4.5-Air-Q4_K_M.gguf" in str(services.engine.started[0])
    assert services.engine.started[1].context_size == 4096
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["state"] == "done"
    assert "Slow power is better than no power." in job["output"]
    assert "GLM-4.5-Air-Q4_K_M.gguf" in job["output"]


def test_api_job_without_a_loaded_model_fails_clearly(app):
    client, services = app
    job_id = client.post("/api/jobs", json={"prompt": "Hi"}).json()["id"]
    services.jobs.run_next()
    job = client.get(f"/api/jobs/{job_id}").json()
    assert job["state"] == "failed" and "No model is loaded" in job["error"]
    retried = client.post(f"/api/jobs/{job_id}/retry")
    assert retried.status_code == 200 and retried.json()["id"] != job_id


def test_api_job_errors(app):
    client, _ = app
    assert client.post("/api/jobs", json={"prompt": "Hi", "model_id": 999}).status_code == 404
    assert client.post("/api/jobs", json={"prompt": ""}).status_code == 422
    assert client.get("/api/jobs/42").status_code == 404
    assert client.post("/api/jobs/42/cancel").status_code == 404


def test_api_plan_includes_forecast(app):
    client, _ = app
    model_id = client.get("/api/library").json()[0]["id"]
    result = client.post(f"/api/library/{model_id}/plan", json={"extra_args": "-b 4096"}).json()
    assert result["forecast"]["speed"]["tokens_per_second"] > 0
    assert any(o["is_current"] for o in result["forecast"]["quant_options"])
    assert result["command"][-2:] == ["-b", "4096"]
    bad = client.post(f"/api/library/{model_id}/plan", json={"extra_args": '"unclosed'})
    assert bad.status_code == 422
