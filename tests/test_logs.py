"""The Logs page and `shai logs`: reading, filtering, hiding secrets and copy text."""

import logging
from pathlib import Path

import pytest

from stronghouse.cli.commands import run
from stronghouse.config import load_settings
from stronghouse.core.log_view import (
    REDACTED,
    copy_text,
    filter_by_logger,
    find_source,
    only_problems,
    prepare_lines,
    redact,
    report_text,
    shorten_home,
    system_facts,
)
from stronghouse.core.models import HardwareProfile
from stronghouse.services.log_files import configure_logging, read_log, read_tail
from tests.test_api import client  # noqa: F401  (pytest fixture)

APP_LOG = [
    "2026-11-24 10:00:00,001 INFO stronghouse.services.engine: starting engine",
    "2026-11-24 10:00:01,002 INFO stronghouse.services.job_runner: job queued",
    "2026-11-24 10:00:02,003 ERROR stronghouse.services.quest_runner: quest crashed",
    "Traceback (most recent call last):",
    '  File "x.py", line 1, in <module>',
    "ValueError: boom",
    "2026-11-24 10:00:03,004 WARNING stronghouse.services.skills.files: refused path",
    "2026-11-24 10:00:04,005 INFO stronghouse.services.job_runner_extra: not a job line",
]


# ---------- hiding secrets ----------


@pytest.mark.parametrize(
    ("line", "secret"),
    [
        ("token hf_abcdefghijklmnopqrstuvwxyz123", "hf_abcdefghijklmnopqrstuvwxyz123"),
        ("Authorization: Bearer abc.def-123", "abc.def-123"),
        ("key sk-proj_ABCDEFGHIJKLMNOPQRSTUV", "sk-proj_ABCDEFGHIJKLMNOPQRSTUV"),
        ("push ghp_ABCDEFGHIJKLMNOPQRSTUVWX", "ghp_ABCDEFGHIJKLMNOPQRSTUVWX"),
        ("GET https://me:hunter2@example.com/x", "hunter2"),
        ("url ...?token=s3cr3t&x=1", "s3cr3t"),
        ('{"api_key": "abc123"}', "abc123"),
        ("HF_TOKEN=abc123", "abc123"),
        ("password: letmein", "letmein"),
    ],
)
def test_secrets_are_hidden(line, secret):
    hidden = redact(line)
    assert secret not in hidden
    assert REDACTED in hidden


def test_ordinary_log_lines_are_unchanged():
    line = "llama_model_load: loaded 48 layers, n_ctx = 8192, tokens per second 12.5"
    assert redact(line) == line


def test_home_folder_becomes_tilde():
    assert shorten_home("/home/phil/models/a.gguf", "/home/phil") == "~/models/a.gguf"
    assert shorten_home("/home/philip/x", "/home/phil") == "/home/philip/x"
    assert shorten_home("/x", "/") == "/x"


# ---------- filtering ----------


def test_filter_by_logger_keeps_tracebacks_with_their_record():
    agents = find_source("agents")
    assert agents is not None
    kept = filter_by_logger(APP_LOG, agents.loggers)
    # The whole traceback stays with its record; skills.files is a sub-logger of skills.
    assert kept == APP_LOG[2:7]


def test_jobs_filter_matches_whole_logger_names_only():
    jobs = find_source("jobs")
    assert jobs is not None
    kept = filter_by_logger(APP_LOG, jobs.loggers)
    assert kept == [APP_LOG[1]]


def test_only_problems_keeps_warnings_errors_and_their_tracebacks():
    assert only_problems(APP_LOG) == APP_LOG[2:7]


def test_engine_out_of_memory_counts_as_a_problem():
    lines = ["load_tensors: ok", "ggml_cuda: failed to allocate buffer", "cudaMalloc out of memory"]
    assert only_problems(lines) == lines[1:]


def test_prepare_lines_filters_then_hides_secrets():
    source = find_source("app")
    assert source is not None
    lines = prepare_lines(["/home/me/x token=abc"], source, home="/home/me")
    assert lines == [f"~/x token={REDACTED}"]


# ---------- copy text ----------


def test_copy_text_is_a_fenced_block_that_cannot_be_broken_out_of():
    text = copy_text("Engine", "~/.local/share/stronghouse/engine.log", ["a", "```", "b"])
    assert text.startswith("### Engine log (~/.local/share/stronghouse/engine.log)\n```\n")
    assert text.endswith("\n```")
    assert text.count("```") == 2


def test_copy_text_of_an_empty_log_says_so():
    assert "(empty)" in copy_text("App", "app.log", [])


def test_report_has_facts_and_every_section():
    hardware = HardwareProfile("RTX 4060 Laptop", 8 * 10**9, 7 * 10**9, 16 * 10**9, 9 * 10**9, 20)
    facts = system_facts(hardware, "ready", "glm.gguf", "version: 6000")
    text = report_text("0.1.0", facts, ["### A", "### B"])
    assert "## StronghouseAI 0.1.0 logs" in text
    assert "- GPU: RTX 4060 Laptop, 8.0 GB VRAM (7.0 GB free), backend cuda" in text
    assert "- Engine state: ready, model glm.gguf" in text
    assert "### A\n\n### B" in text


def test_facts_without_gpu_or_engine():
    facts = system_facts(HardwareProfile(None, 0, 0, 1, 1, 4), "stopped", None, None)
    assert "GPU: no GPU detected" in facts
    assert "Engine: llama-server not found" in facts


# ---------- reading files ----------


def test_read_tail_returns_the_last_lines(tmp_path: Path):
    path = tmp_path / "x.log"
    path.write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
    assert read_tail(path, 3) == ["line 97", "line 98", "line 99"]


def test_read_tail_drops_the_line_cut_by_the_byte_limit(tmp_path: Path):
    path = tmp_path / "x.log"
    path.write_text("first long line\nsecond\nthird\n")
    assert read_tail(path, 10, max_bytes=12) == ["third"]


def test_read_tail_of_a_missing_file_is_empty(tmp_path: Path):
    assert read_tail(tmp_path / "nope.log", 10) == []


def test_read_log_filters_and_reports_location(tmp_path: Path):
    settings = load_settings({"STRONGHOUSE_DATA_DIR": str(tmp_path)})
    settings.app_log_path.write_text("\n".join(APP_LOG) + "\n")
    jobs = find_source("jobs")
    assert jobs is not None
    view = read_log(settings, jobs, 50)
    assert view.exists and view.size_bytes > 0
    assert view.lines == [APP_LOG[1]]
    assert view.location.endswith("app.log")
    assert "job queued" in view.copy_text


def test_configure_logging_writes_app_log_without_http_noise(tmp_path: Path):
    settings = load_settings({"STRONGHOUSE_DATA_DIR": str(tmp_path)})
    root = logging.getLogger()
    saved = (root.handlers[:], root.level)
    root.handlers = []
    try:
        configure_logging(settings)
        logging.getLogger("stronghouse.test").warning("hello from the test", extra={"job_id": 7})
        logging.getLogger("httpx").info("HTTP Request: GET /health")
        for handler in root.handlers:
            handler.flush()
        text = settings.app_log_path.read_text()
        assert "WARNING stronghouse.test: hello from the test job_id=7" in text
        assert "/health" not in text
    finally:
        for handler in root.handlers:
            handler.close()
        root.handlers, level = saved
        root.setLevel(level)
        logging.getLogger("httpx").setLevel(logging.NOTSET)


# ---------- API ----------


def test_logs_api_lists_sources(client):  # noqa: F811
    test_client, _ = client
    keys = [s["key"] for s in test_client.get("/api/logs").json()]
    assert keys == ["app", "engine", "jobs", "agents", "startup"]


def test_logs_api_reads_a_log_with_secrets_hidden(client):  # noqa: F811
    test_client, services = client
    services.settings.engine_log_path.write_text("load ok\nHF_TOKEN=hf_abcdefghijklmnopqrstuv\n")
    body = test_client.get("/api/logs/engine?lines=10").json()
    assert body["exists"] is True
    assert body["lines"][0] == "load ok"
    assert "abcdefghijklmnop" not in body["copy_text"]


def test_logs_api_problems_filter(client):  # noqa: F811
    test_client, services = client
    services.settings.engine_log_path.write_text("all fine\nerror: bad\n")
    body = test_client.get("/api/logs/engine?problems=true").json()
    assert body["lines"] == ["error: bad"]
    assert "Only warnings and errors." in body["copy_text"]


def test_logs_api_unknown_log_and_bad_line_count(client):  # noqa: F811
    test_client, _ = client
    assert test_client.get("/api/logs/nope").status_code == 404
    assert test_client.get("/api/logs/app?lines=0").status_code == 422


def test_logs_report_has_every_file_log_and_hardware(client):  # noqa: F811
    test_client, _ = client
    text = test_client.get("/api/logs/report").json()["copy_text"]
    for title in ("App log", "Engine log", "Startup log"):
        assert f"### {title}" in text
    assert "- GPU: GPU" in text
    assert "Engine state: stopped" in text


# ---------- terminal ----------


def test_shai_logs_reads_files_without_the_app_running(tmp_path: Path, capsys):
    settings = load_settings({"STRONGHOUSE_DATA_DIR": str(tmp_path)})
    settings.engine_log_path.write_text("ok\nerror: password=hunter2\n")

    class NoServer:
        def __getattr__(self, name):
            raise AssertionError("shai logs must not need the app")

    assert run(["logs", "engine", "--problems"], settings, api=NoServer()) == 0
    out = capsys.readouterr().out
    assert "error: password=" in out and "hunter2" not in out
    assert "\nok\n" not in out

    assert run(["logs", "--lines", "5"], settings, api=NoServer()) == 0
    assert "### Startup log" in capsys.readouterr().out
