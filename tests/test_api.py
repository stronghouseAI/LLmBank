"""HTTP API wiring, with real storage and fake hardware, hub and engine."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from stronghouse.app import build_services, create_app
from stronghouse.config import load_settings
from stronghouse.core.models import HardwareProfile
from stronghouse.services.engine import EngineError
from tests.gguf_builder import moe_model_spec, write_gguf

GIB = 1024**3


class FakeEngine:
    def __init__(self):
        self.started = None

    def version(self):
        return "version: test"

    def start(self, model_path, plan):
        self.started = (model_path, plan)
        return self.status()

    def stop(self):
        self.started = None

    def status(self):
        from stronghouse.services.engine import EngineState, EngineStatus

        state = EngineState.LOADING if self.started else EngineState.STOPPED
        model_path, plan = self.started or (None, None)
        return EngineStatus(state, model_path and str(model_path), plan, [], [])

    def benchmark(self):
        raise EngineError("No model is loaded and ready yet.")


@pytest.fixture
def client(tmp_path: Path):
    settings = load_settings({"STRONGHOUSE_DATA_DIR": str(tmp_path)})
    services = build_services(settings)
    services.engine = FakeEngine()
    services.probe_hardware = lambda: HardwareProfile(
        "GPU", 8 * GIB, 8 * GIB, 32 * GIB, 32 * GIB, 8
    )
    folder = settings.archive_dir / "owner/repo"
    folder.mkdir(parents=True)
    metadata, tensors = moe_model_spec(blocks=4)
    write_gguf(folder / "m.gguf", metadata, tensors)
    services.indexer.rescan()
    with TestClient(create_app(services)) as test_client:
        yield test_client, services


def test_ui_is_served(client):
    test_client, _ = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert "StronghouseAI" in response.text
    # Always revalidated, so a git pull is picked up without a hard refresh.
    assert test_client.get("/app.js").headers["cache-control"] == "no-cache"


def test_hardware_and_library(client):
    test_client, _ = client
    assert test_client.get("/api/hardware").json()["gpu_name"] == "GPU"
    (entry,) = test_client.get("/api/library").json()
    assert entry["is_moe"] is True


def test_profile_and_plan_for_library_model(client):
    test_client, _ = client
    entry_id = test_client.get("/api/library").json()[0]["id"]

    profile = test_client.get(f"/api/library/{entry_id}/profile").json()
    assert profile["expert_count"] == 8

    result = test_client.post(f"/api/library/{entry_id}/plan", json={"context": 4096}).json()
    assert result["plan"]["strategy"] == "full_gpu"
    assert "--model" in result["command"]


def test_manual_override_is_applied(client):
    test_client, _ = client
    entry_id = test_client.get("/api/library").json()[0]["id"]
    result = test_client.post(
        f"/api/library/{entry_id}/plan", json={"context": 4096, "n_cpu_moe": 99}
    ).json()
    assert result["plan"]["n_cpu_moe"] == 4  # capped at the block count


def test_start_engine_uses_plan(client):
    test_client, services = client
    entry_id = test_client.get("/api/library").json()[0]["id"]
    response = test_client.post("/api/engine/start", json={"model_id": entry_id})
    assert response.status_code == 200
    model_path, plan = services.engine.started
    assert plan.context_size == 8192
    assert model_path.name == "m.gguf"


def test_loading_a_model_unloads_the_old_one_before_planning(client):
    """The loaded model's GPU memory must not count as taken when planning the next one."""
    test_client, services = client
    services.probe_hardware = lambda: HardwareProfile(
        "GPU", 8 * GIB, GIB // 2 if services.engine.started else 8 * GIB, 32 * GIB, 32 * GIB, 8
    )
    entry_id = test_client.get("/api/library").json()[0]["id"]
    first = test_client.post("/api/engine/start", json={"model_id": entry_id}).json()
    preview = test_client.post(f"/api/library/{entry_id}/plan", json={}).json()["plan"]
    second = test_client.post("/api/engine/start", json={"model_id": entry_id}).json()
    assert second["plan"] == first["plan"]
    assert any("A model is loaded" in note for note in preview["notes"])


def test_benchmark_without_model_is_a_clear_error(client):
    test_client, _ = client
    response = test_client.post("/api/engine/benchmark")
    assert response.status_code == 409
    assert "No model" in response.json()["detail"]


def test_unknown_model_is_404(client):
    test_client, _ = client
    assert test_client.get("/api/library/999/profile").status_code == 404


def test_unsafe_download_is_rejected(client):
    test_client, _ = client
    response = test_client.post("/api/downloads", json={"repo_id": "../x", "filename": "a.gguf"})
    assert response.status_code == 400


def test_agent_shop_endpoints(client):
    test_client, _ = client
    catalog = test_client.get("/api/agents/catalog").json()
    assert catalog["recommended_max_skills"] == 5
    assert catalog["temperature_by_creativity"]["3"] == 0.3
    assert catalog["max_steps_by_diligence"]["10"] == 22
    assert "starters" not in catalog and "classes" not in catalog
    every_skill = [skill["name"] for skill in catalog["skills"]]

    created = test_client.post(
        "/api/agents", json={"name": "Newsie", "personality": "Reads news", "skills": every_skill}
    ).json()
    assert created["skills"] == every_skill and created["level"] == 1
    assert created["tasks_done"] == 0

    card = test_client.get(f"/api/agents/{created['id']}/card").json()
    imported = test_client.post("/api/agents/import", json=card).json()
    assert imported["name"] == "Newsie" and imported["id"] != created["id"]

    bad = test_client.post("/api/agents", json={"name": "X", "skills": ["fly"]})
    assert bad.status_code == 400

    quest = test_client.post("/api/quests", json={"task": "hi", "agent_ids": [imported["id"]]})
    assert quest.status_code == 200
    assert test_client.post("/api/quests", json={"task": "hi", "agent_ids": []}).status_code == 400


class FakeHubForFiles:
    def list_gguf_files(self, repo_id):
        from stronghouse.services.huggingface import RemoteFile

        return [
            RemoteFile("q4.gguf", 5 * GIB, None),
            RemoteFile("big-00001-of-00002.gguf", 20 * GIB, None),
            RemoteFile("big-00002-of-00002.gguf", 20 * GIB, None),
        ]


def test_hub_files_are_marked_with_whether_they_fit(client):
    test_client, services = client
    services.hub = FakeHubForFiles()
    files = test_client.get("/api/hub/files", params={"repo_id": "owner/repo"}).json()
    assert files == [
        {"filename": "q4.gguf", "size_bytes": 5 * GIB, "fit": "gpu"},
        {"filename": "big-00001-of-00002.gguf", "size_bytes": 40 * GIB, "fit": "too_big"},
    ]
    bad = test_client.get("/api/hub/search", params={"size": "enormous"})
    assert bad.status_code == 422


def test_invalid_input_is_one_readable_sentence(client):
    test_client, _ = client
    entry_id = test_client.get("/api/library").json()[0]["id"]
    response = test_client.post(f"/api/library/{entry_id}/plan", json={"context": 0})
    assert response.status_code == 422
    assert response.json()["detail"] == "context: Input should be greater than 0"


def test_unexpected_crash_is_json_without_the_exception_text(tmp_path: Path):
    settings = load_settings({"STRONGHOUSE_DATA_DIR": str(tmp_path)})
    services = build_services(settings)

    def broken_probe():
        raise RuntimeError("secret /home/me/token path")

    services.probe_hardware = broken_probe
    with TestClient(create_app(services), raise_server_exceptions=False) as test_client:
        response = test_client.get("/api/hardware")
    assert response.status_code == 500
    detail = response.json()["detail"]
    assert "RuntimeError" in detail and "secret" not in detail


class StreamingEngine(FakeEngine):
    def __init__(self, chunks, fail_after=False):
        super().__init__()
        self.chunks, self.fail_after, self.payload = chunks, fail_after, None

    def stream_chat(self, payload):
        self.payload = payload
        yield from self.chunks
        if self.fail_after:
            raise httpx.ReadError("connection reset")


def test_chat_passes_extra_fields_through(client):
    test_client, services = client
    services.engine = StreamingEngine([b'data: {"choices":[]}\n\n'])
    body = {"messages": [{"role": "user", "content": "hi"}], "temperature": 0.2}
    response = test_client.post("/api/chat", json=body)
    assert response.status_code == 200
    assert services.engine.payload["temperature"] == 0.2


def test_chat_without_messages_is_rejected(client):
    test_client, _ = client
    assert test_client.post("/api/chat", json={"messages": []}).status_code == 422
    assert test_client.post("/api/chat", content=b"not json").status_code == 422


def test_chat_that_breaks_mid_reply_ends_with_an_error_event(client):
    test_client, services = client
    services.engine = StreamingEngine([b'data: {"choices":[]}\n\n'], fail_after=True)
    response = test_client.post("/api/chat", json={"messages": [{"role": "user", "content": "x"}]})
    last_event = response.text.strip().split("\n\n")[-1]
    assert json.loads(last_event.removeprefix("data: "))["error"]["message"].startswith(
        "The engine stopped mid-reply"
    )


def test_chat_is_offline_unless_internet_is_turned_on(client):
    test_client, services = client
    services.engine = StreamingEngine([b'data: {"choices":[]}\n\n'])
    body = {"messages": [{"role": "user", "content": "hi"}]}
    test_client.post("/api/chat", json=body)
    assert "tools" not in services.engine.payload
    assert "web" not in services.engine.payload

    test_client.post("/api/chat", json={**body, "web": True})
    names = {tool["function"]["name"] for tool in services.engine.payload["tools"]}
    assert names == {"web_search", "fetch_web_page", "read_news_feed"}
    assert "web" not in services.engine.payload
    assert services.engine.payload["messages"][0]["role"] == "system"


class FakeSkills:
    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def execute(self, agent_id, skill, arguments):
        self.calls.append((skill, arguments))
        if self.error:
            raise self.error
        return "1. Result - https://example.org"


def test_chat_web_action_runs_an_approved_search(client):
    test_client, services = client
    services.skills = FakeSkills()
    body = {"name": "web_search", "arguments": {"query": "llama.cpp release"}}
    response = test_client.post("/api/chat/web", json=body)
    assert response.json() == {"result": "1. Result - https://example.org", "ok": True}
    assert services.skills.calls == [("web_search", {"query": "llama.cpp release"})]


def test_chat_web_action_refuses_anything_but_web_tools(client):
    test_client, services = client
    services.skills = FakeSkills()
    for name in ("run_command", "write_file", "read_file"):
        response = test_client.post("/api/chat/web", json={"name": name, "arguments": {}})
        assert response.status_code == 400
    assert services.skills.calls == []


def test_chat_web_action_failure_goes_back_to_the_model_as_text(client):
    from stronghouse.services.skills import SkillError

    test_client, services = client
    services.skills = FakeSkills(SkillError("example.org refused the request (403)."))
    response = test_client.post(
        "/api/chat/web",
        json={"name": "fetch_web_page", "arguments": {"url": "https://example.org"}},
    )
    assert response.status_code == 200
    assert response.json() == {
        "result": "Error: example.org refused the request (403).",
        "ok": False,
    }


def test_chat_file_action_lists_an_approved_folder(client):
    test_client, services = client
    services.skills = FakeSkills()
    body = {"name": "list_folder", "arguments": {"path": "~/Downloads"}}
    response = test_client.post("/api/chat/files", json=body)
    assert response.json()["ok"] is True
    assert services.skills.calls == [("list_folder", {"path": "~/Downloads"})]


def test_chat_file_action_refuses_writing_commands_and_web(client):
    test_client, services = client
    services.skills = FakeSkills()
    for name in ("write_file", "move_file", "run_command", "web_search"):
        response = test_client.post("/api/chat/files", json={"name": name, "arguments": {}})
        assert response.status_code == 400
    assert services.skills.calls == []


def test_chat_offers_file_tools_only_when_asked(client):
    test_client, services = client
    seen = []
    services.engine.stream_chat = lambda payload: (
        seen.append(payload) or iter([b"data: [DONE]\n\n"])
    )
    messages = [{"role": "user", "content": "hi"}]
    test_client.post("/api/chat", json={"messages": messages})
    test_client.post("/api/chat", json={"messages": messages, "files": True})
    assert "tools" not in seen[0] and "files" not in seen[0]
    assert seen[1]["tools"][0]["function"]["name"] == "list_folder"
    assert "files" not in seen[1]


def test_text_call_guidance_is_added_only_with_tools(client):
    test_client, services = client
    seen = []
    services.engine.stream_chat = lambda payload: (
        seen.append(payload) or iter([b"data: [DONE]\n\n"])
    )
    messages = [{"role": "user", "content": "hi"}]
    test_client.post("/api/chat", json={"messages": messages, "text_tool_calls": True})
    test_client.post(
        "/api/chat", json={"messages": messages, "files": True, "text_tool_calls": True}
    )
    assert seen[0]["messages"] == messages and "text_tool_calls" not in seen[0]
    assert "<tool_call>" in seen[1]["messages"][0]["content"]
    assert "text_tool_calls" not in seen[1]
