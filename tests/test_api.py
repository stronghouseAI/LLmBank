"""HTTP API wiring, with real storage and fake hardware, hub and engine."""

from pathlib import Path

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
        return EngineStatus(state, None, None, [], [])

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
    assert "starters" not in catalog and "classes" not in catalog
    every_skill = [skill["name"] for skill in catalog["skills"]]

    created = test_client.post(
        "/api/agents", json={"name": "Newsie", "personality": "Reads news", "skills": every_skill}
    ).json()
    assert created["skills"] == every_skill and created["level"] == 1

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
