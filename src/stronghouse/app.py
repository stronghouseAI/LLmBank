"""Wire settings, storage and services together into the FastAPI app.

This is the only place that constructs concrete services, so tests can build an
app with fakes by passing their own Services object.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from stronghouse.api.agent_routes import router as agent_router
from stronghouse.api.routes import router
from stronghouse.config import Settings
from stronghouse.core.launch_args import build_server_args
from stronghouse.core.models import HardwareProfile, LaunchPlan
from stronghouse.services.download_manager import DownloadManager
from stronghouse.services.engine import EngineController
from stronghouse.services.hardware_probe import probe_hardware
from stronghouse.services.huggingface import HuggingFaceClient
from stronghouse.services.library_indexer import LibraryIndexer
from stronghouse.services.quest_runner import QuestRunner
from stronghouse.services.skills import SkillExecutor
from stronghouse.storage.agent_store import AgentStore
from stronghouse.storage.library_db import LibraryStore


@dataclass
class Services:
    store: LibraryStore
    indexer: LibraryIndexer
    hub: HuggingFaceClient
    downloads: DownloadManager
    engine: EngineController
    probe_hardware: Callable[[], HardwareProfile]
    settings: Settings
    agents: AgentStore
    quests: QuestRunner

    def engine_command_preview(self, model_path: Path, plan: LaunchPlan) -> list[str]:
        """The exact llama-server command a plan would run, for display and copy-paste."""
        settings = self.settings
        return build_server_args(
            settings.llama_server_path, model_path, plan, settings.host, settings.engine_port
        )


def build_services(settings: Settings) -> Services:
    store = LibraryStore(settings.database_path)
    indexer = LibraryIndexer(store, settings.archive_dir)
    hub = HuggingFaceClient(settings.huggingface_url, settings.huggingface_token)
    engine = EngineController(
        settings.llama_server_path,
        settings.host,
        settings.engine_port,
        log_path=settings.data_dir / "engine.log",
    )
    downloads = DownloadManager(hub, indexer, settings.archive_dir, engine.version)
    agents = AgentStore(settings.agents_database_path)
    quests = QuestRunner(agents, engine, SkillExecutor(agents.add_note))
    return Services(
        store=store,
        indexer=indexer,
        hub=hub,
        downloads=downloads,
        engine=engine,
        probe_hardware=probe_hardware,
        settings=settings,
        agents=agents,
        quests=quests,
    )


class UiFiles(StaticFiles):
    """The web UI, always revalidated so a `git pull` shows up on the next page load.

    Without this, browsers may keep using an old app.js for a while after an update.
    """

    async def get_response(self, path: str, scope):  # type: ignore[no-untyped-def]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


def create_app(services: Services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        # Never leave a llama-server holding GPU memory after the app quits.
        services.engine.stop()

    app = FastAPI(title="StronghouseAI", version="0.1.0", lifespan=lifespan)
    app.state.services = services
    app.include_router(router)
    app.include_router(agent_router)
    static_dir = resources.files("stronghouse.ui") / "static"
    app.mount("/", UiFiles(directory=str(static_dir), html=True), name="ui")
    return app
