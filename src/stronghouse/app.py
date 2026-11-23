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

from stronghouse.api.routes import router
from stronghouse.config import Settings
from stronghouse.core.launch_args import build_server_args
from stronghouse.core.models import HardwareProfile, LaunchPlan
from stronghouse.services.download_manager import DownloadManager
from stronghouse.services.engine import EngineController
from stronghouse.services.hardware_probe import probe_hardware
from stronghouse.services.huggingface import HuggingFaceClient
from stronghouse.services.library_indexer import LibraryIndexer
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
    return Services(
        store=store,
        indexer=indexer,
        hub=hub,
        downloads=downloads,
        engine=engine,
        probe_hardware=probe_hardware,
        settings=settings,
    )


def create_app(services: Services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        # Never leave a llama-server holding GPU memory after the app quits.
        services.engine.stop()

    app = FastAPI(title="StronghouseAI", version="0.1.0", lifespan=lifespan)
    app.state.services = services
    app.include_router(router)
    static_dir = resources.files("stronghouse.ui") / "static"
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")
    return app
