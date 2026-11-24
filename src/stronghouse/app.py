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

from stronghouse import __version__
from stronghouse.api.agent_routes import router as agent_router
from stronghouse.api.errors import install_error_handlers
from stronghouse.api.job_routes import router as job_router
from stronghouse.api.log_routes import router as log_router
from stronghouse.api.routes import router
from stronghouse.config import Settings
from stronghouse.core.jobs import Job
from stronghouse.core.launch_args import ExtraArgsError, build_server_args
from stronghouse.core.models import HardwareProfile, LaunchPlan
from stronghouse.core.run_setup import plan_with_settings
from stronghouse.services.download_manager import DownloadManager
from stronghouse.services.engine import EngineController
from stronghouse.services.hardware_probe import probe_hardware
from stronghouse.services.huggingface import HuggingFaceClient
from stronghouse.services.job_runner import JobError, JobRunner
from stronghouse.services.library_indexer import LibraryIndexer
from stronghouse.services.notifier import DesktopNotifier
from stronghouse.services.quest_runner import QuestRunner
from stronghouse.services.skills import SkillExecutor
from stronghouse.storage.agent_store import AgentStore
from stronghouse.storage.job_store import JobStore
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
    # Runs skills for agents and the Chat page's internet access; each use is approved first.
    skills: SkillExecutor
    jobs: JobRunner | None = None

    def engine_command_preview(self, model_path: Path, plan: LaunchPlan) -> list[str]:
        """The exact llama-server command a plan would run, for display and copy-paste."""
        settings = self.settings
        return build_server_args(
            settings.llama_server_path, model_path, plan, settings.host, settings.engine_port
        )

    def load_model_for_job(self, job: Job) -> str:
        """Make sure the job's model is loaded or loading; return its file name.

        A job with no model uses whatever is loaded. A job for another model replaces
        the loaded one, planned with Auto-fit and the job's saved settings.

        Raises:
            JobError: if no model is loaded, or the job's model is gone or unplannable.
        """
        status = self.engine.status()
        if job.model_id is None:
            if status.model_path is None:
                raise JobError("No model is loaded. Load one on the Run page or pick one.")
            return Path(status.model_path).name
        entry = self.store.get(job.model_id)
        if entry is None:
            raise JobError(f"Model {job.model_id} is no longer in the vault.")
        if status.model_path == entry.path:
            return entry.filename
        try:
            profile = self.indexer.profile(Path(entry.path))
            self.engine.stop()  # free its GPU memory first, or the plan sees a full GPU
            plan = plan_with_settings(
                profile, self.probe_hardware(), job.plan_settings, self.settings.extra_args
            )
        except (ValueError, OSError, ExtraArgsError) as error:
            raise JobError(f"Could not plan {entry.filename}: {error}") from error
        self.engine.start(Path(entry.path), plan)
        return entry.filename


def build_services(settings: Settings) -> Services:
    store = LibraryStore(settings.database_path)
    indexer = LibraryIndexer(store, settings.archive_dir)
    hub = HuggingFaceClient(settings.huggingface_url, settings.huggingface_token)
    engine = EngineController(
        settings.llama_server_path,
        settings.host,
        settings.engine_port,
        log_path=settings.engine_log_path,
    )
    downloads = DownloadManager(hub, indexer, settings.archive_dir, engine.version)
    agents = AgentStore(settings.agents_database_path)
    skills = SkillExecutor(agents.add_note)
    quests = QuestRunner(agents, engine, skills)
    services = Services(
        store=store,
        indexer=indexer,
        hub=hub,
        downloads=downloads,
        engine=engine,
        probe_hardware=probe_hardware,
        settings=settings,
        agents=agents,
        quests=quests,
        skills=skills,
    )
    services.jobs = JobRunner(
        JobStore(settings.jobs_database_path),
        engine,
        load_model=services.load_model_for_job,
        notify=DesktopNotifier(settings.notify).notify,
        output_dir=settings.jobs_output_dir,
    )
    return services


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
        if services.jobs is not None:
            services.jobs.start()
        yield
        if services.jobs is not None:
            services.jobs.stop()
        # Never leave a llama-server holding GPU memory after the app quits.
        services.engine.stop()

    app = FastAPI(title="StronghouseAI", version=__version__, lifespan=lifespan)
    app.state.services = services
    install_error_handlers(app)
    app.include_router(router)
    app.include_router(agent_router)
    app.include_router(job_router)
    app.include_router(log_router)
    static_dir = resources.files("stronghouse.ui") / "static"
    app.mount("/", UiFiles(directory=str(static_dir), html=True), name="ui")
    return app
