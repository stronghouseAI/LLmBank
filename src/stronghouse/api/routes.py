"""HTTP API used by the web UI. A thin layer: validate input, call a service, return data."""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from stronghouse.core.models import KvCacheType, LibraryEntry, ModelProfile
from stronghouse.core.offload_planner import plan_offload
from stronghouse.services.engine import EngineError
from stronghouse.services.gguf_reader import GgufFormatError
from stronghouse.storage.archive import UnsafePathError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

DEFAULT_CONTEXT = 8192
MAX_CONTEXT = 1_048_576


class PlanRequest(BaseModel):
    context: int = Field(DEFAULT_CONTEXT, gt=0, le=MAX_CONTEXT)
    kv_cache_type: KvCacheType = KvCacheType.F16
    # Manual overrides for people who want to fine-tune after the auto plan.
    n_cpu_moe: int | None = Field(None, ge=0)
    n_gpu_layers: int | None = Field(None, ge=0)


class StartRequest(PlanRequest):
    model_id: int


class TagsRequest(BaseModel):
    tags: list[str]


class DownloadRequest(BaseModel):
    repo_id: str
    filename: str


@router.get("/hardware")
def get_hardware(request: Request) -> dict:
    return asdict(request.app.state.services.probe_hardware())


@router.get("/library")
def list_library(request: Request, search: str = "") -> list[LibraryEntry]:
    return request.app.state.services.store.list_entries(search)


@router.post("/library/rescan")
def rescan_library(request: Request) -> dict:
    return {"indexed": request.app.state.services.indexer.rescan()}


@router.put("/library/{entry_id}/tags")
def set_tags(entry_id: int, body: TagsRequest, request: Request) -> LibraryEntry:
    entry = request.app.state.services.store.set_tags(entry_id, body.tags)
    if entry is None:
        raise HTTPException(404, "Model not found")
    return entry


@router.delete("/library/{entry_id}")
def forget_model(entry_id: int, request: Request) -> dict:
    """Remove from the library list only. Files on disk are never deleted by the app."""
    request.app.state.services.store.remove(entry_id)
    return {"removed": entry_id}


@router.get("/library/{entry_id}/profile")
def get_profile(entry_id: int, request: Request) -> dict:
    profile = _load_profile(request, entry_id)
    return _profile_summary(profile)


@router.post("/library/{entry_id}/plan")
def make_plan(entry_id: int, body: PlanRequest, request: Request) -> dict:
    services = request.app.state.services
    entry = _entry(request, entry_id)
    plan = _plan_for(request, entry_id, body)
    command = services.engine_command_preview(Path(entry.path), plan)
    return {"plan": asdict(plan), "command": command}


@router.post("/engine/start")
def start_engine(body: StartRequest, request: Request) -> dict:
    services = request.app.state.services
    entry = _entry(request, body.model_id)
    plan = _plan_for(request, body.model_id, body)
    try:
        return asdict(services.engine.start(Path(entry.path), plan))
    except EngineError as error:
        raise HTTPException(503, str(error)) from error


@router.post("/engine/stop")
def stop_engine(request: Request) -> dict:
    request.app.state.services.engine.stop()
    return asdict(request.app.state.services.engine.status())


@router.get("/engine")
def engine_status(request: Request) -> dict:
    services = request.app.state.services
    return {**asdict(services.engine.status()), "version": services.engine.version()}


@router.post("/engine/benchmark")
def benchmark(request: Request) -> dict:
    try:
        return asdict(request.app.state.services.engine.benchmark())
    except EngineError as error:
        raise HTTPException(409, str(error)) from error
    except httpx.HTTPError as error:
        raise HTTPException(502, f"Engine did not answer: {error}") from error


@router.post("/chat")
async def chat(request: Request) -> StreamingResponse:
    engine = request.app.state.services.engine
    payload = await request.json()
    try:
        stream = engine.stream_chat(payload)
        first = next(stream)  # surface "not ready" before the response starts
    except EngineError as error:
        raise HTTPException(409, str(error)) from error
    except httpx.HTTPError as error:
        raise HTTPException(502, f"Engine did not answer: {error}") from error

    def body():
        yield first
        yield from stream

    return StreamingResponse(body(), media_type="text/event-stream")


@router.get("/hub/search")
def hub_search(request: Request, q: str = "") -> list:
    return _hub_call(lambda hub: hub.search_models(q), request)


@router.get("/hub/files")
def hub_files(request: Request, repo_id: str) -> list:
    return _hub_call(lambda hub: hub.list_gguf_files(repo_id), request)


@router.post("/downloads")
def start_download(body: DownloadRequest, request: Request) -> dict:
    try:
        return asdict(request.app.state.services.downloads.start(body.repo_id, body.filename))
    except UnsafePathError as error:
        raise HTTPException(400, str(error)) from error


@router.get("/downloads")
def list_downloads(request: Request) -> list:
    return [asdict(job) for job in request.app.state.services.downloads.jobs()]


def _hub_call(call, request: Request) -> list:
    try:
        return [asdict(item) for item in call(request.app.state.services.hub)]
    except httpx.HTTPError as error:
        raise HTTPException(502, f"Hugging Face is unreachable: {error}") from error


def _entry(request: Request, entry_id: int) -> LibraryEntry:
    entry = request.app.state.services.store.get(entry_id)
    if entry is None:
        raise HTTPException(404, "Model not found")
    return entry


def _load_profile(request: Request, entry_id: int) -> ModelProfile:
    entry = _entry(request, entry_id)
    try:
        return request.app.state.services.indexer.profile(Path(entry.path))
    except (GgufFormatError, ValueError, OSError) as error:
        raise HTTPException(422, f"Could not read model file: {error}") from error


def _plan_for(request: Request, entry_id: int, body: PlanRequest):
    profile = _load_profile(request, entry_id)
    hardware = request.app.state.services.probe_hardware()
    plan = plan_offload(profile, hardware, body.context, body.kv_cache_type)
    overrides = {}
    if body.n_cpu_moe is not None:
        overrides["n_cpu_moe"] = min(body.n_cpu_moe, profile.block_count)
    if body.n_gpu_layers is not None:
        overrides["n_gpu_layers"] = body.n_gpu_layers
    if overrides:
        plan = replace(plan, notes=[*plan.notes, "Manual override applied."], **overrides)
    return plan


def _profile_summary(profile: ModelProfile) -> dict:
    return {
        "architecture": profile.architecture,
        "block_count": profile.block_count,
        "context_length": profile.context_length,
        "is_moe": profile.is_moe,
        "expert_count": profile.expert_count,
        "experts_used": profile.experts_used,
        "expert_bytes": profile.total_expert_bytes,
        "dense_bytes": profile.total_dense_bytes + profile.output_bytes,
        "embedding_bytes": profile.token_embedding_bytes,
        "kv_bytes_per_token_f16": profile.kv_bytes_per_token_f16,
    }
