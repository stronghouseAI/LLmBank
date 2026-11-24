"""HTTP API used by the web UI. A thin layer: validate input, call a service, return data."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from stronghouse.core.chat_transcript import ChatTurn, transcript_filename, transcript_text
from stronghouse.core.hub_filters import (
    ModelKind,
    SearchFilters,
    SizeClass,
    SortOrder,
    combine_split_parts,
    file_fit,
)
from stronghouse.core.launch_args import ExtraArgsError
from stronghouse.core.models import KvCacheType, LibraryEntry, ModelProfile, PlanSettings
from stronghouse.core.run_setup import forecast, plan_with_settings
from stronghouse.services.engine import EngineError, EngineState
from stronghouse.services.gguf_reader import GgufFormatError
from stronghouse.storage.archive import UnsafePathError

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

DEFAULT_CONTEXT = 8192
MAX_CONTEXT = 1_048_576
LOADED_MODEL_NOTE = (
    "A model is loaded and using GPU memory. Loading this one unloads it first,"
    " so the real plan may put more on the GPU."
)


class PlanRequest(BaseModel):
    context: int = Field(DEFAULT_CONTEXT, gt=0, le=MAX_CONTEXT)
    kv_cache_type: KvCacheType = KvCacheType.F16
    # Manual overrides for people who want to fine-tune after the auto plan.
    n_cpu_moe: int | None = Field(None, ge=0)
    n_gpu_layers: int | None = Field(None, ge=0)
    # Speed flags, as in a hand-written llama-server command. None = llama.cpp default.
    threads: int | None = Field(None, ge=1, le=1024)
    batch_size: int | None = Field(None, ge=1, le=65536)
    ubatch_size: int | None = Field(None, ge=1, le=65536)
    # Extra llama-server flags; None uses STRONGHOUSE_EXTRA_ARGS.
    extra_args: str | None = Field(None, max_length=2000)

    def to_settings(self) -> PlanSettings:
        return PlanSettings(
            context=self.context,
            kv_cache_type=self.kv_cache_type,
            n_cpu_moe=self.n_cpu_moe,
            n_gpu_layers=self.n_gpu_layers,
            threads=self.threads,
            batch_size=self.batch_size,
            ubatch_size=self.ubatch_size,
            extra_args=self.extra_args,
        )


class StartRequest(PlanRequest):
    model_id: int


class TagsRequest(BaseModel):
    tags: list[str]


class DownloadRequest(BaseModel):
    repo_id: str
    filename: str


class ChatRequest(BaseModel):
    """An OpenAI-style chat request. Extra fields (temperature, chat_template_kwargs...)
    are passed through to llama-server untouched."""

    model_config = ConfigDict(extra="allow")
    messages: list[dict] = Field(min_length=1)


class TranscriptTurn(BaseModel):
    role: str = Field(max_length=32)
    content: str
    thinking: str = ""


class TranscriptRequest(BaseModel):
    messages: list[TranscriptTurn] = Field(min_length=1)
    include_thinking: bool = True


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
    profile = _load_profile(request, entry_id)
    hardware = services.probe_hardware()
    plan = _plan_for(request, profile, body)
    if services.engine.status().state is not EngineState.STOPPED:
        plan = replace(plan, notes=[*plan.notes, LOADED_MODEL_NOTE])
    command = services.engine_command_preview(Path(entry.path), plan)
    outlook = forecast(profile, plan, hardware, entry.filename, services.settings.memory_speeds)
    return {"plan": asdict(plan), "command": command, "forecast": asdict(outlook)}


@router.post("/engine/start")
def start_engine(body: StartRequest, request: Request) -> dict:
    services = request.app.state.services
    entry = _entry(request, body.model_id)
    profile = _load_profile(request, body.model_id)
    services.engine.stop()  # free its GPU memory first, or the plan sees a full GPU
    plan = _plan_for(request, profile, body)
    outlook = forecast(
        profile, plan, services.probe_hardware(), entry.filename, services.settings.memory_speeds
    )
    try:
        started = services.engine.start(Path(entry.path), plan)
    except EngineError as error:
        raise HTTPException(503, str(error)) from error
    return {**asdict(started), "forecast": asdict(outlook)}


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
def chat(body: ChatRequest, request: Request) -> StreamingResponse:
    engine = request.app.state.services.engine
    try:
        stream = engine.stream_chat(body.model_dump())
        first = next(stream)  # surface "not ready" before the response starts
    except EngineError as error:
        raise HTTPException(409, str(error)) from error
    except httpx.HTTPError as error:
        raise HTTPException(502, f"Engine did not answer: {error}") from error
    except StopIteration:
        raise HTTPException(502, "Engine closed the reply before sending anything.") from None

    def events():
        yield first
        try:
            yield from stream
        except httpx.HTTPError as error:
            # Headers are already sent, so report the failure as a final stream event.
            log.warning("Chat stream broke off: %s", error)
            yield sse_error(f"The engine stopped mid-reply: {error}")

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/chat/transcript")
def chat_transcript(body: TranscriptRequest, request: Request) -> dict:
    """The conversation as plain text and a file name, for Copy chat and Export .txt.

    Built here rather than in the page so the format is tested; the browser saves the file.
    """
    status = request.app.state.services.engine.status()
    model = Path(status.model_path).name if status.model_path else None
    saved_at = datetime.now()
    turns = [ChatTurn(m.role, m.content, m.thinking) for m in body.messages]
    log.info("chat transcript built", extra={"messages": len(turns), "model": model})
    return {
        "text": transcript_text(turns, model, saved_at, body.include_thinking),
        "filename": transcript_filename(model, saved_at),
    }


def sse_error(message: str) -> bytes:
    """A server-sent event the chat UI shows as an error instead of a silent cut-off."""
    return f"data: {json.dumps({'error': {'message': message}})}\n\n".encode()


@router.get("/hub/search")
def hub_search(
    request: Request,
    q: str = "",
    size: SizeClass = SizeClass.ANY,
    kind: ModelKind = ModelKind.ANY,
    moe: bool = False,
    sort: SortOrder = SortOrder.DOWNLOADS,
) -> list:
    filters = SearchFilters(size=size, kind=kind, moe_only=moe, sort=sort)
    return _hub_call(lambda hub: hub.search_models(q, filters), request)


@router.get("/hub/files")
def hub_files(request: Request, repo_id: str) -> list:
    """GGUF files with split parts combined, each marked with whether it fits this machine."""
    files = _hub_call(lambda hub: hub.list_gguf_files(repo_id), request)
    hardware = request.app.state.services.probe_hardware()
    combined = combine_split_parts([(f["filename"], f["size_bytes"]) for f in files])
    return [
        {"filename": name, "size_bytes": size, "fit": file_fit(size, hardware).value}
        for name, size in combined
    ]


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


def _plan_for(request: Request, profile: ModelProfile, body: PlanRequest):
    services = request.app.state.services
    try:
        return plan_with_settings(
            profile, services.probe_hardware(), body.to_settings(), services.settings.extra_args
        )
    except ExtraArgsError as error:
        raise HTTPException(422, str(error)) from error


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
