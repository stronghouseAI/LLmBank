"""HTTP API for background jobs: queue prompts for a slow big model and collect answers."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Request
from pydantic import Field

from stronghouse.api.routes import PlanRequest
from stronghouse.core.jobs import Job
from stronghouse.services.job_runner import MAX_PROMPT_LENGTH, JobError, JobRunner

router = APIRouter(prefix="/api")

MAX_JOB_TOKENS = 131_072


class JobBody(PlanRequest):
    """A prompt plus how to run it. Plan fields are used only if model_id is given."""

    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_LENGTH)
    model_id: int | None = None
    think: bool = True
    max_tokens: int | None = Field(None, ge=1, le=MAX_JOB_TOKENS)


@router.post("/jobs")
def add_job(body: JobBody, request: Request) -> dict:
    runner = _runner(request)
    if body.model_id is not None and request.app.state.services.store.get(body.model_id) is None:
        raise HTTPException(404, f"No model with id {body.model_id} in the vault.")
    try:
        job = runner.submit(
            body.prompt, body.model_id, body.to_settings(), body.think, body.max_tokens
        )
    except JobError as error:
        raise HTTPException(422, str(error)) from error
    return _job_view(job)


@router.get("/jobs")
def list_jobs(request: Request) -> list[dict]:
    store = _runner(request).store
    return [_job_view(job) for job in store.list()]


@router.get("/jobs/{job_id}")
def get_job(job_id: int, request: Request) -> dict:
    runner = _runner(request)
    job = runner.store.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job with id {job_id}.")
    return {**_job_view(job), "output": runner.read_output(job)}


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: int, request: Request) -> dict:
    return _job_view(_act(request, lambda runner: runner.cancel(job_id)))


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: int, request: Request) -> dict:
    return _job_view(_act(request, lambda runner: runner.retry(job_id)))


def _act(request: Request, action) -> Job:
    try:
        return action(_runner(request))
    except JobError as error:
        status = 404 if str(error).startswith("No job") else 409
        raise HTTPException(status, str(error)) from error


def _runner(request: Request) -> JobRunner:
    runner = request.app.state.services.jobs
    if runner is None:
        raise HTTPException(503, "Background jobs are not available.")
    return runner


def _job_view(job: Job) -> dict:
    view = asdict(job)
    view["plan_settings"] = asdict(job.plan_settings)
    return view
