"""Background jobs: prompts that run on a big, slow model while you do something else.

A job is one prompt for one model. Jobs run one at a time in the order they were
queued; the answer is written to a Markdown file as it is generated, and a desktop
notification says when it is done. See docs/adr/0011-big-models-on-small-machines.md.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum

from stronghouse.core.models import PlanSettings

TITLE_LENGTH = 60
SLUG_LENGTH = 40


class JobState(StrEnum):
    QUEUED = "queued"
    LOADING = "loading"  # loading the job's model into the engine
    RUNNING = "running"  # generating
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def finished(self) -> bool:
        return self in (JobState.DONE, JobState.FAILED, JobState.CANCELLED)


@dataclass(frozen=True)
class Job:
    """One queued prompt. model_id None means "whatever model is loaded when it starts"."""

    prompt: str
    model_id: int | None
    plan_settings: PlanSettings
    think: bool
    max_tokens: int | None
    state: JobState
    created_at: str
    title: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    output_path: str | None = None
    generated_tokens: int = 0
    tokens_per_second: float | None = None
    error: str | None = None
    id: int | None = None


def job_title(prompt: str) -> str:
    """A short one-line title from the start of the prompt."""
    line = " ".join(prompt.split())
    return line if len(line) <= TITLE_LENGTH else line[: TITLE_LENGTH - 1].rstrip() + "…"


def output_filename(job_id: int, title: str) -> str:
    """e.g. "0007-summarise-the-attached-report.md"."""
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:SLUG_LENGTH].strip("-")
    return f"{job_id:04d}-{slug or 'job'}.md"


def output_header(job: Job, model_name: str) -> str:
    """The top of a job's answer file: what was asked and of which model."""
    quoted = "\n".join(f"> {line}" for line in job.prompt.splitlines() or [""])
    return f"# {job.title}\n\nModel: `{model_name}`\n\n## Prompt\n\n{quoted}\n\n## Answer\n\n"


def thinking_section(thinking: str) -> str:
    """The model's reasoning, folded away at the end of the answer file."""
    if not thinking.strip():
        return ""
    return f"\n\n<details><summary>Thinking</summary>\n\n{thinking.strip()}\n\n</details>\n"


def notification_text(job: Job) -> tuple[str, str]:
    """(summary, body) for the desktop notification when a job finishes."""
    if job.state is JobState.DONE:
        speed = f" at {job.tokens_per_second:g} tokens/s" if job.tokens_per_second else ""
        return "StronghouseAI job done", f"{job.title}\n{job.generated_tokens} tokens{speed}"
    if job.state is JobState.FAILED:
        return "StronghouseAI job failed", f"{job.title}\n{job.error or 'Unknown error'}"
    return "StronghouseAI job stopped", job.title


def split_sse(buffer: str) -> tuple[list[dict], str]:
    """Pull complete server-sent events out of a text buffer.

    Returns the decoded JSON of every complete "data: {...}" line, and the unfinished
    remainder to keep for the next chunk. "[DONE]" and non-data lines are skipped.
    """
    *lines, rest = buffer.split("\n")
    events = []
    for line in lines:
        line = line.strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            events.append(json.loads(line[len("data: ") :]))
        except json.JSONDecodeError:
            continue
    return events, rest
