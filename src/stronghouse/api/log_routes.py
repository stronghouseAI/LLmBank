"""HTTP API for the Logs page: read the app's logs, with secrets hidden, ready to copy."""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

from stronghouse import __version__
from stronghouse.core.log_view import LOG_SOURCES, find_source, report_text, system_facts
from stronghouse.services.log_files import read_all_logs, read_log

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

DEFAULT_LINES = 500
MAX_LINES = 5000
REPORT_LINES = 200


@router.get("/logs")
def list_logs() -> list[dict]:
    return [{"key": s.key, "title": s.title, "description": s.description} for s in LOG_SOURCES]


# Declared before /logs/{key} so "report" is not taken for a log name.
@router.get("/logs/report")
def logs_report(
    request: Request,
    lines: int = Query(REPORT_LINES, ge=1, le=MAX_LINES),
    problems: bool = False,
) -> dict:
    """Every log plus a few facts about this computer, as one text to paste."""
    services = request.app.state.services
    try:
        hardware = services.probe_hardware()
    except Exception:  # a broken probe must not stop the report: it is a debugging aid
        log.warning("hardware probe failed while building the log report")
        hardware = None
    status = services.engine.status()
    model = Path(status.model_path).name if status.model_path else None
    facts = system_facts(hardware, status.state.value, model, services.engine.version())
    views = read_all_logs(services.settings, lines, problems)
    return {"copy_text": report_text(__version__, facts, [v.copy_text for v in views])}


@router.get("/logs/{key}")
def get_log(
    key: str,
    request: Request,
    lines: int = Query(DEFAULT_LINES, ge=1, le=MAX_LINES),
    problems: bool = False,
) -> dict:
    source = find_source(key)
    if source is None:
        raise HTTPException(404, f"No log called {key!r}.")
    return asdict(read_log(request.app.state.services.settings, source, lines, problems))
