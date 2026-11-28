"""Pure rules for the log viewer: which logs exist, hiding secrets, filtering, copy text.

No file access here: services/log_files.py reads the files and passes the lines in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from stronghouse.core.models import HardwareProfile


@dataclass(frozen=True)
class LogSource:
    """One log the user can open on the Logs page or with `shai logs`."""

    key: str
    title: str
    description: str
    # Which file it reads: "app", "engine" or "startup" (paths come from Settings).
    file: str
    # Only lines from these Python loggers (prefix match). Empty means every line.
    loggers: tuple[str, ...] = ()


LOG_SOURCES: tuple[LogSource, ...] = (
    LogSource("app", "App", "Everything the StronghouseAI server logged.", "app"),
    LogSource(
        "engine",
        "Engine",
        "llama-server output for the model loaded last (loading, memory, crashes).",
        "engine",
    ),
    LogSource(
        "jobs",
        "Jobs",
        "Background jobs: queued, model loads, finished and failed.",
        "app",
        (
            "stronghouse.services.job_runner",
            "stronghouse.api.job_routes",
            "stronghouse.services.notifier",
        ),
    ),
    LogSource(
        "agents",
        "Agents",
        "Agent tasks and the actions you allowed.",
        "app",
        (
            "stronghouse.services.quest_runner",
            "stronghouse.services.skills",
            "stronghouse.api.agent_routes",
        ),
    ),
    LogSource(
        "startup",
        "Startup",
        "Terminal output when `shai` started the app in the background.",
        "startup",
    ),
)

_SOURCES_BY_KEY = {source.key: source for source in LOG_SOURCES}

REDACTED = "[hidden]"

# (pattern, replacement). Order matters: specific token shapes first, then key=value.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bhf_[A-Za-z0-9]{16,}"), "hf_" + REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}"), "sk-" + REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"), "gh_" + REDACTED),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1" + REDACTED),
    (re.compile(r"(?i)(\b[a-z][a-z0-9+.-]*://)[^/\s:@]+:[^/\s@]+@"), r"\1" + REDACTED + "@"),
    (
        re.compile(
            r"(?i)\b((?:[a-z0-9]+[_-])?(?:token|api[_-]?key|apikey|password|passwd|secret))"
            r"(\"?'?\s*[:=]\s*\"?'?)([^\s\"'&,;}]+)"
        ),
        r"\1\2" + REDACTED,
    ),
)

# A line that reports a problem. Indented lines after it (a traceback) are kept with it.
_PROBLEM = re.compile(
    r"(?i)\b(warn(ing)?|error|errors|critical|fatal|fail(ed|ure)?|exception|traceback|abort(ed)?)\b"
    r"|out of memory"
)


def find_source(key: str) -> LogSource | None:
    return _SOURCES_BY_KEY.get(key)


def redact(line: str) -> str:
    """Hide tokens, passwords and keys so a copied log is safe to paste anywhere."""
    for pattern, replacement in _SECRET_PATTERNS:
        line = pattern.sub(replacement, line)
    return line


def shorten_home(line: str, home: str) -> str:
    """Show the home folder as ~ so a pasted log does not carry the user name."""
    home = home.rstrip("/")
    if not home or home == "/":
        return line
    return re.sub(re.escape(home) + r"(?=/|\b|$)", "~", line)


# app.log records start with a timestamp; anything else (a traceback) belongs to the one above.
_RECORD_START = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")


def _starts_record(line: str) -> bool:
    return bool(_RECORD_START.match(line))


def _is_indented(line: str) -> bool:
    """For logs without timestamps (llama-server): indented or blank lines continue a record."""
    return line[:1].isspace() or not line.strip()


def filter_by_logger(lines: list[str], loggers: tuple[str, ...]) -> list[str]:
    """Keep the records written by the given loggers (app.log format: date time LEVEL name: msg)."""
    if not loggers:
        return list(lines)
    kept: list[str] = []
    keeping = False
    for line in lines:
        if not _starts_record(line):
            if keeping:
                kept.append(line)
            continue
        parts = line.split(" ", 4)
        name = parts[3].rstrip(":") if len(parts) >= 4 else ""
        keeping = any(name == prefix or name.startswith(prefix + ".") for prefix in loggers)
        if keeping:
            kept.append(line)
    return kept


def only_problems(lines: list[str]) -> list[str]:
    """Keep warnings, errors and failures, plus the traceback lines that follow them."""
    timestamped = any(_starts_record(line) for line in lines)
    kept: list[str] = []
    keeping = False
    for line in lines:
        continues = not _starts_record(line) if timestamped else _is_indented(line)
        if continues and keeping:
            kept.append(line)
            continue
        if continues and timestamped:
            continue  # the tail of a record that was not kept
        keeping = bool(_PROBLEM.search(line))
        if keeping:
            kept.append(line)
    return kept


def prepare_lines(
    lines: list[str], source: LogSource, home: str, problems_only: bool = False
) -> list[str]:
    """Filter a raw log for one source and make it safe to show and copy."""
    selected = filter_by_logger(lines, source.loggers)
    if problems_only:
        selected = only_problems(selected)
    return [shorten_home(redact(line), home) for line in selected]


def copy_text(title: str, location: str, lines: list[str], note: str = "") -> str:
    """One log as a Markdown section, ready to paste into a chat with Claude."""
    header = f"### {title} log ({location})"
    if note:
        header += f"\n{note}"
    body = "\n".join(lines) if lines else "(empty)"
    # A ``` inside the log would end the code block early.
    body = body.replace("```", "'''")
    return f"{header}\n```\n{body}\n```"


def report_text(app_version: str, facts: list[str], sections: list[str]) -> str:
    """Several logs plus a few facts about the computer, for one paste."""
    lines = [f"## StronghouseAI {app_version} logs", *[f"- {fact}" for fact in facts]]
    return "\n".join(lines) + "\n\n" + "\n\n".join(sections) + "\n"


def _gb(size: int) -> str:
    return f"{size / 1_000_000_000:.1f} GB"


def system_facts(
    hardware: HardwareProfile | None,
    engine_state: str,
    model_file: str | None,
    engine_version: str | None,
) -> list[str]:
    """A few lines about this computer and the engine, so a pasted log has its context."""
    facts: list[str] = []
    if hardware is not None:
        gpu = "no GPU detected"
        if hardware.has_gpu_hardware:
            vram, free = _gb(hardware.gpu_total_bytes), _gb(hardware.gpu_free_bytes)
            backend = hardware.gpu_backend.value if hardware.has_gpu else "not used (CPU build)"
            gpu = f"{hardware.gpu_name}, {vram} VRAM ({free} free), backend {backend}"
        facts += [
            f"GPU: {gpu}",
            f"RAM: {_gb(hardware.ram_total_bytes)} ({_gb(hardware.ram_available_bytes)} free)",
            f"CPU threads: {hardware.cpu_threads}",
        ]
    facts.append(f"Engine: {engine_version or 'llama-server not found'}")
    facts.append(f"Engine state: {engine_state}" + (f", model {model_file}" if model_file else ""))
    return facts
