"""Read the app's log files for the Logs page and `shai logs`.

Reading only: the app never deletes or edits a log from here. Secrets are hidden by
core/log_view.py before any line leaves this module.
"""

from __future__ import annotations

import logging
import logging.handlers
from dataclasses import dataclass
from pathlib import Path

from stronghouse.config import (
    LOG_FILE_BACKUPS,
    LOG_FILE_MAX_BYTES,
    LOG_READ_MAX_BYTES,
    Settings,
)
from stronghouse.core.log_view import (
    LOG_SOURCES,
    LogSource,
    copy_text,
    prepare_lines,
    shorten_home,
)

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# Libraries that log every HTTP request; the engine health check alone would flood app.log.
_CHATTY_LOGGERS = ("httpx", "httpcore")


# Attributes every LogRecord has; anything else came from extra={...} and is shown as key=value.
_STANDARD_RECORD_FIELDS = frozenset(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class KeyValueFormatter(logging.Formatter):
    """The usual line, plus the extra={...} fields, so structured details reach the log file."""

    def format(self, record: logging.LogRecord) -> str:
        line = super().format(record)
        extras = {k: v for k, v in vars(record).items() if k not in _STANDARD_RECORD_FIELDS}
        if not extras:
            return line
        details = " ".join(f"{key}={value!r}" for key, value in extras.items())
        first, newline, rest = line.partition("\n")  # keep a traceback below the details
        return f"{first} {details}{newline}{rest}"


@dataclass(frozen=True)
class LogView:
    """One log, filtered and with secrets hidden, plus its copy-ready text."""

    key: str
    title: str
    description: str
    location: str
    exists: bool
    size_bytes: int
    lines: list[str]
    copy_text: str


def configure_logging(settings: Settings) -> None:
    """Log to the terminal and to app.log (rotated), so the Logs page has something to show."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        settings.app_log_path,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUPS,
        encoding="utf-8",
    )
    handlers: list[logging.Handler] = [logging.StreamHandler(), file_handler]
    for handler in handlers:
        handler.setFormatter(KeyValueFormatter(LOG_FORMAT))
    logging.basicConfig(level=logging.INFO, handlers=handlers)
    for name in _CHATTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


def read_tail(path: Path, max_lines: int, max_bytes: int = LOG_READ_MAX_BYTES) -> list[str]:
    """The last max_lines lines of a text file, reading no more than max_bytes from its end."""
    if max_lines <= 0 or not path.is_file():
        return []
    with path.open("rb") as handle:
        size = handle.seek(0, 2)
        start = max(0, size - max_bytes)
        handle.seek(start)
        data = handle.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    if start > 0 and lines:
        lines = lines[1:]  # the first line was cut in half by the byte limit
    return lines[-max_lines:]


def read_log(
    settings: Settings,
    source: LogSource,
    max_lines: int,
    problems_only: bool = False,
) -> LogView:
    """Read one log source, newest lines last."""
    path = settings.log_path(source.file)
    home = str(Path.home())
    # Filtered sources share app.log with everything else, so read more and filter down.
    raw_lines = max_lines * 10 if source.loggers or problems_only else max_lines
    lines = prepare_lines(read_tail(path, raw_lines), source, home, problems_only)[-max_lines:]
    location = shorten_home(str(path), home)
    note = "Only warnings and errors." if problems_only else ""
    exists = path.is_file()
    return LogView(
        key=source.key,
        title=source.title,
        description=source.description,
        location=location,
        exists=exists,
        size_bytes=path.stat().st_size if exists else 0,
        lines=lines,
        copy_text=copy_text(source.title, location, lines, note),
    )


def read_all_logs(settings: Settings, max_lines: int, problems_only: bool = False) -> list[LogView]:
    """Every log source that reads its own file (Jobs and Agents are already inside App)."""
    return [
        read_log(settings, source, max_lines, problems_only)
        for source in LOG_SOURCES
        if not source.loggers
    ]
