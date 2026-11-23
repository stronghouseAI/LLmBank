"""Plain-text output for the terminal: colours, tables, sizes and progress bars.

Colours are switched off when output is not a terminal or NO_COLOR is set.
"""

from __future__ import annotations

import os
import sys

_STYLES = {
    "bold": "1",
    "dim": "2",
    "green": "32",
    "yellow": "33",
    "red": "31",
    "cyan": "36",
    "brass": "38;5;178",
}


def use_colour(stream=None) -> bool:
    stream = stream or sys.stdout
    return "NO_COLOR" not in os.environ and hasattr(stream, "isatty") and stream.isatty()


def paint(text: str, *styles: str) -> str:
    if not use_colour() or not styles:
        return text
    codes = ";".join(_STYLES[style] for style in styles)
    return f"\033[{codes}m{text}\033[0m"


def format_bytes(size: int | float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit in ("GB", "TB") else f"{value:.0f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Left-aligned columns, sized to the widest cell (ignoring colour codes)."""
    if not rows:
        return ""
    widths = [len(h) for h in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], _visible_len(cell))
    lines = [
        "  ".join(_cell(paint(h.upper(), "dim"), w) for h, w in zip(headers, widths, strict=True))
    ]
    for row in rows:
        lines.append(
            "  ".join(_cell(cell, w) for cell, w in zip(row, widths, strict=True)).rstrip()
        )
    return "\n".join(lines)


def progress_bar(done: int, total: int, width: int = 30) -> str:
    fraction = done / total if total else 0.0
    filled = int(fraction * width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {fraction * 100:5.1f}%"


def _visible_len(text: str) -> int:
    length, in_escape = 0, False
    for char in text:
        if char == "\033":
            in_escape = True
        elif in_escape and char == "m":
            in_escape = False
        elif not in_escape:
            length += 1
    return length


def _cell(text: str, width: int) -> str:
    """Pad text to width visible characters; colour codes take no room on screen."""
    return text + " " * max(0, width - _visible_len(text))
