"""Pure rules for saving a chat: the plain-text transcript and its file name.

No file access here: the browser saves the text the API builds from these functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

# "tool" turns are the Chat page's internet steps: what was asked and what came back.
ROLE_NAMES = {"user": "You", "assistant": "Model", "system": "System", "tool": "Internet"}
RULE = "-" * 60


@dataclass(frozen=True)
class ChatTurn:
    """One message in the conversation as the Chat page shows it."""

    role: str
    content: str
    # The model's reasoning before its answer, for thinking models. Empty when there was none.
    thinking: str = ""


def transcript_text(
    turns: list[ChatTurn],
    model: str | None,
    saved_at: datetime,
    include_thinking: bool = True,
) -> str:
    """The whole conversation as plain text, oldest message first, ending in a newline."""
    header = [
        "StronghouseAI chat",
        f"Model: {model or 'unknown'}",
        f"Saved: {saved_at.strftime('%Y-%m-%d %H:%M')}",
        f"Messages: {len(turns)}",
    ]
    blocks = ["\n".join(header)]
    for turn in turns:
        blocks.append(_turn_text(turn, include_thinking))
    return f"\n\n{RULE}\n\n".join(blocks) + "\n"


def _turn_text(turn: ChatTurn, include_thinking: bool) -> str:
    name = ROLE_NAMES.get(turn.role, turn.role.capitalize() or "Unknown")
    parts = [f"{name}:"]
    if include_thinking and turn.thinking.strip():
        parts.append("[Thinking]\n" + turn.thinking.strip() + "\n[End of thinking]")
    parts.append(turn.content.strip() or "(empty)")
    return "\n".join(parts)


def transcript_filename(model: str | None, saved_at: datetime) -> str:
    """A safe file name such as chat-Qwen3-30B-A3B-Q4_K_M-2026-11-24-1350.txt."""
    stem = re.sub(r"\.gguf$", "", model or "", flags=re.IGNORECASE)
    stem = re.sub(r"-0+1-of-\d+$", "", stem)  # first part of a split model
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.")[:60]
    stamp = saved_at.strftime("%Y-%m-%d-%H%M")
    return f"chat-{stem}-{stamp}.txt" if stem else f"chat-{stamp}.txt"
