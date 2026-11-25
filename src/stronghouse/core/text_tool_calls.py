"""Tool calls written as plain text, for models whose chat template has no tool support.

Many local models never send OpenAI-style `tool_calls`: their template drops the tool list,
or llama.cpp does not recognise their call format, so the call arrives as ordinary text.
The terminal chat teaches them one simple text format and reads it back here.
See docs/adr/0016-text-tool-calls.md.
"""

from __future__ import annotations

import json
import re

from stronghouse.core.chat_web import WebCall, offer_tools
from stronghouse.core.skill_catalog import SKILLS

# The tag Qwen, Hermes and many fine-tunes already use, and Nvidia Nemotron's own.
_TAG = re.compile(r"<tool_call>|<TOOLCALL>", re.IGNORECASE)
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")
_REFUSAL = re.compile(
    r"(can ?not|can't|cannot|do not|don't|unable to|no) (have )?(directly )?"
    r"(access|browse|see|view|read|reach)\b.{0,60}"
    r"(file|folder|director|computer|system|internet|web|online)",
    re.IGNORECASE,
)

TEXT_CALL_GUIDANCE = (
    "You really do have these tools: the app runs them on the user's computer after the "
    "user approves. Never say you cannot access files or the internet; call a tool "
    "instead. If you cannot send a native tool call, write exactly one line and nothing "
    'else: <tool_call>{{"name": "<tool>", "arguments": {{...}}}}</tool_call>\n'
    "For example: {example}\n"
    "The tools and their arguments:\n{signatures}\n"
    "The result comes back in the next message; then answer the user."
)
# What the model reads back after a text call, sent as a user message so any template works.
TEXT_RESULT = "Result of {name} (from the app, not typed by the user):\n{result}"


def with_text_call_guidance(payload: dict, names: tuple[str, ...]) -> dict:
    """A copy of a chat request that also teaches the plain-text tool call format.

    The tools are listed with their arguments too, because a template without tool
    support drops the tool list the model would otherwise see.
    """
    if "list_folder" in names:
        example = example_call("list_folder", {"path": "~/Downloads"})
    else:
        example = example_call("web_search", {"query": "llama.cpp latest release"})
    guidance = TEXT_CALL_GUIDANCE.format(example=example, signatures=_signatures(names))
    return offer_tools(payload, (), guidance)


def _signatures(names: tuple[str, ...]) -> str:
    lines = []
    for name in names:
        spec = SKILLS[name]
        arguments = json.dumps(
            {
                key: value.get("description", "")
                for key, value in spec.parameters["properties"].items()
            }
        )
        lines.append(f"- {name} {arguments}: {spec.description}")
    return "\n".join(lines)


def example_call(name: str, arguments: dict) -> str:
    return f"<tool_call>{json.dumps({'name': name, 'arguments': arguments})}</tool_call>"


def starts_text_call(text: str) -> int:
    """Where a text tool call begins in `text`, or -1. Used to stop printing it."""
    found = _TAG.search(text)
    return found.start() if found else -1


def find_text_tool_calls(text: str, allowed: set[str]) -> list[WebCall]:
    """Tool calls written in `text`, keeping only tools in `allowed`.

    Accepts <tool_call>{...}</tool_call>, Nemotron's <TOOLCALL>[...]</TOOLCALL>, and a
    reply that is nothing but one JSON object {"name": ..., "arguments": {...}}.
    """
    found: list[dict] = []
    for match in _TAG.finditer(text):
        found += _json_calls(text[match.end() :])
    if not found:
        found = _json_calls(_FENCE.sub("", text.strip()), whole=True)
    calls = []
    for item in found:
        name, arguments = item.get("name"), item.get("arguments", item.get("parameters", {}))
        if name in allowed and isinstance(arguments, dict):
            calls.append(WebCall(f"text_{len(calls)}", name, json.dumps(arguments)))
    return calls


def looks_like_refusal(text: str) -> bool:
    """True when a reply says it cannot access things, the sign a model ignored its tools."""
    return bool(_REFUSAL.search(text))


def _json_calls(text: str, whole: bool = False) -> list[dict]:
    """The call objects at the start of `text`: one object or a list of them."""
    text = text.lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(text)
    except json.JSONDecodeError:
        return []
    if whole and text[end:].strip():
        return []
    items = value if isinstance(value, list) else [value]
    return [item for item in items if isinstance(item, dict) and "name" in item]
