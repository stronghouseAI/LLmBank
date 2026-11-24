"""Pure rules for letting the Chat page's model use the internet.

The Chat page is offline unless the user turns on "Use internet". Then the model
gets the agents' three web skills as tools, and the page asks the user before
each one runs. See docs/adr/0014-chat-internet-access.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from stronghouse.core.skill_catalog import SKILLS, SkillCategory

WEB_TOOLS: tuple[str, ...] = tuple(
    name for name, spec in SKILLS.items() if spec.category is SkillCategory.WEB
)
# Searches and page reads a single reply may make before the model must answer.
MAX_WEB_STEPS = 6

WEB_GUIDANCE = (
    "You can use the internet with the tools web_search, fetch_web_page and "
    "read_news_feed. Use them only when the answer needs current or outside "
    "information you do not know; otherwise answer directly. The user approves "
    "every use, so make each one count. Today's date is {today}. When you use what "
    "you found, say which site it came from."
)
# What the model hears when the user says no to a web call.
WEB_DENIED = (
    "The user did not allow this. Do not ask again; answer from what you know "
    "and say you could not check online."
)
# How a web call is put to the user: (sentence, the argument it shows).
WEB_ACTIONS: dict[str, tuple[str, str]] = {
    "web_search": ("Search the web for \u201c{}\u201d", "query"),
    "fetch_web_page": ("Open the page {}", "url"),
    "read_news_feed": ("Read the news feed {}", "url"),
}


@dataclass
class WebCall:
    """One tool call the model asked for, put together from its streamed pieces."""

    id: str = ""
    name: str = ""
    arguments: str = ""  # JSON text, as the model wrote it

    def parsed_arguments(self) -> dict:
        try:
            value = json.loads(self.arguments or "{}")
        except json.JSONDecodeError:
            return {}
        return value if isinstance(value, dict) else {}

    def as_message_part(self) -> dict:
        """The OpenAI-style entry for the assistant message's tool_calls."""
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.name, "arguments": self.arguments},
        }


@dataclass
class ToolCallCollector:
    """Joins streamed tool-call deltas, which arrive in pieces keyed by their index."""

    _calls: dict[int, WebCall] = field(default_factory=dict)

    def add(self, parts: list[dict]) -> None:
        for part in parts:
            call = self._calls.setdefault(part.get("index") or 0, WebCall())
            function = part.get("function") or {}
            call.id = part.get("id") or call.id
            call.name += function.get("name") or ""
            call.arguments += function.get("arguments") or ""

    def calls(self) -> list[WebCall]:
        """The complete calls in order, each with an id (made up if the model gave none)."""
        found = [self._calls[index] for index in sorted(self._calls) if self._calls[index].name]
        return [
            WebCall(call.id or f"call_{number}", call.name, call.arguments)
            for number, call in enumerate(found)
        ]


def describe_web_call(name: str, arguments: dict) -> str:
    """The plain sentence the user is asked to allow, e.g. 'Search the web for "x"'."""
    if name not in WEB_ACTIONS:
        return f"Use {name}"
    sentence, key = WEB_ACTIONS[name]
    return sentence.format(arguments.get(key, ""))


def is_web_tool(name: str) -> bool:
    return name in WEB_TOOLS


def web_tool_definitions() -> list[dict]:
    """OpenAI-style tool list for the web skills, the same ones agents equip."""
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": SKILLS[name].description,
                "parameters": SKILLS[name].parameters,
            },
        }
        for name in WEB_TOOLS
    ]


def with_web_tools(payload: dict, today: str) -> dict:
    """A copy of a chat request that offers the web tools and says when to use them.

    The guidance goes into the conversation's system message, or a new one at the top,
    so templates that allow only one system message still work.
    """
    guidance = WEB_GUIDANCE.format(today=today)
    messages = [dict(message) for message in payload["messages"]]
    if messages and messages[0].get("role") == "system":
        messages[0]["content"] = f"{messages[0].get('content') or ''}\n\n{guidance}".strip()
    else:
        messages.insert(0, {"role": "system", "content": guidance})
    return {**payload, "messages": messages, "tools": web_tool_definitions()}
