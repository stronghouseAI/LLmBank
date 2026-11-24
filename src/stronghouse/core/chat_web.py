"""Pure rules for letting the Chat page's model use the internet.

The Chat page is offline unless the user turns on "Use internet". Then the model
gets the agents' three web skills as tools, and the page asks the user before
each one runs. See docs/adr/0014-chat-internet-access.md.
"""

from __future__ import annotations

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
