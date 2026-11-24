"""Internet access in Chat: off by default, web tools only, each use asked for."""

from datetime import datetime

from stronghouse.core.chat_transcript import ChatTurn, transcript_text
from stronghouse.core.chat_web import WEB_TOOLS, is_web_tool, web_tool_definitions, with_web_tools
from stronghouse.core.skill_catalog import SKILLS


def test_chat_offers_exactly_the_agents_web_skills():
    assert set(WEB_TOOLS) == {"web_search", "fetch_web_page", "read_news_feed"}
    names = [tool["function"]["name"] for tool in web_tool_definitions()]
    assert names == list(WEB_TOOLS)
    for tool in web_tool_definitions():
        assert tool["function"]["parameters"] == SKILLS[tool["function"]["name"]].parameters


def test_files_and_commands_are_never_web_tools():
    for name in ("read_file", "write_file", "run_command", "open_with_app", "take_note"):
        assert not is_web_tool(name)


def test_web_tools_add_guidance_as_a_new_system_message():
    payload = {"messages": [{"role": "user", "content": "news?"}], "temperature": 0.3}
    result = with_web_tools(payload, "2026-11-24")
    assert result["temperature"] == 0.3
    assert result["messages"][0]["role"] == "system"
    assert "2026-11-24" in result["messages"][0]["content"]
    assert result["messages"][1] == {"role": "user", "content": "news?"}
    assert len(result["tools"]) == len(WEB_TOOLS)
    assert "tools" not in payload and len(payload["messages"]) == 1  # input left alone


def test_web_guidance_joins_an_existing_system_message():
    payload = {
        "messages": [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "x"}]
    }
    messages = with_web_tools(payload, "2026-11-24")["messages"]
    assert len(messages) == 2
    assert messages[0]["content"].startswith("Be brief.\n\n")
    assert payload["messages"][0]["content"] == "Be brief."


def test_transcript_names_internet_steps():
    turns = [ChatTurn("user", "q"), ChatTurn("tool", 'Search the web for "q"\n\nresults')]
    text = transcript_text(turns, None, datetime(2026, 11, 24, 12, 0))
    assert "Internet:\nSearch the web" in text
