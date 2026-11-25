"""File access in chat: off by default, read-only file tools only, each use asked for."""

from stronghouse.core.chat_files import (
    FILE_TOOLS,
    describe_file_call,
    is_file_tool,
    with_file_tools,
)
from stronghouse.core.chat_web import WEB_TOOLS, with_web_tools
from stronghouse.core.skill_catalog import SKILLS, SkillCategory


def test_chat_offers_only_reading_skills():
    assert set(FILE_TOOLS) == {"list_folder", "read_file", "search_files", "read_pdf"}
    for name in FILE_TOOLS:
        assert SKILLS[name].needs_approval
        assert SKILLS[name].category in (SkillCategory.FILES, SkillCategory.DOCUMENTS)
    for name in ("write_file", "move_file", "run_command", "open_with_app", "web_search"):
        assert not is_file_tool(name)


def test_file_tools_add_guidance_and_leave_the_input_alone():
    payload = {"messages": [{"role": "user", "content": "what is in Downloads?"}]}
    result = with_file_tools(payload)
    assert "~/Downloads" in result["messages"][0]["content"]
    assert [tool["function"]["name"] for tool in result["tools"]] == list(FILE_TOOLS)
    assert "tools" not in payload and len(payload["messages"]) == 1


def test_web_and_file_tools_can_be_offered_together():
    payload = {"messages": [{"role": "user", "content": "x"}]}
    result = with_file_tools(with_web_tools(payload, "2026-11-25"))
    names = [tool["function"]["name"] for tool in result["tools"]]
    assert names == [*WEB_TOOLS, *FILE_TOOLS]
    system = result["messages"][0]
    assert system["role"] == "system" and "web_search" in system["content"]
    assert "list_folder" in system["content"] and len(result["messages"]) == 2


def test_file_calls_are_described_in_plain_words():
    assert describe_file_call("list_folder", {"path": "~/Downloads"}) == (
        "List the folder ~/Downloads"
    )
    assert describe_file_call("search_files", {"folder": "~", "query": "cv"}) == (
        "Search the files in ~ for “cv”"
    )
    assert describe_file_call("read_file", {}) == "Read the file "
    assert describe_file_call("mystery", {}) == "Use mystery"
