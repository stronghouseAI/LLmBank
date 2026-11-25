"""Tool calls written as text, for models without native tool calling."""

from stronghouse.core.text_tool_calls import (
    find_text_tool_calls,
    looks_like_refusal,
    starts_text_call,
    with_text_call_guidance,
)

FILES = {"list_folder", "read_file"}


def test_tagged_calls_are_read():
    text = 'Let me look.\n<tool_call>{"name": "list_folder", "arguments": {"path": "~/Downloads"}}'
    (call,) = find_text_tool_calls(text + "</tool_call>", FILES)
    assert (call.name, call.parsed_arguments()) == ("list_folder", {"path": "~/Downloads"})
    (unclosed,) = find_text_tool_calls(text, FILES)  # the model stopped before the end tag
    assert unclosed.name == "list_folder"


def test_nemotron_lists_and_bare_json_are_read():
    nemotron = '<TOOLCALL>[{"name": "read_file", "arguments": {"path": "a.txt"}}]</TOOLCALL>'
    assert [c.name for c in find_text_tool_calls(nemotron, FILES)] == ["read_file"]
    bare = '```json\n{"name": "list_folder", "arguments": {"path": "~"}}\n```'
    assert [c.name for c in find_text_tool_calls(bare, FILES)] == ["list_folder"]


def test_only_offered_tools_and_real_calls_count():
    assert find_text_tool_calls('<tool_call>{"name": "run_command", "arguments": {}}', FILES) == []
    assert find_text_tool_calls('Use {"name": "list_folder"} like this.', FILES) == []
    assert find_text_tool_calls("Your Downloads folder is ~/Downloads.", FILES) == []


def test_where_a_text_call_starts():
    assert starts_text_call("ok <tool_call>{") == 3
    assert starts_text_call("plain answer") == -1


def test_refusals_are_spotted():
    assert looks_like_refusal("I cannot access your personal files, including the folder")
    assert looks_like_refusal("I don't have access to the internet.")
    assert not looks_like_refusal("I can't see why that would fail.")
    assert not looks_like_refusal("Your folder has 3 files.")


def test_guidance_lists_tools_with_arguments():
    payload = {"messages": [{"role": "user", "content": "x"}], "tools": [{"t": 1}]}
    result = with_text_call_guidance(payload, ("list_folder", "read_file"))
    system = result["messages"][0]["content"]
    assert '<tool_call>{"name": "list_folder", "arguments": {"path": "~/Downloads"}}' in system
    assert '- read_file {"path": "Full path to the file"}' in system
    assert result["tools"] == [{"t": 1}]
