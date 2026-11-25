"""Pure rules for letting the chat model look at files on the user's computer.

Chat cannot see the computer unless the user turns on file access (`shai chat --files`
or `/files`). Then the model gets the agents' read-only file skills as tools, and the
user is asked before each one runs. Nothing is written, moved or deleted.
See docs/adr/0015-chat-file-access.md.
"""

from __future__ import annotations

from collections import defaultdict

from stronghouse.core.chat_web import offer_tools

# Read-only on purpose: chat answers questions about files, agents change them.
FILE_TOOLS: tuple[str, ...] = ("list_folder", "read_file", "search_files", "read_pdf")
# Folder listings and file reads one reply may make before the model must answer.
# Higher than the web limit because "look in Downloads, then open that file" is common.
MAX_FILE_STEPS = 10

FILE_GUIDANCE = (
    "You can look at files on the user's computer with the tools list_folder, read_file, "
    "search_files and read_pdf. Use them when the question is about the user's own files "
    "or folders; otherwise answer directly. Paths may start with ~ for the user's home "
    "folder, for example ~/Downloads or ~/Documents. You can only read, not change "
    "anything. The user approves every use, so make each one count."
)
# What the model hears when the user says no to a file call.
FILE_DENIED = (
    "The user did not allow this. Do not ask again; answer without it and say you "
    "could not look at that file or folder."
)
# How a file call is put to the user; {names} are filled from the call's arguments.
FILE_ACTIONS: dict[str, str] = {
    "list_folder": "List the folder {path}",
    "read_file": "Read the file {path}",
    "search_files": "Search the files in {folder} for “{query}”",
    "read_pdf": "Read the PDF {path}",
}


def is_file_tool(name: str) -> bool:
    return name in FILE_TOOLS


def describe_file_call(name: str, arguments: dict) -> str:
    """The plain sentence the user is asked to allow, e.g. 'List the folder ~/Downloads'."""
    if name not in FILE_ACTIONS:
        return f"Use {name}"
    return FILE_ACTIONS[name].format_map(defaultdict(str, arguments))


def with_file_tools(payload: dict) -> dict:
    """A copy of a chat request that offers the read-only file tools."""
    return offer_tools(payload, FILE_TOOLS, FILE_GUIDANCE)
