"""The skill library: every tool an agent can equip, grouped by category.

Pure definitions only; services/skills/ carries them out. Every skill that
touches the computer or the internet needs the user's approval each time.
Only skills that read nothing outside the app (notes, maths, the clock) run
without asking. See docs/adr/0006-skill-library.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SkillCategory(StrEnum):
    FILES = "files"
    WEB = "web"
    DOCUMENTS = "documents"
    DESKTOP = "desktop"
    UTILITY = "utility"


@dataclass(frozen=True)
class SkillSpec:
    """A tool an agent can use. `parameters` is a JSON Schema object."""

    name: str
    title: str
    description: str
    parameters: dict
    needs_approval: bool
    category: SkillCategory


def _schema(required: list[str], **properties: str) -> dict:
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "description": text} for name, text in properties.items()
        },
        "required": required,
    }


_F, _W, _D, _K, _U = (
    SkillCategory.FILES,
    SkillCategory.WEB,
    SkillCategory.DOCUMENTS,
    SkillCategory.DESKTOP,
    SkillCategory.UTILITY,
)

SKILLS: dict[str, SkillSpec] = {
    spec.name: spec
    for spec in [
        # ----- files -----
        SkillSpec("read_file", "Read file", "Read a text file and return its contents.",
                  _schema(["path"], path="Full path to the file"), True, _F),
        SkillSpec("list_folder", "List folder", "List the files and folders inside a folder.",
                  _schema(["path"], path="Full path to the folder"), True, _F),
        SkillSpec("search_files", "Search files",
                  "Find text files under a folder whose name or contents contain the query.",
                  _schema(["folder", "query"], folder="Folder to search in",
                          query="Text to look for"), True, _F),
        SkillSpec("write_file", "Write file",
                  "Create or overwrite a text file with the given content.",
                  _schema(["path", "content"], path="Full path to the file",
                          content="Full text to write"), True, _F),
        SkillSpec("move_file", "Move or rename",
                  "Move or rename a file or folder. Never overwrites an existing file.",
                  _schema(["source", "destination"], source="Current full path",
                          destination="New full path"), True, _F),
        # ----- web -----
        SkillSpec("fetch_web_page", "Fetch web page",
                  "Download a web page and return its readable text.",
                  _schema(["url"], url="An http or https address"), True, _W),
        SkillSpec("read_news_feed", "Read news feed",
                  "Read an RSS or Atom news feed and return the latest headlines with links.",
                  _schema(["url"], url="Address of the RSS or Atom feed"), True, _W),
        SkillSpec("web_search", "Search the web",
                  "Search the web (DuckDuckGo) and return the top results with links.",
                  _schema(["query"], query="What to search for"), True, _W),
        # ----- documents -----
        SkillSpec("read_pdf", "Read PDF", "Extract the text of a PDF file.",
                  _schema(["path"], path="Full path to the PDF"), True, _D),
        SkillSpec("take_note", "Take note",
                  "Save a short note to your own journal so you remember it in future quests.",
                  _schema(["note"], note="What to remember"), False, _D),
        # ----- desktop -----
        SkillSpec("open_with_app", "Open with app",
                  "Open a file, folder or web address in the user's default app.",
                  _schema(["target"], target="Full path or web address"), True, _K),
        SkillSpec("copy_to_clipboard", "Copy to clipboard",
                  "Put text on the user's clipboard so they can paste it anywhere.",
                  _schema(["text"], text="The text to copy"), True, _K),
        SkillSpec("run_command", "Run command",
                  "Run a shell command on the user's computer and return its output.",
                  _schema(["command"], command="The shell command to run"), True, _K),
        # ----- utility -----
        SkillSpec("calculator", "Calculator",
                  "Work out a maths expression exactly, e.g. '(17.5 * 12) / 3' or 'sqrt(2)'.",
                  _schema(["expression"], expression="The maths expression"), False, _U),
        SkillSpec("current_datetime", "Date and time",
                  "Get today's date, the day of the week and the current local time.",
                  _schema([]), False, _U),
    ]
}  # fmt: skip
