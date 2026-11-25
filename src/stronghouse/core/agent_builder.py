"""Building an agent from the terminal: the yes/no questions and the input rules.

Pure rules only; the prompts and printing live in cli/agentvault.py. The questions
narrow which skills an agent gets and how it behaves. There are no presets: every
answer only adds what that one agent needs (phil, 2026-11-25). See
docs/adr/0017-agent-vault-in-the-terminal.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from stronghouse.core.agents import STAT_MAX, STAT_MIN, max_steps_for_diligence
from stronghouse.core.skill_catalog import SKILLS


@dataclass(frozen=True)
class SkillQuestion:
    """A yes/no question; yes adds these skills."""

    text: str
    skills: tuple[str, ...]


SKILL_QUESTIONS: tuple[SkillQuestion, ...] = (
    SkillQuestion("Does it need to read files and folders on your computer?",
                  ("list_folder", "read_file", "search_files")),
    SkillQuestion("Does it need to create, change or move files?", ("write_file", "move_file")),
    SkillQuestion("Does it need to read PDFs?", ("read_pdf",)),
    SkillQuestion("Does it need the internet (search, web pages, news feeds)?",
                  ("web_search", "fetch_web_page", "read_news_feed")),
    SkillQuestion("Does it need to run terminal commands?", ("run_command",)),
    SkillQuestion("Does it need to open apps or copy text to your clipboard?",
                  ("open_with_app", "copy_to_clipboard")),
    SkillQuestion("Does it work with numbers, dates or times?", ("calculator", "current_datetime")),
    SkillQuestion("Should it remember things between tasks (its own notes)?", ("take_note",)),
)  # fmt: skip

EXACT_WORK_QUESTION = "Is the work exact (facts, files, data) rather than creative writing?"
LONG_TASKS_QUESTION = "Will its tasks take many steps (e.g. go through a whole folder)?"

# Behaviour picked by the two questions above, as agent stats (1-10).
CREATIVITY_EXACT, CREATIVITY_CREATIVE = 2, 7  # temperature 0.2 / 0.7
DILIGENCE_SHORT, DILIGENCE_LONG = 4, 8  # 10 / 18 steps per task

YES_WORDS = frozenset({"y", "yes"})
NO_WORDS = frozenset({"n", "no"})


def parse_yes_no(text: str, default: bool) -> bool | None:
    """True for yes, False for no, the default for an empty answer, None if unclear."""
    answer = text.strip().lower()
    if not answer:
        return default
    if answer in YES_WORDS:
        return True
    if answer in NO_WORDS:
        return False
    return None


def skills_from_answers(answers: list[bool]) -> list[str]:
    """The skills that the yes answers add, in catalog order, without repeats."""
    if len(answers) != len(SKILL_QUESTIONS):
        raise ValueError("One answer per question is needed.")
    chosen = {
        skill
        for question, yes in zip(SKILL_QUESTIONS, answers, strict=True)
        if yes
        for skill in question.skills
    }
    return [name for name in SKILLS if name in chosen]


def creativity_for(exact_work: bool) -> int:
    return CREATIVITY_EXACT if exact_work else CREATIVITY_CREATIVE


def diligence_for(long_tasks: bool) -> int:
    return DILIGENCE_LONG if long_tasks else DILIGENCE_SHORT


def parse_skill_picks(text: str, offered: list[str]) -> list[str]:
    """Turn "1,3" or "read_file web_search" into skill names.

    Numbers count from 1 in `offered`; names may be any skill in the catalog.

    Raises:
        ValueError: naming the first pick that is not a skill.
    """
    picks: list[str] = []
    for token in text.replace(",", " ").split():
        if token.isdigit() and 1 <= int(token) <= len(offered):
            name = offered[int(token) - 1]
        elif token in SKILLS:
            name = token
        else:
            raise ValueError(f'"{token}" is not a skill. See them with: shai agentvault skills')
        if name not in picks:
            picks.append(name)
    return picks


def creativity_for_temperature(temperature: float) -> int:
    """Temperature 0.1-1.0 (as shown in the app) to the creativity stat 1-10."""
    creativity = round(temperature * 10)
    if not STAT_MIN <= creativity <= STAT_MAX:
        raise ValueError("Temperature must be between 0.1 and 1.0.")
    return creativity


def diligence_for_max_steps(max_steps: int) -> int:
    """Max steps per task (4-22, as shown in the app) to the diligence stat, rounding down."""
    lowest, highest = max_steps_for_diligence(STAT_MIN), max_steps_for_diligence(STAT_MAX)
    if not lowest <= max_steps <= highest:
        raise ValueError(f"Max steps must be between {lowest} and {highest}.")
    return (max_steps - 2) // 2
