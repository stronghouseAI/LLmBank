"""Agents as RPG-style characters: the character sheet, skills, stats and levels.

Pure data and rules only. Running agents lives in services/quest_runner.py and
executing skills in services/skills.py. See docs/adr/0005-agent-shop.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from stronghouse.core.skill_catalog import SKILLS

# Any agent may equip any skill (phil, 2026-11-23: build agents from the bottom up).
# Small local models do get confused by long tool lists, so the UI suggests keeping
# to this many; it is advice, not a rule.
RECOMMENDED_MAX_SKILLS = 5
STAT_MIN, STAT_MAX = 1, 10
XP_PER_QUEST = 10
XP_PER_LEVEL = 50
MAX_JOURNAL_NOTES_IN_PROMPT = 20


@dataclass(frozen=True)
class Agent:
    """The character sheet."""

    name: str
    portrait: str  # an emoji or short symbol
    personality: str
    skills: list[str]
    creativity: int = 3  # 1-10, maps to sampling temperature
    diligence: int = 5  # 1-10, maps to how many steps a quest may take
    xp: int = 0
    id: int | None = None

    @property
    def level(self) -> int:
        return 1 + self.xp // XP_PER_LEVEL


def validate_agent(agent: Agent) -> None:
    """Raise ValueError with a readable message if the character sheet is invalid."""
    if not agent.name.strip():
        raise ValueError("An agent needs a name.")
    unknown = [skill for skill in agent.skills if skill not in SKILLS]
    if unknown:
        raise ValueError(f"Unknown skills: {', '.join(unknown)}")
    for stat_name in ("creativity", "diligence"):
        value = getattr(agent, stat_name)
        if not STAT_MIN <= value <= STAT_MAX:
            raise ValueError(f"{stat_name} must be between {STAT_MIN} and {STAT_MAX}.")


def temperature_for(agent: Agent) -> float:
    """Creativity 1 -> 0.1 (precise), 10 -> 1.0 (inventive)."""
    return round(agent.creativity / 10, 2)


def max_steps_for(agent: Agent) -> int:
    """Diligence 1 -> 4 steps, 10 -> 22 steps. Each step is one model reply."""
    return 2 + agent.diligence * 2


def system_prompt_for(agent: Agent, journal: list[str], home_folder: str) -> str:
    """Turn the character sheet and recent journal notes into model instructions.

    home_folder is stated explicitly because models otherwise guess paths like
    "/Downloads" for "my downloads folder" and waste an approval on a wrong path.
    """
    lines = [
        f"You are {agent.name}, a level {agent.level} agent.",
    ]
    if agent.personality.strip():
        lines.append(f"Your job and personality: {agent.personality.strip()}")
    lines += [
        "You work for the user on their own computer, running a local model.",
        f"The user's home folder is {home_folder}. Always use full paths starting with "
        f"{home_folder}, e.g. {home_folder}/Downloads for their downloads folder.",
        "Use your tools when they help. Every action on the computer is shown to the user, "
        "who approves or denies it. If an action is denied, find another way or explain.",
        "When the task is done, reply with a clear final answer and no tool calls.",
    ]
    recent = journal[-MAX_JOURNAL_NOTES_IN_PROMPT:]
    if recent:
        lines.append("Notes from your journal:")
        lines += [f"- {note}" for note in recent]
    return "\n".join(lines)


def tool_definitions(agent: Agent) -> list[dict]:
    """OpenAI-style tool list for the agent's equipped skills."""
    return [
        {
            "type": "function",
            "function": {
                "name": SKILLS[name].name,
                "description": SKILLS[name].description,
                "parameters": SKILLS[name].parameters,
            },
        }
        for name in agent.skills
    ]


class QuestState(StrEnum):
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"  # the app restarted mid-quest


class StepKind(StrEnum):
    TASK = "task"
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    APPROVAL = "approval"
    REVIEW = "review"
    ERROR = "error"


@dataclass(frozen=True)
class QuestStep:
    kind: StepKind
    agent_name: str
    text: str


@dataclass(frozen=True)
class PendingAction:
    """A tool call waiting for the user's yes or no."""

    agent_name: str
    skill: str
    arguments: dict


@dataclass(frozen=True)
class Quest:
    task: str
    agent_ids: list[int]  # one agent, or [worker, reviewer] for a party
    state: QuestState
    steps: list[QuestStep] = field(default_factory=list)
    result: str | None = None
    pending: PendingAction | None = None
    created_at: str = ""
    id: int | None = None


REVIEW_APPROVED_MARKER = "APPROVED"


def review_prompt(task: str, work: str) -> str:
    """What the reviewer in a party is asked to do."""
    return (
        f"Another agent was given this task:\n{task}\n\nTheir result:\n{work}\n\n"
        f"Check the result. If it fully completes the task, reply with the single word "
        f"{REVIEW_APPROVED_MARKER} on the first line, then a one-sentence summary. "
        f"Otherwise list the concrete problems to fix."
    )


def revision_prompt(feedback: str) -> str:
    return f"A reviewer checked your work and found problems:\n{feedback}\n\nPlease fix them."


def review_approved(review: str) -> bool:
    first_line = review.strip().splitlines()[0] if review.strip() else ""
    return first_line.strip().strip(".!*").upper() == REVIEW_APPROVED_MARKER


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text as the model produced it; may be malformed


@dataclass(frozen=True)
class ModelTurn:
    """One reply from the model: text, tool calls, or both."""

    content: str
    tool_calls: list[ToolCall]


AGENT_CARD_FORMAT = "stronghouse-agent"
# Version 2 dropped agent classes; version 1 cards still load (their class is ignored).
AGENT_CARD_VERSION = 2


def agent_to_card(agent: Agent) -> dict:
    """A shareable agent file. Progress (xp) and ids are not part of the card."""
    return {
        "format": AGENT_CARD_FORMAT,
        "version": AGENT_CARD_VERSION,
        "name": agent.name,
        "portrait": agent.portrait,
        "personality": agent.personality,
        "skills": list(agent.skills),
        "creativity": agent.creativity,
        "diligence": agent.diligence,
    }


def agent_from_card(card: dict) -> Agent:
    """Read an agent file, dropping unknown skills so cards from newer versions still load.

    Raises:
        ValueError: if the card is not a valid agent card.
    """
    if card.get("format") != AGENT_CARD_FORMAT:
        raise ValueError("This is not a StronghouseAI agent card.")
    try:
        agent = Agent(
            name=str(card["name"]),
            portrait=str(card.get("portrait", "")),
            personality=str(card.get("personality", "")),
            skills=[skill for skill in card.get("skills", []) if skill in SKILLS],
            creativity=int(card.get("creativity", 3)),
            diligence=int(card.get("diligence", 5)),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"The agent card is missing or has a bad field: {error}") from error
    validate_agent(agent)
    return agent
