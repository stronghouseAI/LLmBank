"""Agents as RPG-style characters: the character sheet, skills, stats and levels.

Pure data and rules only. Running agents lives in services/quest_runner.py and
executing skills in services/skills.py. See docs/adr/0005-agent-shop.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# Small local models get confused by long tool lists, so each agent equips a
# limited number of skills. This is both a game rule and a real quality limit.
MAX_SKILL_SLOTS = 5
STAT_MIN, STAT_MAX = 1, 10
XP_PER_QUEST = 10
XP_PER_LEVEL = 50
MAX_JOURNAL_NOTES_IN_PROMPT = 20


class AgentClass(StrEnum):
    SCHOLAR = "scholar"  # research and explanation
    SCRIBE = "scribe"  # writing and editing
    ARCHIVIST = "archivist"  # organising files and notes
    BUILDER = "builder"  # code and commands
    SCOUT = "scout"  # finding things on the computer
    REVIEWER = "reviewer"  # checks another agent's work


CLASS_DESCRIPTIONS = {
    AgentClass.SCHOLAR: "You research questions carefully and explain answers clearly.",
    AgentClass.SCRIBE: "You write and edit text with care for clarity and tone.",
    AgentClass.ARCHIVIST: "You organise files and information neatly and safely.",
    AgentClass.BUILDER: "You write code and use commands to build and fix things.",
    AgentClass.SCOUT: "You search the computer to find files and facts quickly.",
    AgentClass.REVIEWER: "You check work done by others and point out concrete problems.",
}


@dataclass(frozen=True)
class SkillSpec:
    """A tool an agent can use. `parameters` is a JSON Schema object."""

    name: str
    title: str
    description: str
    parameters: dict
    needs_approval: bool


def _schema(required: list[str], **properties: str) -> dict:
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "description": d} for name, d in properties.items()
        },
        "required": required,
    }


SKILLS: dict[str, SkillSpec] = {
    spec.name: spec
    for spec in [
        SkillSpec(
            "read_file", "Read file", "Read a text file and return its contents.",
            _schema(["path"], path="Absolute path or a path starting with ~"), True,
        ),
        SkillSpec(
            "list_folder", "List folder", "List the files and folders inside a folder.",
            _schema(["path"], path="Folder path"), True,
        ),
        SkillSpec(
            "search_files", "Search files",
            "Find text files under a folder whose name or contents contain the query.",
            _schema(["folder", "query"], folder="Folder to search in", query="Text to look for"),
            True,
        ),
        SkillSpec(
            "write_file", "Write file", "Create or overwrite a text file with the given content.",
            _schema(["path", "content"], path="File path", content="Full text to write"), True,
        ),
        SkillSpec(
            "run_command", "Run command",
            "Run a shell command on the user's computer and return its output.",
            _schema(["command"], command="The shell command to run"), True,
        ),
        SkillSpec(
            "take_note", "Take note",
            "Save a short note to your own journal so you remember it in future quests.",
            _schema(["note"], note="What to remember"), False,
        ),
    ]
}  # fmt: skip


@dataclass(frozen=True)
class Agent:
    """The character sheet."""

    name: str
    agent_class: AgentClass
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
    if len(set(agent.skills)) > MAX_SKILL_SLOTS:
        raise ValueError(f"An agent can equip at most {MAX_SKILL_SLOTS} skills.")
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


def system_prompt_for(agent: Agent, journal: list[str]) -> str:
    """Turn the character sheet and recent journal notes into model instructions."""
    lines = [
        f"You are {agent.name}, a level {agent.level} {agent.agent_class.value}.",
        CLASS_DESCRIPTIONS[agent.agent_class],
    ]
    if agent.personality.strip():
        lines.append(f"Personality: {agent.personality.strip()}")
    lines += [
        "You work for the user on their own computer, running a local model.",
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
AGENT_CARD_VERSION = 1

# Starting characters for the shop. Users can recruit them as-is or edit them.
STARTER_AGENTS = [
    Agent(
        name="Mira", agent_class=AgentClass.SCOUT, portrait="🧭",
        personality="Quick, curious and precise. Reports exactly where things are.",
        skills=["list_folder", "search_files", "read_file", "take_note"],
        creativity=2, diligence=5,
    ),
    Agent(
        name="Quill", agent_class=AgentClass.SCRIBE, portrait="🪶",
        personality="Warm and clear. Writes in plain English and keeps things short.",
        skills=["read_file", "write_file", "take_note"],
        creativity=6, diligence=4,
    ),
    Agent(
        name="Forge", agent_class=AgentClass.BUILDER, portrait="🔨",
        personality="Practical and careful. Explains each command before running it.",
        skills=["list_folder", "read_file", "write_file", "run_command", "take_note"],
        creativity=3, diligence=7,
    ),
    Agent(
        name="Warden", agent_class=AgentClass.REVIEWER, portrait="🛡️",
        personality="Strict but fair. Checks facts and points out exactly what is wrong.",
        skills=["read_file", "list_folder"],
        creativity=2, diligence=4,
    ),
]  # fmt: skip


def agent_to_card(agent: Agent) -> dict:
    """A shareable agent file. Progress (xp) and ids are not part of the card."""
    return {
        "format": AGENT_CARD_FORMAT,
        "version": AGENT_CARD_VERSION,
        "name": agent.name,
        "agent_class": agent.agent_class.value,
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
            agent_class=AgentClass(card["agent_class"]),
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
