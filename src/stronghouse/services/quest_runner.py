"""Run quests: an agent (or a party of two) works on a task using the loaded model.

Every skill that touches the computer pauses the quest until the user approves
or denies that exact action. A party is a worker plus a reviewer: the reviewer
checks the worker's result and can send it back for one more revision round.
See docs/adr/0005-agent-shop.md.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Protocol

import httpx

from stronghouse.core.agents import (
    SKILLS,
    XP_PER_QUEST,
    Agent,
    ModelTurn,
    PendingAction,
    Quest,
    QuestState,
    QuestStep,
    StepKind,
    max_steps_for,
    review_approved,
    review_prompt,
    revision_prompt,
    system_prompt_for,
    temperature_for,
    tool_definitions,
)
from stronghouse.services.engine import EngineError
from stronghouse.services.skills import SkillError, SkillExecutor
from stronghouse.storage.agent_store import AgentStore

log = logging.getLogger(__name__)

MAX_REVIEW_ROUNDS = 2
MAX_PARTY_SIZE = 2
LOG_TEXT_LIMIT = 2000
DENIED_RESULT = "The user denied this action. Do not try it again; find another way or explain."


class ChatModel(Protocol):
    def complete_chat(
        self, messages: list[dict], tools: list[dict], temperature: float
    ) -> ModelTurn: ...


class QuestError(ValueError):
    """The quest request or decision is not valid right now."""


class _QuestStopped(Exception):
    """Internal: the quest was cancelled or failed and the worker thread should stop."""


@dataclass
class _LiveQuest:
    quest: Quest
    decided: threading.Event = field(default_factory=threading.Event)
    approved: bool = False
    cancelled: bool = False


class QuestRunner:
    def __init__(self, store: AgentStore, model: ChatModel, skills: SkillExecutor) -> None:
        self._store = store
        self._model = model
        self._skills = skills
        self._live: dict[int, _LiveQuest] = {}
        self._lock = threading.Lock()

    def start(self, task: str, agent_ids: list[int]) -> Quest:
        """Start a quest in the background for one agent or a [worker, reviewer] party.

        Raises:
            QuestError: if the task is empty or the agents are not valid.
        """
        task = task.strip()
        if not task:
            raise QuestError("Give the quest a task.")
        if not 1 <= len(agent_ids) <= MAX_PARTY_SIZE or len(set(agent_ids)) != len(agent_ids):
            raise QuestError(f"A quest needs 1 to {MAX_PARTY_SIZE} different agents.")
        agents = [self._store.get_agent(agent_id) for agent_id in agent_ids]
        if any(agent is None for agent in agents):
            raise QuestError("One of the chosen agents does not exist.")

        quest = self._store.create_quest(task, agent_ids)
        quest = replace(quest, steps=[QuestStep(StepKind.TASK, "you", task)])
        live = _LiveQuest(quest)
        with self._lock:
            self._live[quest.id] = live
        threading.Thread(
            target=self._run, args=(live, agents), daemon=True, name=f"quest-{quest.id}"
        ).start()
        return quest

    def get(self, quest_id: int) -> Quest | None:
        with self._lock:
            live = self._live.get(quest_id)
            if live:
                return live.quest
        return self._store.get_quest(quest_id)

    def list(self) -> list[Quest]:
        with self._lock:
            live = {quest_id: item.quest for quest_id, item in self._live.items()}
        return [live.get(quest.id, quest) for quest in self._store.list_quests()]

    def decide(self, quest_id: int, approve: bool) -> Quest:
        """Answer the pending action of a quest that is waiting for approval.

        Raises:
            QuestError: if the quest is not waiting for a decision.
        """
        with self._lock:
            live = self._live.get(quest_id)
            if live is None or live.quest.state is not QuestState.WAITING_APPROVAL:
                raise QuestError("This quest is not waiting for a decision.")
            live.approved = approve
            live.decided.set()
            return live.quest

    def cancel(self, quest_id: int) -> None:
        with self._lock:
            live = self._live.get(quest_id)
            if live is None:
                raise QuestError("This quest is not running.")
            live.cancelled = True
            live.decided.set()

    # ----- the quest itself -----

    def _run(self, live: _LiveQuest, agents: list[Agent]) -> None:
        try:
            result = self._run_party(live, agents)
            for agent in agents:
                self._store.add_xp(agent.id, XP_PER_QUEST)
            self._finish(live, QuestState.DONE, result)
        except _QuestStopped:
            state = QuestState.CANCELLED if live.cancelled else QuestState.FAILED
            self._finish(live, state, live.quest.result)
        except (EngineError, httpx.HTTPError) as error:
            self._log(live, StepKind.ERROR, "system", f"The model could not answer: {error}")
            self._finish(live, QuestState.FAILED, None)
        except Exception as error:
            # A background thread has no caller to raise to; record it loudly instead.
            log.exception("quest crashed", extra={"quest_id": live.quest.id})
            self._log(live, StepKind.ERROR, "system", f"Unexpected error: {error}")
            self._finish(live, QuestState.FAILED, None)

    def _run_party(self, live: _LiveQuest, agents: list[Agent]) -> str:
        worker = agents[0]
        worker_messages = self._opening_messages(worker, live.quest.task)
        result = self._work(live, worker, worker_messages)
        if len(agents) == 1:
            return result

        reviewer = agents[1]
        for round_number in range(1, MAX_REVIEW_ROUNDS + 1):
            review_messages = self._opening_messages(
                reviewer, review_prompt(live.quest.task, result)
            )
            review = self._work(live, reviewer, review_messages, StepKind.REVIEW)
            if review_approved(review) or round_number == MAX_REVIEW_ROUNDS:
                return result
            worker_messages.append({"role": "user", "content": revision_prompt(review)})
            result = self._work(live, worker, worker_messages)
        return result

    def _opening_messages(self, agent: Agent, task: str) -> list[dict]:
        notes = [entry["note"] for entry in self._store.journal(agent.id)]
        return [
            {"role": "system", "content": system_prompt_for(agent, notes, str(Path.home()))},
            {"role": "user", "content": task},
        ]

    def _work(
        self,
        live: _LiveQuest,
        agent: Agent,
        messages: list[dict],
        final_kind: StepKind = StepKind.MESSAGE,
    ) -> str:
        """Let one agent think and use tools until it gives a final answer."""
        tools = tool_definitions(agent)
        for _ in range(max_steps_for(agent)):
            self._stop_if_cancelled(live)
            turn = self._model.complete_chat(messages, tools, temperature_for(agent))
            messages.append(_assistant_message(turn))
            if not turn.tool_calls:
                self._log(live, final_kind, agent.name, turn.content)
                return turn.content
            if turn.content.strip():
                self._log(live, StepKind.MESSAGE, agent.name, turn.content)
            for call in turn.tool_calls:
                result = self._use_skill(live, agent, call.name, call.arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        self._log(live, StepKind.ERROR, agent.name, "Ran out of steps before finishing.")
        raise _QuestStopped

    def _use_skill(self, live: _LiveQuest, agent: Agent, name: str, raw_arguments: str) -> str:
        if name not in agent.skills:
            return f"You do not have the skill '{name}'. Your skills: {', '.join(agent.skills)}"
        try:
            arguments = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            return f"Your arguments were not valid JSON: {raw_arguments[:200]}"
        if not isinstance(arguments, dict):
            return "Arguments must be a JSON object."

        self._log(live, StepKind.TOOL_CALL, agent.name, _describe_call(name, arguments))
        if SKILLS[name].needs_approval and not self._ask_permission(live, agent, name, arguments):
            self._log(live, StepKind.APPROVAL, "you", "Denied")
            return DENIED_RESULT
        if SKILLS[name].needs_approval:
            self._log(live, StepKind.APPROVAL, "you", "Approved")

        try:
            result = self._skills.execute(agent.id, name, arguments)
        except SkillError as error:
            result = f"Error: {error}"
        self._log(live, StepKind.TOOL_RESULT, agent.name, result)
        return result

    def _ask_permission(self, live: _LiveQuest, agent: Agent, name: str, arguments: dict) -> bool:
        pending = PendingAction(agent_name=agent.name, skill=name, arguments=arguments)
        with self._lock:
            live.decided.clear()
            live.quest = replace(live.quest, state=QuestState.WAITING_APPROVAL, pending=pending)
        live.decided.wait()
        self._stop_if_cancelled(live)
        with self._lock:
            live.quest = replace(live.quest, state=QuestState.RUNNING, pending=None)
            return live.approved

    def _stop_if_cancelled(self, live: _LiveQuest) -> None:
        if live.cancelled:
            self._log(live, StepKind.ERROR, "you", "Quest cancelled.")
            raise _QuestStopped

    def _log(self, live: _LiveQuest, kind: StepKind, agent_name: str, text: str) -> None:
        if len(text) > LOG_TEXT_LIMIT:
            text = text[:LOG_TEXT_LIMIT] + " [...]"
        with self._lock:
            live.quest = replace(
                live.quest, steps=[*live.quest.steps, QuestStep(kind, agent_name, text)]
            )
            quest = live.quest
        self._store.save_quest(quest)

    def _finish(self, live: _LiveQuest, state: QuestState, result: str | None) -> None:
        with self._lock:
            live.quest = replace(live.quest, state=state, result=result, pending=None)
            quest = live.quest
            self._live.pop(quest.id, None)
        self._store.save_quest(quest)


def _assistant_message(turn: ModelTurn) -> dict:
    message: dict = {"role": "assistant", "content": turn.content}
    if turn.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in turn.tool_calls
        ]
    return message


def _describe_call(name: str, arguments: dict) -> str:
    shown = ", ".join(f"{key}={value!r}" for key, value in arguments.items())
    return f"{name}({shown})"
