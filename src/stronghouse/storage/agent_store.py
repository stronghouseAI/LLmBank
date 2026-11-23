"""SQLite storage for agents, their journals and quest logs."""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from stronghouse.core.agents import Agent, Quest, QuestState, QuestStep, StepKind
from stronghouse.storage.sqlite_db import open_database

_MIGRATIONS_PACKAGE = "stronghouse.storage.agent_migrations"
_UNFINISHED_STATES = (QuestState.RUNNING.value, QuestState.WAITING_APPROVAL.value)
QUEST_HISTORY_LIMIT = 50


class AgentStore:
    def __init__(self, database_path: Path) -> None:
        self._lock = threading.Lock()
        self._connection = open_database(database_path, _MIGRATIONS_PACKAGE)
        self._mark_unfinished_quests_interrupted()

    # ----- agents -----

    def save_agent(self, agent: Agent) -> Agent:
        """Insert a new agent (id is None) or update an existing one."""
        values = (
            agent.name.strip(), agent.portrait, agent.personality,
            json.dumps(agent.skills), agent.creativity, agent.diligence, agent.xp,
        )  # fmt: skip
        with self._lock, self._connection:
            if agent.id is None:
                cursor = self._connection.execute(
                    """INSERT INTO agents (name, portrait, personality, skills,
                                           creativity, diligence, xp)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    values,
                )
                agent_id = cursor.lastrowid
            else:
                self._connection.execute(
                    """UPDATE agents SET name = ?, portrait = ?, personality = ?,
                       skills = ?, creativity = ?, diligence = ?, xp = ? WHERE id = ?""",
                    (*values, agent.id),
                )
                agent_id = agent.id
        return self.get_agent(agent_id)  # type: ignore[return-value]

    def get_agent(self, agent_id: int) -> Agent | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM agents WHERE id = ?", (agent_id,)
            ).fetchone()
        return _to_agent(row) if row else None

    def list_agents(self) -> list[Agent]:
        with self._lock:
            rows = self._connection.execute("SELECT * FROM agents ORDER BY id").fetchall()
        return [_to_agent(row) for row in rows]

    def delete_agent(self, agent_id: int) -> None:
        with self._lock, self._connection:
            self._connection.execute("DELETE FROM agents WHERE id = ?", (agent_id,))

    def add_xp(self, agent_id: int, amount: int) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE agents SET xp = xp + ? WHERE id = ?", (amount, agent_id)
            )

    # ----- journal -----

    def add_note(self, agent_id: int, note: str) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO journal_notes (agent_id, note, created_at) VALUES (?, ?, ?)",
                (agent_id, note.strip(), _now()),
            )

    def journal(self, agent_id: int) -> list[dict]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, note, created_at FROM journal_notes WHERE agent_id = ? ORDER BY id",
                (agent_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_note(self, agent_id: int, note_id: int) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "DELETE FROM journal_notes WHERE id = ? AND agent_id = ?", (note_id, agent_id)
            )

    # ----- quests -----

    def create_quest(self, task: str, agent_ids: list[int]) -> Quest:
        created_at = _now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                "INSERT INTO quests (task, agent_ids, state, created_at) VALUES (?, ?, ?, ?)",
                (task, json.dumps(agent_ids), QuestState.RUNNING.value, created_at),
            )
        return Quest(
            task=task,
            agent_ids=agent_ids,
            state=QuestState.RUNNING,
            created_at=created_at,
            id=cursor.lastrowid,
        )

    def save_quest(self, quest: Quest) -> None:
        """Persist state, log and result. The pending action is kept in memory only."""
        steps = json.dumps([asdict(step) for step in quest.steps])
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE quests SET state = ?, steps = ?, result = ? WHERE id = ?",
                (quest.state.value, steps, quest.result, quest.id),
            )

    def get_quest(self, quest_id: int) -> Quest | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM quests WHERE id = ?", (quest_id,)
            ).fetchone()
        return _to_quest(row) if row else None

    def list_quests(self, limit: int = QUEST_HISTORY_LIMIT) -> list[Quest]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM quests ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_to_quest(row) for row in rows]

    def close(self) -> None:
        self._connection.close()

    def _mark_unfinished_quests_interrupted(self) -> None:
        # A quest's worker thread does not survive a restart; say so honestly.
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE quests SET state = ? WHERE state IN (?, ?)",
                (QuestState.INTERRUPTED.value, *_UNFINISHED_STATES),
            )


def _to_agent(row) -> Agent:
    return Agent(
        id=row["id"],
        name=row["name"],
        portrait=row["portrait"],
        personality=row["personality"],
        skills=json.loads(row["skills"]),
        creativity=row["creativity"],
        diligence=row["diligence"],
        xp=row["xp"],
    )


def _to_quest(row) -> Quest:
    return Quest(
        id=row["id"],
        task=row["task"],
        agent_ids=json.loads(row["agent_ids"]),
        state=QuestState(row["state"]),
        steps=[
            QuestStep(StepKind(s["kind"]), s["agent_name"], s["text"])
            for s in json.loads(row["steps"])
        ],
        result=row["result"],
        created_at=row["created_at"],
    )


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
