"""HTTP API for the Agent shop: characters, journals and quests."""

from __future__ import annotations

from dataclasses import asdict, replace

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from stronghouse.core.agents import (
    CLASS_DESCRIPTIONS,
    MAX_SKILL_SLOTS,
    SKILLS,
    STARTER_AGENTS,
    STAT_MAX,
    STAT_MIN,
    Agent,
    AgentClass,
    Quest,
    agent_from_card,
    agent_to_card,
    validate_agent,
)
from stronghouse.services.quest_runner import QuestError

router = APIRouter(prefix="/api")


class AgentBody(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    agent_class: AgentClass
    portrait: str = Field("", max_length=16)
    personality: str = Field("", max_length=2000)
    skills: list[str] = []
    creativity: int = Field(3, ge=STAT_MIN, le=STAT_MAX)
    diligence: int = Field(5, ge=STAT_MIN, le=STAT_MAX)


class QuestBody(BaseModel):
    task: str = Field(min_length=1, max_length=10_000)
    agent_ids: list[int]


class DecisionBody(BaseModel):
    approve: bool


@router.get("/agents/catalog")
def catalog() -> dict:
    """Everything the character creator needs: classes, skills, limits and starters."""
    return {
        "classes": [{"id": c.value, "description": d} for c, d in CLASS_DESCRIPTIONS.items()],
        "skills": [asdict(spec) for spec in SKILLS.values()],
        "max_skill_slots": MAX_SKILL_SLOTS,
        "stat_range": [STAT_MIN, STAT_MAX],
        "starters": [agent_to_card(agent) for agent in STARTER_AGENTS],
    }


@router.get("/agents")
def list_agents(request: Request) -> list[dict]:
    return [_agent_json(agent) for agent in _store(request).list_agents()]


@router.post("/agents")
def create_agent(body: AgentBody, request: Request) -> dict:
    agent = _valid(Agent(**body.model_dump()))
    return _agent_json(_store(request).save_agent(agent))


@router.put("/agents/{agent_id}")
def update_agent(agent_id: int, body: AgentBody, request: Request) -> dict:
    existing = _agent(request, agent_id)
    agent = _valid(replace(existing, **body.model_dump()))
    return _agent_json(_store(request).save_agent(agent))


@router.delete("/agents/{agent_id}")
def delete_agent(agent_id: int, request: Request) -> dict:
    _store(request).delete_agent(agent_id)
    return {"deleted": agent_id}


@router.get("/agents/{agent_id}/card")
def export_card(agent_id: int, request: Request) -> dict:
    return agent_to_card(_agent(request, agent_id))


@router.post("/agents/import")
def import_card(card: dict, request: Request) -> dict:
    try:
        agent = agent_from_card(card)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return _agent_json(_store(request).save_agent(agent))


@router.get("/agents/{agent_id}/journal")
def journal(agent_id: int, request: Request) -> list[dict]:
    _agent(request, agent_id)
    return _store(request).journal(agent_id)


@router.delete("/agents/{agent_id}/journal/{note_id}")
def delete_note(agent_id: int, note_id: int, request: Request) -> dict:
    _store(request).delete_note(agent_id, note_id)
    return {"deleted": note_id}


@router.post("/quests")
def start_quest(body: QuestBody, request: Request) -> dict:
    try:
        return _quest_json(_runner(request).start(body.task, body.agent_ids))
    except QuestError as error:
        raise HTTPException(400, str(error)) from error


@router.get("/quests")
def list_quests(request: Request) -> list[dict]:
    return [_quest_json(quest) for quest in _runner(request).list()]


@router.get("/quests/{quest_id}")
def get_quest(quest_id: int, request: Request) -> dict:
    quest = _runner(request).get(quest_id)
    if quest is None:
        raise HTTPException(404, "Quest not found")
    return _quest_json(quest)


@router.post("/quests/{quest_id}/decision")
def decide(quest_id: int, body: DecisionBody, request: Request) -> dict:
    try:
        return _quest_json(_runner(request).decide(quest_id, body.approve))
    except QuestError as error:
        raise HTTPException(409, str(error)) from error


@router.post("/quests/{quest_id}/cancel")
def cancel(quest_id: int, request: Request) -> dict:
    try:
        _runner(request).cancel(quest_id)
    except QuestError as error:
        raise HTTPException(409, str(error)) from error
    return {"cancelled": quest_id}


def _store(request: Request):
    return request.app.state.services.agents


def _runner(request: Request):
    return request.app.state.services.quests


def _agent(request: Request, agent_id: int) -> Agent:
    agent = _store(request).get_agent(agent_id)
    if agent is None:
        raise HTTPException(404, "Agent not found")
    return agent


def _valid(agent: Agent) -> Agent:
    try:
        validate_agent(agent)
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    return agent


def _agent_json(agent: Agent) -> dict:
    return {**asdict(agent), "level": agent.level}


def _quest_json(quest: Quest) -> dict:
    return asdict(quest)
