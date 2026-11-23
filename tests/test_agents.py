"""Agent shop: character rules, storage, skills and quests (with a scripted fake model)."""

import json
import time
from pathlib import Path

import pytest

from stronghouse.core.agents import (
    MAX_SKILL_SLOTS,
    STARTER_AGENTS,
    Agent,
    AgentClass,
    ModelTurn,
    QuestState,
    StepKind,
    ToolCall,
    agent_from_card,
    agent_to_card,
    max_steps_for,
    review_approved,
    system_prompt_for,
    temperature_for,
    validate_agent,
)
from stronghouse.services.engine import EngineError
from stronghouse.services.quest_runner import DENIED_RESULT, QuestError, QuestRunner
from stronghouse.services.skills import SkillError, SkillExecutor
from stronghouse.storage.agent_store import AgentStore


def make_agent(**changes) -> Agent:
    values = dict(
        name="Mira",
        agent_class=AgentClass.SCOUT,
        portrait="🧭",
        personality="Precise.",
        skills=["read_file", "write_file", "take_note"],
        creativity=3,
        diligence=2,
    )
    values.update(changes)
    return Agent(**values)


# ----- character rules -----


def test_stats_map_to_real_settings():
    agent = make_agent(creativity=7, diligence=5)
    assert temperature_for(agent) == 0.7
    assert max_steps_for(agent) == 12


def test_level_grows_with_xp():
    assert make_agent(xp=0).level == 1
    assert make_agent(xp=120).level == 3


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"name": "  "}, "name"),
        ({"skills": ["fly"]}, "Unknown skills"),
        ({"creativity": 11}, "creativity"),
        (
            {
                "skills": [
                    "read_file",
                    "write_file",
                    "take_note",
                    "run_command",
                    "list_folder",
                    "search_files",
                ]
            },
            f"at most {MAX_SKILL_SLOTS}",
        ),
    ],
)
def test_invalid_character_sheets_are_rejected(changes, message):
    with pytest.raises(ValueError, match=message):
        validate_agent(make_agent(**changes))


def test_system_prompt_includes_identity_and_journal():
    prompt = system_prompt_for(make_agent(), ["The user's projects live in ~/code"])
    assert "You are Mira" in prompt
    assert "scout" in prompt
    assert "~/code" in prompt


def test_agent_card_round_trip_drops_unknown_skills():
    card = agent_to_card(make_agent(xp=500))
    assert "xp" not in card
    card["skills"].append("teleport")
    agent = agent_from_card(card)
    assert agent.skills == ["read_file", "write_file", "take_note"]
    assert agent.xp == 0


def test_bad_agent_card_is_rejected():
    with pytest.raises(ValueError, match="not a StronghouseAI agent card"):
        agent_from_card({"name": "x"})


def test_starter_agents_are_valid():
    for agent in STARTER_AGENTS:
        validate_agent(agent)


def test_review_marker():
    assert review_approved("APPROVED\nLooks good.")
    assert review_approved("**Approved.**")
    assert not review_approved("Not approved: the file is missing.")


# ----- storage -----


@pytest.fixture
def store(tmp_path: Path):
    agent_store = AgentStore(tmp_path / "agents.sqlite3")
    yield agent_store
    agent_store.close()


def test_save_update_and_delete_agent(store: AgentStore):
    saved = store.save_agent(make_agent())
    assert saved.id is not None
    store.add_xp(saved.id, 10)
    assert store.get_agent(saved.id).xp == 10

    from dataclasses import replace

    renamed = store.save_agent(replace(store.get_agent(saved.id), name="Mira II"))
    assert renamed.name == "Mira II" and renamed.xp == 10

    store.add_note(saved.id, "remember this")
    store.delete_agent(saved.id)
    assert store.list_agents() == []
    assert store.journal(saved.id) == []  # journal goes with the agent


def test_unfinished_quests_become_interrupted_after_restart(tmp_path: Path):
    path = tmp_path / "agents.sqlite3"
    first = AgentStore(path)
    quest = first.create_quest("do it", [1])
    first.close()
    assert AgentStore(path).get_quest(quest.id).state is QuestState.INTERRUPTED


# ----- skills -----


def test_file_skills(tmp_path: Path):
    notes: list[tuple[int, str]] = []
    skills = SkillExecutor(lambda agent_id, note: notes.append((agent_id, note)))
    target = tmp_path / "sub" / "hello.txt"

    assert "Wrote" in skills.execute(1, "write_file", {"path": str(target), "content": "hi there"})
    assert skills.execute(1, "read_file", {"path": str(target)}) == "hi there"
    assert "sub/" in skills.execute(1, "list_folder", {"path": str(tmp_path)})
    assert str(target) in skills.execute(
        1, "search_files", {"folder": str(tmp_path), "query": "THERE"}
    )
    assert skills.execute(7, "take_note", {"note": "likes tea"}) == "Noted."
    assert notes == [(7, "likes tea")]


def test_run_command_returns_exit_code_and_output():
    result = SkillExecutor(lambda *_: None).execute(1, "run_command", {"command": "echo hello"})
    assert result.startswith("exit code 0")
    assert "hello" in result


def test_skill_errors_are_readable(tmp_path: Path):
    skills = SkillExecutor(lambda *_: None)
    with pytest.raises(SkillError, match="Missing argument"):
        skills.execute(1, "read_file", {})
    with pytest.raises(SkillError):
        skills.execute(1, "read_file", {"path": str(tmp_path / "nope.txt")})


# ----- quests -----


class ScriptedModel:
    """Replies with a fixed script of turns and records what it was sent."""

    def __init__(self, turns: list[ModelTurn]):
        self.turns = list(turns)
        self.calls: list[list[dict]] = []

    def complete_chat(self, messages, tools, temperature):
        self.calls.append([dict(m) for m in messages])
        if not self.turns:
            raise EngineError("No model is loaded and ready yet.")
        return self.turns.pop(0)


def say(text: str) -> ModelTurn:
    return ModelTurn(content=text, tool_calls=[])


def use(name: str, **arguments) -> ModelTurn:
    return ModelTurn(content="", tool_calls=[ToolCall("c1", name, json.dumps(arguments))])


def wait_for_state(runner: QuestRunner, quest_id: int, *states: QuestState, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        quest = runner.get(quest_id)
        if quest.state in states:
            return quest
        time.sleep(0.01)
    raise AssertionError(f"quest stayed {runner.get(quest_id).state}")


def make_runner(store: AgentStore, turns: list[ModelTurn]):
    model = ScriptedModel(turns)
    return QuestRunner(store, model, SkillExecutor(store.add_note)), model


def test_quest_asks_permission_before_touching_files(store: AgentStore, tmp_path: Path):
    agent = store.save_agent(make_agent())
    target = tmp_path / "out.txt"
    runner, _model = make_runner(
        store, [use("write_file", path=str(target), content="hello"), say("Done, file written.")]
    )

    quest = runner.start("Write hello to a file", [agent.id])
    waiting = wait_for_state(runner, quest.id, QuestState.WAITING_APPROVAL)
    assert waiting.pending.skill == "write_file"
    assert not target.exists()  # nothing happens before approval

    runner.decide(quest.id, approve=True)
    done = wait_for_state(runner, quest.id, QuestState.DONE)
    assert target.read_text() == "hello"
    assert done.result == "Done, file written."
    assert store.get_agent(agent.id).xp == 10
    assert store.get_quest(quest.id).state is QuestState.DONE  # persisted


def test_denied_action_is_not_run_and_agent_is_told(store: AgentStore, tmp_path: Path):
    agent = store.save_agent(make_agent())
    target = tmp_path / "out.txt"
    runner, model = make_runner(
        store, [use("write_file", path=str(target), content="x"), say("Understood, I stopped.")]
    )
    quest = runner.start("Write a file", [agent.id])
    wait_for_state(runner, quest.id, QuestState.WAITING_APPROVAL)
    runner.decide(quest.id, approve=False)
    wait_for_state(runner, quest.id, QuestState.DONE)

    assert not target.exists()
    assert model.calls[-1][-1] == {"role": "tool", "tool_call_id": "c1", "content": DENIED_RESULT}


def test_notes_need_no_approval_and_reach_future_quests(store: AgentStore):
    agent = store.save_agent(make_agent())
    runner, _model = make_runner(store, [use("take_note", note="likes short answers"), say("ok")])
    quest = runner.start("Remember I like short answers", [agent.id])
    wait_for_state(runner, quest.id, QuestState.DONE)

    runner2, model2 = make_runner(store, [say("hi")])
    quest2 = runner2.start("Say hi", [agent.id])
    wait_for_state(runner2, quest2.id, QuestState.DONE)
    assert "likes short answers" in model2.calls[0][0]["content"]


def test_unequipped_skill_is_refused_without_asking(store: AgentStore):
    agent = store.save_agent(make_agent(skills=["read_file"]))
    runner, model = make_runner(store, [use("run_command", command="rm -rf ~"), say("Sorry.")])
    quest = runner.start("Clean up", [agent.id])
    wait_for_state(runner, quest.id, QuestState.DONE)
    assert "do not have the skill" in model.calls[-1][-1]["content"]


def test_running_out_of_steps_fails_the_quest(store: AgentStore, tmp_path: Path):
    agent = store.save_agent(make_agent(diligence=1, skills=["take_note"]))
    runner, _ = make_runner(store, [use("take_note", note="again")] * 10)
    quest = runner.start("Loop forever", [agent.id])
    failed = wait_for_state(runner, quest.id, QuestState.FAILED)
    assert any("Ran out of steps" in step.text for step in failed.steps)


def test_no_model_loaded_fails_with_clear_message(store: AgentStore):
    agent = store.save_agent(make_agent())
    runner, _ = make_runner(store, [])
    quest = runner.start("Anything", [agent.id])
    failed = wait_for_state(runner, quest.id, QuestState.FAILED)
    assert any("No model is loaded" in step.text for step in failed.steps)


def test_cancel_while_waiting_for_approval(store: AgentStore, tmp_path: Path):
    agent = store.save_agent(make_agent())
    runner, _ = make_runner(store, [use("read_file", path=str(tmp_path / "x"))])
    quest = runner.start("Read", [agent.id])
    wait_for_state(runner, quest.id, QuestState.WAITING_APPROVAL)
    runner.cancel(quest.id)
    wait_for_state(runner, quest.id, QuestState.CANCELLED)
    with pytest.raises(QuestError):
        runner.decide(quest.id, approve=True)


def test_party_reviewer_sends_work_back_once(store: AgentStore):
    worker = store.save_agent(make_agent(name="Quill"))
    reviewer = store.save_agent(make_agent(name="Warden", agent_class=AgentClass.REVIEWER))
    runner, model = make_runner(
        store,
        [
            say("Draft v1"),
            say("The title is missing."),
            say("Draft v2 with title"),
            say("APPROVED\nComplete."),
        ],
    )
    quest = runner.start("Write a note with a title", [worker.id, reviewer.id])
    done = wait_for_state(runner, quest.id, QuestState.DONE)

    assert done.result == "Draft v2 with title"
    assert [s.kind for s in done.steps].count(StepKind.REVIEW) == 2
    revision_request = model.calls[2][-1]["content"]
    assert "The title is missing." in revision_request
    assert store.get_agent(reviewer.id).xp == 10


@pytest.mark.parametrize("agent_ids", [[], [1, 1], [1, 2, 3], [999]])
def test_invalid_party_is_rejected(store: AgentStore, agent_ids):
    store.save_agent(make_agent())
    store.save_agent(make_agent(name="Two"))
    runner, _ = make_runner(store, [])
    with pytest.raises(QuestError):
        runner.start("task", agent_ids)
