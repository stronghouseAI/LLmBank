"""The terminal client (`shai`), run against the real app with a fake engine and hub."""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stronghouse.app import build_services, create_app
from stronghouse.cli import commands
from stronghouse.cli.client import ApiClient
from stronghouse.config import load_settings
from stronghouse.core.agents import Agent, ModelTurn, ToolCall
from stronghouse.core.models import HardwareProfile
from stronghouse.services.engine import EngineState, EngineStatus
from stronghouse.services.huggingface import RemoteFile, RemoteModel
from stronghouse.services.quest_runner import QuestRunner
from stronghouse.services.skills import SkillExecutor
from tests.gguf_builder import moe_model_spec, write_gguf

GIB = 1024**3


class FakeEngine:
    """Loads instantly, streams a thinking-then-answer reply, and plays a quest script."""

    def __init__(self):
        self.started = None
        self.chat_bodies: list[dict] = []
        self.turns: list[ModelTurn] = []

    def version(self):
        return "test"

    def start(self, model_path, plan):
        self.started = (model_path, plan)
        return self.status()

    def stop(self):
        self.started = None

    def status(self):
        state = EngineState.READY if self.started else EngineState.STOPPED
        model_path, plan = self.started or (None, None)
        return EngineStatus(state, model_path and str(model_path), plan, [], [])

    def stream_chat(self, payload):
        self.chat_bodies.append(payload)
        for delta in ({"reasoning_content": "Seventeen times"}, {"reasoning_content": " 23."},
                      {"content": "17 × 23 = "}, {"content": "391"}):  # fmt: skip # noqa: RUF001
            yield f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    def complete_chat(self, messages, tools, temperature):
        return self.turns.pop(0)


class FakeHub:
    def search_models(self, query, filters=None):
        return [
            RemoteModel("Qwen/Qwen3-8B-GGUF", 900, 5, None, [], 8_000_000_000, "8B"),
            RemoteModel("Qwen/Qwen3-32B-GGUF", 800, 5, None, [], 32_000_000_000, "32B"),
        ]

    def list_gguf_files(self, repo_id):
        if "32B" in repo_id:
            return [RemoteFile("Qwen3-32B-Q4_K_M.gguf", 19 * GIB, None)]
        return [
            RemoteFile("Qwen3-8B-Q4_K_M.gguf", 5 * GIB, None),
            RemoteFile("Qwen3-8B-Q8_0.gguf", int(8.7 * GIB), None),
            RemoteFile("Qwen3-8B-BF16.gguf", 24 * GIB, None),
        ]


@pytest.fixture
def shai(tmp_path: Path, capsys, monkeypatch):
    settings = load_settings({"STRONGHOUSE_DATA_DIR": str(tmp_path)})
    services = build_services(settings)
    services.engine = FakeEngine()
    services.hub = FakeHub()
    services.probe_hardware = lambda: HardwareProfile(
        "GPU", 8 * GIB, 8 * GIB, 16 * GIB, 12 * GIB, 8
    )
    services.quests = QuestRunner(
        services.agents, services.engine, SkillExecutor(services.agents.add_note)
    )
    for name in ("qwen3-8b-q4.gguf", "deepseek-v2-lite-q4.gguf"):
        folder = settings.archive_dir / "owner" / name.split("-q4")[0]
        folder.mkdir(parents=True)
        write_gguf(folder / name, *moe_model_spec(blocks=4))
    services.indexer.rescan()

    with TestClient(create_app(services)) as http:
        api = ApiClient(http)

        def run(*argv: str, answers: list[str] | None = None) -> tuple[int, str]:
            if answers is not None:  # typed replies for prompts
                replies = iter(answers)
                monkeypatch.setattr(commands, "prompt_user", lambda _prompt: next(replies))
            code = commands.run(list(argv), settings, api=api)
            output = capsys.readouterr()
            return code, output.out + output.err

        yield run, services


def test_vault_ls_and_plan_by_name(shai):
    run, _ = shai
    code, out = run("vault", "ls")
    assert code == 0 and "qwen3-8b-q4.gguf" in out and "deepseek-v2-lite-q4.gguf" in out
    code, out = run("vault", "plan", "qwen3", "--ctx", "4096", "--kv", "q8_0")
    assert code == 0 and "strategy" in out and "4096 tokens" in out and "llama-server" in out


def test_unknown_or_ambiguous_model_names_explain_themselves(shai):
    run, _ = shai
    code, out = run("vault", "load", "mistral")
    assert code == 1 and "No model in your vault" in out
    code, out = run("vault", "load", "q4")
    assert code == 1 and "matches several models" in out


def test_tables_line_up_with_colour_on(monkeypatch):
    from stronghouse.cli import render

    monkeypatch.setattr(render, "use_colour", lambda stream=None: True)
    text = render.table(["id", "file"], [["1234", render.paint("qwen3-8b.gguf", "bold")]])
    header, row = (re.sub(r"\033\[[0-9;]*m", "", line) for line in text.splitlines())
    assert header.index("FILE") == row.index("qwen3")


def test_load_waits_until_ready_then_unload(shai):
    run, services = shai
    code, out = run("load", "deepseek")
    assert code == 0 and "Ready." in out
    assert "deepseek" in str(services.engine.started[0])
    run("unload")
    assert services.engine.started is None


def test_discover_with_max_gb_shows_the_best_file_that_fits(shai):
    run, _ = shai
    code, out = run("discover", "qwen3", "--max-gb", "8", "--moe")
    assert code == 0
    assert "Qwen3-8B-Q4_K_M.gguf" in out  # 8.7 GB Q8_0 is over the limit
    assert "Qwen3-32B" not in out  # its only file is 19 GB


def test_files_marks_fit_and_hides_too_big(shai):
    run, _ = shai
    _, out = run("files", "Qwen/Qwen3-8B-GGUF")
    assert "fits in GPU" in out and "GPU + RAM" in out and "BF16" not in out
    assert "1 file(s) hidden" in out
    _, out = run("files", "Qwen/Qwen3-8B-GGUF", "--all")
    assert "BF16" in out and "too big" in out


def test_ask_streams_thinking_then_answer(shai):
    run, services = shai
    run("load", "qwen3")
    code, out = run("ask", "--no-think", "What", "is", "17", "times", "23?")
    assert code == 0 and "thought for" in out and "17 × 23 = 391" in out  # noqa: RUF001
    assert services.engine.chat_bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_quest_asks_before_each_action(shai, tmp_path: Path):
    run, services = shai
    note = tmp_path / "note.txt"
    note.write_text("The cube is green.")
    services.agents.save_agent(
        Agent(name="Reader", portrait="📖", personality="", skills=["read_file"])
    )
    services.engine.started = ("model", None)
    services.engine.turns = [
        ModelTurn("", [ToolCall("c1", "read_file", json.dumps({"path": str(note)}))]),
        ModelTurn("The note says the cube is green.", []),
    ]
    code, out = run("quest", "reader", "What", "does", "the", "note", "say?", answers=["y"])
    assert code == 0
    assert "wants to use read_file" in out and "Allowed." in out
    assert "The note says the cube is green." in out


@pytest.mark.parametrize("words", [(), ("help",), ("-h",), ("-help",), ("--help",)])
def test_help_shows_the_full_reference(shai, words):
    run, _ = shai
    code, out = run(*words)
    assert code == 0
    for command in (
        "vault ls",
        "load MODEL",
        "discover QUERY",
        "--max-gb",
        "quest AGENT",
        "EXAMPLES",
    ):
        assert command in out


def test_reference_lists_every_command():
    from stronghouse.cli.reference import SECTIONS

    documented = " ".join(usage for _, rows in SECTIONS for usage, _ in rows)
    parser_commands = commands.build_parser()._subparsers._group_actions[0].choices
    for name in parser_commands:
        assert name in documented, f"{name} is missing from shai help"


def test_single_dash_help_on_a_command(shai):
    run, _ = shai
    with pytest.raises(SystemExit) as exit_info:
        run("discover", "-help")
    assert exit_info.value.code == 0


def test_names_match_loosely_and_chat_takes_a_first_message(shai):
    run, services = shai
    code, out = run("load", "QWEN3_8B")  # file is qwen3-8b-q4.gguf
    assert code == 0 and "Ready." in out
    code, out = run("chat", "how", "are", "you?", answers=["/exit"])
    assert code == 0 and "391" in out
    assert services.engine.chat_bodies[-1]["messages"][0]["content"] == "how are you?"


def test_stronghouse_reuses_a_running_app_instead_of_clashing_on_the_port(monkeypatch):
    import stronghouse.__main__ as entry

    opened = []
    monkeypatch.setattr(entry, "app_is_running", lambda _settings: True)
    monkeypatch.setattr(entry.webbrowser, "open", opened.append)
    monkeypatch.setattr(entry.uvicorn, "run", lambda *a, **k: pytest.fail("started a second app"))
    entry.serve()
    assert opened == ["http://127.0.0.1:8765"]
