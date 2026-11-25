"""The terminal client (`shai`), run against the real app with a fake engine and hub."""

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stronghouse.app import build_services, create_app
from stronghouse.cli import commands, prompting
from stronghouse.cli.client import ApiClient
from stronghouse.config import load_settings
from stronghouse.core.agents import Agent, ModelTurn, ToolCall
from stronghouse.core.models import HardwareProfile
from stronghouse.services.engine import EngineState, EngineStatus
from stronghouse.services.huggingface import RemoteFile, RemoteModel
from stronghouse.services.job_runner import JobRunner
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
        self.replies: list[list[dict]] = []  # scripted deltas; the default answer when empty

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
        default = ({"reasoning_content": "Seventeen times"}, {"reasoning_content": " 23."},
                   {"content": "17 × 23 = "}, {"content": "391"})  # fmt: skip # noqa: RUF001
        for delta in self.replies.pop(0) if self.replies else default:
            yield f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode()
        yield b"data: [DONE]\n\n"

    def complete_chat(self, messages, tools, temperature):
        return self.turns.pop(0)

    def render_prompt(self, payload):
        return "<|system|>" + payload["messages"][0]["content"]


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
    services.jobs = JobRunner(
        services.jobs.store, services.engine, services.load_model_for_job,
        notify=lambda *_: None, output_dir=settings.jobs_output_dir,
    )  # fmt: skip
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
                monkeypatch.setattr(prompting, "prompt_user", lambda _prompt: next(replies))
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
        "agentvault new",
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


def test_plan_shows_speed_forecast_and_flags(shai):
    run, _ = shai
    code, out = run("vault", "plan", "qwen3", "--threads", "6", "--extra=-b 4096")
    assert code == 0 and "tokens/s" in out and "(estimate)" in out
    assert "--threads 6" in out and "-b 4096" in out


def test_job_add_ls_show(shai):
    run, services = shai
    services.jobs.stop(wait_seconds=5)  # run the queue by hand below
    code, out = run("job", "add", "Plan", "my", "week", "--model", "deepseek", "--no-think")
    assert code == 0 and "Queued job 1" in out and "shai job show 1" in out
    _, out = run("job", "ls")
    assert "Plan my week" in out and "queued" in out
    services.jobs.run_next()
    assert "deepseek" in str(services.engine.started[0])
    code, out = run("job", "show", "1")
    assert code == 0 and "17 × 23 = 391" in out  # noqa: RUF001
    code, out = run("job", "cancel", "1")
    assert code == 1 and "already finished" in out


class FakeSkills:
    def __init__(self):
        self.calls = []

    def execute(self, agent_id, skill, arguments):
        self.calls.append((skill, arguments))
        return "1. llama.cpp b9000 released - https://example.org"


def _search_call(query: str) -> list[dict]:
    """A reply that asks for a web search, split into pieces like a real stream."""
    arguments = json.dumps({"query": query})
    return [
        {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "web_search"}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": arguments[:5]}}]},
        {"tool_calls": [{"index": 0, "function": {"arguments": arguments[5:]}}]},
    ]


def test_ask_is_offline_without_web(shai):
    run, services = shai
    run("load", "qwen3")
    run("ask", "hi")
    assert "tools" not in services.engine.chat_bodies[-1]


def test_ask_with_web_asks_before_searching_and_answers_with_the_result(shai):
    run, services = shai
    run("load", "qwen3")
    services.skills = FakeSkills()
    services.engine.replies = [_search_call("llama.cpp release"), [{"content": "It is b9000."}]]
    code, out = run("ask", "--web", "latest", "llama.cpp?", answers=["y"])
    assert code == 0
    assert "Search the web for \u201cllama.cpp release\u201d" in out
    assert "It is b9000." in out
    assert services.skills.calls == [("web_search", {"query": "llama.cpp release"})]
    first, second = services.engine.chat_bodies[-2:]
    assert len(first["tools"]) == 3
    assistant, tool = second["messages"][-2:]
    assert assistant["tool_calls"][0]["function"]["name"] == "web_search"
    assert tool == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "1. llama.cpp b9000 released - https://example.org",
    }


def test_saying_no_keeps_the_model_offline(shai):
    run, services = shai
    run("load", "qwen3")
    services.skills = FakeSkills()
    services.engine.replies = [_search_call("weather"), [{"content": "I could not check."}]]
    code, out = run("ask", "--web", "weather?", answers=["n"])
    assert code == 0 and "Denied." in out and "I could not check." in out
    assert services.skills.calls == []
    assert "did not allow" in services.engine.chat_bodies[-1]["messages"][-1]["content"]


def test_web_stops_being_offered_after_the_step_limit(shai):
    from stronghouse.core.chat_web import MAX_WEB_STEPS

    run, services = shai
    run("load", "qwen3")
    services.skills = FakeSkills()
    services.engine.replies = [_search_call("again")] * MAX_WEB_STEPS + [[{"content": "Done."}]]
    code, out = run("ask", "--web", "loop", answers=["y"] * MAX_WEB_STEPS)
    assert code == 0 and "Done." in out
    assert len(services.skills.calls) == MAX_WEB_STEPS
    assert "tools" not in services.engine.chat_bodies[-1]


def test_web_command_switches_internet_inside_chat(shai):
    run, services = shai
    run("load", "qwen3")
    code, out = run("chat", answers=["/web", "hi", "/web off", "bye", "/exit"])
    assert code == 0 and "Internet is on" in out
    with_web, without = services.engine.chat_bodies[-2:]
    assert "tools" in with_web and "tools" not in without


def _list_call(path: str) -> list[dict]:
    arguments = json.dumps({"path": path})
    function = {"name": "list_folder", "arguments": arguments}
    return [{"tool_calls": [{"index": 0, "id": "f1", "function": function}]}]


def test_ask_with_files_lists_a_folder_after_yes(shai, tmp_path, monkeypatch):
    run, services = shai
    downloads = tmp_path / "home" / "Downloads"
    downloads.mkdir(parents=True)
    (downloads / "invoice.pdf").write_bytes(b"x")
    (downloads / "music").mkdir()
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    run("load", "qwen3")
    services.engine.replies = [_list_call("~/Downloads"), [{"content": "You have 2 items."}]]
    code, out = run("ask", "--files", "what is in my Downloads?", answers=["y"])
    assert code == 0
    assert "look at your files" in out and "List the folder ~/Downloads" in out
    assert "You have 2 items." in out
    first, second = services.engine.chat_bodies[-2:]
    assert first["tools"][0]["function"]["name"] == "list_folder"
    assert second["messages"][-1] == {
        "role": "tool", "tool_call_id": "f1", "content": "music/\ninvoice.pdf",
    }  # fmt: skip


def test_saying_no_keeps_files_private(shai):
    run, services = shai
    run("load", "qwen3")
    services.skills = FakeSkills()
    services.engine.replies = [_list_call("~/Downloads"), [{"content": "I could not look."}]]
    code, out = run("ask", "--files", "list Downloads", answers=[""])
    assert code == 0 and "Denied." in out and "I could not look." in out
    assert services.skills.calls == []
    assert "did not allow" in services.engine.chat_bodies[-1]["messages"][-1]["content"]


def test_tools_that_are_not_switched_on_are_never_run(shai):
    run, services = shai
    run("load", "qwen3")
    services.skills = FakeSkills()
    services.engine.replies = [_search_call("x"), [{"content": "Ok."}]]
    code, out = run("ask", "--files", "hi", answers=[])
    assert code == 0 and "chat cannot use" in out and "Ok." in out
    assert services.skills.calls == []


def test_files_command_switches_file_access_inside_chat(shai):
    run, services = shai
    run("load", "qwen3")
    code, out = run("chat", answers=["/files", "hi", "/files off", "bye", "/exit"])
    assert code == 0 and "File access is on" in out
    with_files, without = services.engine.chat_bodies[-2:]
    assert "list_folder" in str(with_files["tools"])
    assert "tools" not in without and "files" not in without


def test_text_tool_calls_work_for_models_without_native_tools(shai, tmp_path, monkeypatch):
    run, services = shai
    (tmp_path / "home" / "Downloads").mkdir(parents=True)
    (tmp_path / "home" / "Downloads" / "notes.txt").write_text("hi")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    run("load", "qwen3")
    call = '<tool_call>{"name": "list_folder", "arguments": {"path": "~/Downloads"}}</tool_call>'
    services.engine.replies = [
        [{"content": "Let me look. "}, {"content": call[:9]}, {"content": call[9:]}],
        [{"content": "You have notes.txt."}],
    ]
    code, out = run("ask", "--files", "what is in Downloads?", answers=["y"])
    assert code == 0 and "List the folder ~/Downloads" in out and "You have notes.txt." in out
    assert "<tool_call>" not in out and "Let me look." in out
    first, second = services.engine.chat_bodies[-2:]
    assert "<tool_call>" in first["messages"][0]["content"]
    assistant, result = second["messages"][-2:]
    assert assistant == {"role": "assistant", "content": "Let me look. " + call}
    assert result["role"] == "user" and result["content"].endswith("notes.txt")


def test_a_refusal_with_tools_on_shows_a_warning(shai):
    run, services = shai
    run("load", "qwen3")
    services.engine.replies = [[{"content": "I cannot access your personal files."}]]
    code, out = run("ask", "--files", "list Downloads")
    assert code == 0 and "did not use them" in out
    code, out = run("ask", "hi")
    assert "did not use them" not in out


def test_debug_shows_the_prompt_and_raw_output(shai):
    run, services = shai
    run("load", "qwen3")
    call = '<tool_call>{"name": "list_folder", "arguments": {"path": "~"}}</tool_call>'
    services.skills = FakeSkills()
    services.engine.replies = [[{"content": call}], [{"content": "Done."}]]
    code, out = run("ask", "--files", "--debug", "list home", answers=["n"])
    assert code == 0
    assert "[debug] tools sent: list_folder, read_file, search_files, read_pdf" in out
    assert "<|system|>You can look at files" in out
    assert call in out  # raw output is not hidden in debug mode
    assert "[debug] no native tool calls" in out


def test_a_stale_background_app_is_reported(shai, monkeypatch):
    run, _ = shai
    code, out = run("status")
    assert "older install" not in out
    monkeypatch.setattr(commands, "build_id", lambda: "different")
    code, out = run("status")
    assert code == 0 and "older install" in out and "shai stop" in out


# ---------- agent vault ----------


def test_agentvault_new_with_yes_no_questions(shai):
    run, services = shai
    answers = [
        "Researcher",  # name
        "Find facts and save them as Markdown.",  # job
        "y",  # read files
        "y",  # write files
        "n",  # PDFs
        "y",  # internet
        "n",  # commands
        "n",  # apps / clipboard
        "",  # numbers (default no)
        "n",  # notes
        "",  # keep exactly these skills (default yes)
        "y",  # exact work -> temperature 0.2
        "n",  # long tasks -> 10 steps
        "",  # save (default yes)
    ]
    code, out = run("agentvault", "new", answers=answers)
    assert code == 0 and "Saved Researcher" in out
    agent = services.agents.list_agents()[0]
    assert agent.personality == "Find facts and save them as Markdown."
    assert agent.skills == [
        "read_file", "list_folder", "search_files", "write_file", "move_file",
        "fetch_web_page", "read_news_feed", "web_search",
    ]  # fmt: skip
    assert (agent.creativity, agent.diligence) == (2, 4)
    assert "more than 5 skills" in out  # advice, not a rule


def test_agentvault_new_lets_you_trim_the_suggested_skills(shai):
    run, services = shai
    answers = ["Clerk", "", "y", "n", "n", "n", "n", "n", "n", "n",
               "n", "read_file", "", "", ""]  # fmt: skip
    code, out = run("agentvault", "new", answers=answers)
    assert code == 0, out
    assert services.agents.list_agents()[0].skills == ["read_file"]


def test_agentvault_new_in_one_line_and_names_stay_unique(shai):
    run, services = shai
    code, out = run(
        "agentvault", "new", "Newsie", "--skills", "web_search,read_news_feed",
        "--job", "UK headlines", "--temperature", "0.5", "--max-steps", "12",
    )  # fmt: skip
    assert code == 0, out
    agent = services.agents.list_agents()[0]
    assert agent.skills == ["web_search", "read_news_feed"]
    assert (agent.creativity, agent.diligence) == (5, 5)
    code, out = run("agentvault", "new", "newsie", "--skills", "")
    assert code == 1 and "already have an agent" in out
    code, out = run("agentvault", "new", "Other", "--skills", "teleport")
    assert code == 1 and "not a skill" in out


def test_agentvault_show_edit_and_skills(shai):
    run, services = shai
    run("agentvault", "new", "Filer", "--skills", "read_file")
    code, out = run("agentvault", "add-skill", "filer", "write_file", "calculator")
    assert code == 0 and "read_file, write_file, calculator" in out
    code, out = run("agentvault", "remove-skill", "Filer", "calculator")
    assert code == 0 and services.agents.list_agents()[0].skills == ["read_file", "write_file"]
    code, out = run("agentvault", "remove-skill", "Filer", "web_search")
    assert code == 1 and "does not have" in out
    code, out = run("agentvault", "edit", "Filer", "--job", "Tidy my Downloads", "--name", "Tidy")
    assert code == 0 and services.agents.list_agents()[0].name == "Tidy"
    code, out = run("agentvault", "show", "tidy")
    assert code == 0 and "Tidy my Downloads" in out and "temperature  0.3" in out
    code, out = run("agentvault", "ls")
    assert code == 0 and "Tidy" in out
    code, out = run("agentvault", "skills")
    assert code == 0 and "run_command" in out and "asks" in out


def test_agentvault_edit_walks_through_and_enter_keeps_values(shai):
    run, services = shai
    run("agentvault", "new", "Keeper", "--skills", "read_file", "--job", "Keep things")
    code, out = run("agentvault", "edit", "Keeper", answers=["", "", "", "", "20"])
    assert code == 0, out
    agent = services.agents.list_agents()[0]
    assert (agent.name, agent.personality, agent.skills) == ("Keeper", "Keep things", ["read_file"])
    assert agent.diligence == 9


def test_agentvault_rm_asks_first(shai):
    run, services = shai
    run("agentvault", "new", "Temp", "--skills", "")
    code, out = run("agentvault", "rm", "Temp", answers=[""])
    assert code == 0 and "Kept." in out and services.agents.list_agents()
    code, out = run("agentvault", "rm", "Temp", answers=["y"])
    assert code == 0 and not services.agents.list_agents()


def test_agentvault_export_and_import(shai, tmp_path: Path):
    run, services = shai
    run("agentvault", "new", "News Desk", "--skills", "web_search", "--job", "Headlines")
    card = tmp_path / "desk.agent.json"
    code, out = run("agentvault", "export", "News Desk", str(card))
    assert code == 0 and json.loads(card.read_text())["skills"] == ["web_search"]
    code, out = run("agentvault", "export", "News Desk", str(card))
    assert code == 1 and "already exists" in out
    code, out = run("agentvault", "import", str(card))
    assert code == 1 and "--name" in out
    code, out = run("agentvault", "import", str(card), "--name", "Desk Two")
    assert code == 0 and [a.name for a in services.agents.list_agents()] == [
        "News Desk", "Desk Two",
    ]  # fmt: skip


def test_card_filename_is_safe():
    from stronghouse.cli.agentvault import card_filename

    assert card_filename("News Desk!") == "news-desk.agent.json"
    assert card_filename("../..") == "agent.agent.json"


def test_agentvault_run_is_the_same_as_quest(shai, tmp_path: Path):
    run, services = shai
    note = tmp_path / "note.txt"
    note.write_text("Blue.")
    run("agentvault", "new", "Reader", "--skills", "read_file")
    services.engine.started = ("model", None)
    services.engine.turns = [
        ModelTurn("", [ToolCall("c1", "read_file", json.dumps({"path": str(note)}))]),
        ModelTurn("It says blue.", []),
    ]
    code, out = run("agentvault", "run", "Reader", "Read", "the", "note", answers=["n"])
    assert code == 0 and "Denied." in out
