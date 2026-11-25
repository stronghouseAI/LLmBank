"""`shai agentvault`: build, change, share and run agents from the terminal.

The agent counterpart of `shai vault`. Agents live in the app (the same ones the
web app's Agents tab shows); these commands only call its HTTP API. The yes/no
questions behind `agentvault new` are in core/agent_builder.py. See
docs/adr/0017-agent-vault-in-the-terminal.md.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from stronghouse.cli import prompting, render
from stronghouse.cli.client import ApiClient, ApiError
from stronghouse.cli.render import paint
from stronghouse.core.agent_builder import (
    EXACT_WORK_QUESTION,
    LONG_TASKS_QUESTION,
    SKILL_QUESTIONS,
    creativity_for,
    creativity_for_temperature,
    diligence_for,
    diligence_for_max_steps,
    parse_skill_picks,
    parse_yes_no,
    skills_from_answers,
)
from stronghouse.core.agents import (
    RECOMMENDED_MAX_SKILLS,
    max_steps_for_diligence,
    temperature_for_creativity,
)
from stronghouse.core.skill_catalog import SKILLS

POLL_SECONDS = 1.0
DEFAULT_CREATIVITY, DEFAULT_DILIGENCE = 3, 5  # the app's defaults for a new agent
CARD_SUFFIX = ".agent.json"


class Cancelled(Exception):
    """The person stopped the walk-through (Ctrl+D)."""


# ---------- list and look ----------


def agents_ls(api: ApiClient, _args: argparse.Namespace) -> None:
    agents = api.get("/agents")
    if not agents:
        print("No agents yet. Build one with: shai agentvault new")
        return
    rows = [
        [
            str(a["id"]),
            paint(a["name"], "bold"),
            str(a["tasks_done"]),
            ", ".join(a["skills"]) or paint("no skills", "dim"),
        ]
        for a in agents
    ]
    print(render.table(["id", "agent", "tasks", "skills"], rows))


def skills_ls(_api: ApiClient, _args: argparse.Namespace) -> None:
    """Every skill an agent can take, numbered as in `new` and `edit`."""
    print(_numbered_skills(chosen=[]))
    print(paint("Every skill marked 'asks' waits for your y/n each time it is used.", "dim"))


def show(api: ApiClient, args: argparse.Namespace) -> None:
    agent = find_agent(api, args.agent)
    notes = api.get(f"/agents/{agent['id']}/journal")
    print(_sheet(agent))
    print(f"  tasks done   {agent['tasks_done']} · notes {len(notes)}")
    for note in notes[-5:]:
        print(paint(f"    - {note['note']}", "dim"))


# ---------- build ----------


def new(api: ApiClient, args: argparse.Namespace) -> None:
    """Build an agent. Anything not given as a flag is asked for, with yes/no questions."""
    try:
        body = _new_body(api, args)
    except Cancelled:
        print(paint("\nCancelled. Nothing was saved.", "yellow"))
        return
    if body is None:
        return
    saved = api.post("/agents", body)
    print(paint(f"Saved {saved['name']} (id {saved['id']}).", "green"))
    print(f'Give it a task with: shai agentvault run {_quoted(saved["name"])} "TASK"')


def _new_body(api: ApiClient, args: argparse.Namespace) -> dict | None:
    existing = api.get("/agents")
    walk_through = args.name is None or args.skills is None
    if walk_through:
        print(paint("New agent, built from scratch. For y/n, Enter picks the capital.", "bold"))
    name = args.name or _ask_text("Name: ", required=True)
    _check_name_is_free(name, existing)
    job = args.job if args.job is not None else ""
    if args.job is None and walk_through:
        job = _ask_text("Job and instructions (what it does, how, where to save): ")
    skills = _skills_from(args.skills) if args.skills is not None else _ask_skills()
    creativity = _creativity(args, ask=walk_through)
    diligence = _diligence(args, ask=walk_through)
    body = {
        "name": name.strip(),
        "personality": job.strip(),
        "skills": skills,
        "creativity": creativity,
        "diligence": diligence,
    }
    if not walk_through:
        return body
    print()
    print(_sheet(body))
    if not _ask_yes_no(f"Save {body['name']}?", default=True):
        print("Not saved.")
        return None
    return body


def _ask_skills() -> list[str]:
    answers = [_ask_yes_no(question.text, default=False) for question in SKILL_QUESTIONS]
    skills = skills_from_answers(answers)
    print(f"Skills from your answers: {', '.join(skills) or 'none'}")
    if not _ask_yes_no("Keep exactly these?", default=True):
        skills = _ask_skill_picks(skills)
    _advise_on_skill_count(skills)
    return skills


def _ask_skill_picks(current: list[str]) -> list[str]:
    print(_numbered_skills(chosen=current))
    while True:
        text = _ask_text("Skills (numbers or names, e.g. 1,4,9; Enter keeps the marked ones; "
                         "- for none): ")  # fmt: skip
        if not text:
            return current
        if text == "-":
            return []
        try:
            return parse_skill_picks(text, list(SKILLS))
        except ValueError as error:
            print(paint(str(error), "red"))


def _creativity(args: argparse.Namespace, ask: bool) -> int:
    if args.temperature is not None:
        return _checked(creativity_for_temperature, args.temperature)
    if not ask:
        return DEFAULT_CREATIVITY
    return creativity_for(_ask_yes_no(EXACT_WORK_QUESTION, default=True))


def _diligence(args: argparse.Namespace, ask: bool) -> int:
    if args.max_steps is not None:
        return _checked(diligence_for_max_steps, args.max_steps)
    if not ask:
        return DEFAULT_DILIGENCE
    return diligence_for(_ask_yes_no(LONG_TASKS_QUESTION, default=False))


# ---------- change ----------


def edit(api: ApiClient, args: argparse.Namespace) -> None:
    """Change an agent with flags, or walk through every field when no flag is given."""
    agent = find_agent(api, args.agent)
    flags = (args.name, args.job, args.skills, args.temperature, args.max_steps)
    try:
        body = (
            _edited_by_flags(agent, args)
            if any(f is not None for f in flags)
            else (_edited_by_questions(agent))
        )
    except Cancelled:
        print(paint("\nCancelled. Nothing was changed.", "yellow"))
        return
    if body["name"].lower() != agent["name"].lower():
        _check_name_is_free(body["name"], api.get("/agents"))
    _save_changes(api, agent, body)


def _edited_by_flags(agent: dict, args: argparse.Namespace) -> dict:
    body = _body_of(agent)
    if args.name is not None:
        body["name"] = args.name.strip()
    if args.job is not None:
        body["personality"] = args.job.strip()
    if args.skills is not None:
        body["skills"] = _skills_from(args.skills)
    if args.temperature is not None:
        body["creativity"] = _checked(creativity_for_temperature, args.temperature)
    if args.max_steps is not None:
        body["diligence"] = _checked(diligence_for_max_steps, args.max_steps)
    return body


def _edited_by_questions(agent: dict) -> dict:
    body = _body_of(agent)
    print(paint(f"Editing {agent['name']}. Enter keeps the value in brackets.", "bold"))
    body["name"] = _ask_text(f"Name [{agent['name']}]: ") or body["name"]
    job = _ask_text(f"Job and instructions [{_short(agent['personality']) or 'none'}]: ")
    body["personality"] = job or body["personality"]
    body["skills"] = _ask_skill_picks(body["skills"])
    _advise_on_skill_count(body["skills"])
    body["creativity"] = _ask_number(
        f"Temperature 0.1-1.0 [{temperature_for_creativity(body['creativity'])}]: ",
        creativity_for_temperature, body["creativity"], float,
    )  # fmt: skip
    body["diligence"] = _ask_number(
        f"Max steps per task 4-22 [{max_steps_for_diligence(body['diligence'])}]: ",
        diligence_for_max_steps, body["diligence"], int,
    )  # fmt: skip
    return body


def add_skill(api: ApiClient, args: argparse.Namespace) -> None:
    agent = find_agent(api, args.agent)
    added = _skills_from(" ".join(args.skills))
    body = _body_of(agent)
    body["skills"] = body["skills"] + [s for s in added if s not in body["skills"]]
    _advise_on_skill_count(body["skills"])
    _save_changes(api, agent, body)


def remove_skill(api: ApiClient, args: argparse.Namespace) -> None:
    agent = find_agent(api, args.agent)
    removed = _skills_from(" ".join(args.skills))
    missing = [s for s in removed if s not in agent["skills"]]
    if missing:
        raise ApiError(f"{agent['name']} does not have: {', '.join(missing)}")
    body = _body_of(agent)
    body["skills"] = [s for s in body["skills"] if s not in removed]
    _save_changes(api, agent, body)


def rm(api: ApiClient, args: argparse.Namespace) -> None:
    """Delete an agent and its notes, after a y/n unless --yes. Model files are never touched."""
    agent = find_agent(api, args.agent)
    if not args.yes:
        question = f"Delete {agent['name']} and its notes? This cannot be undone."
        try:
            if not _ask_yes_no(question, default=False):
                print("Kept.")
                return
        except Cancelled:
            print("\nKept.")
            return
    api.delete(f"/agents/{agent['id']}")
    print(f"Deleted {agent['name']}.")


def _save_changes(api: ApiClient, agent: dict, body: dict) -> None:
    saved = api.put(f"/agents/{agent['id']}", body)
    print(_sheet(saved))
    print(paint(f"Saved {saved['name']}.", "green"))


# ---------- share ----------


def export(api: ApiClient, args: argparse.Namespace) -> None:
    """Write the agent's card to a .agent.json file (never overwrites without --force)."""
    agent = find_agent(api, args.agent)
    path = Path(args.file).expanduser() if args.file else Path(card_filename(agent["name"]))
    if path.exists() and not args.force:
        raise ApiError(f"{path} already exists. Pick another file or add --force.")
    card = api.get(f"/agents/{agent['id']}/card")
    path.write_text(json.dumps(card, indent=2, ensure_ascii=False) + "\n")
    print(f"Saved {agent['name']} to {path}")


def import_card(api: ApiClient, args: argparse.Namespace) -> None:
    path = Path(args.file).expanduser()
    try:
        card = json.loads(path.read_text())
    except FileNotFoundError as error:
        raise ApiError(f"No file at {path}") from error
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ApiError(f"{path} is not an agent card (not JSON).") from error
    if not isinstance(card, dict):
        raise ApiError(f"{path} is not an agent card.")
    if args.name:
        card["name"] = args.name
    _check_name_is_free(str(card.get("name", "")), api.get("/agents"), import_hint=True)
    saved = api.post("/agents/import", card)
    print(paint(f"Imported {saved['name']} (id {saved['id']}).", "green"))
    dropped = [s for s in card.get("skills", []) if s not in saved["skills"]]
    if dropped:
        print(paint(f"Skills this version does not have were left out: {', '.join(dropped)}",
                    "yellow"))  # fmt: skip


def card_filename(name: str) -> str:
    """A safe file name for an agent card, e.g. "News Desk" -> news-desk.agent.json."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "agent"
    return slug + CARD_SUFFIX


# ---------- run ----------


def run_task(api: ApiClient, args: argparse.Namespace) -> None:
    """Run a task and ask y/n in the terminal for every action, like the app does."""
    worker = find_agent(api, args.agent)
    team = [worker["id"]]
    if args.reviewer:
        team.append(find_agent(api, args.reviewer)["id"])
    current = api.post("/quests", {"task": " ".join(args.task), "agent_ids": team})
    print(paint(f"Task #{current['id']} started with {worker['name']}. Ctrl+C cancels.", "dim"))
    printed = 0
    answered = None  # the request already decided, until the task moves on from it
    try:
        while True:
            for step in current["steps"][printed:]:
                print(f"{paint(step['agent_name'], 'bold')} {_step_text(step)}")
            printed = len(current["steps"])
            request = (printed, current["pending"])
            if current["pending"] and request != answered:
                answered = request
                current = _decide(api, current)
                continue
            if current["state"] not in ("running", "waiting_approval"):
                break
            time.sleep(POLL_SECONDS)
            current = api.get(f"/quests/{current['id']}")
    except KeyboardInterrupt:
        api.post(f"/quests/{current['id']}/cancel")
        print(paint("\nTask cancelled.", "yellow"))
        return
    if current["state"] == "done" and current["result"]:
        print("\n" + paint("Result", "green", "bold") + "\n" + current["result"])
    else:
        print(paint(f"\nTask {current['state']}.", "yellow"))


def _decide(api: ApiClient, current: dict) -> dict:
    pending = current["pending"]
    print(paint(f"\n{pending['agent_name']} wants to use {pending['skill']}:", "yellow", "bold"))
    for key, value in pending["arguments"].items():
        shown = str(value) if len(str(value)) < 400 else str(value)[:400] + "…"
        print(f"  {key}: {shown}")
    answer = prompting.prompt_user(paint("Allow once? [y/N] ", "yellow")).strip().lower()
    approve = answer in ("y", "yes")
    print(paint("Allowed." if approve else "Denied.", "green" if approve else "red"))
    return api.post(f"/quests/{current['id']}/decision", {"approve": approve})


def _step_text(step: dict) -> str:
    colour = {"tool_call": "cyan", "error": "red", "review": "green"}.get(step["kind"])
    return paint(step["text"], colour) if colour else step["text"]


# ---------- lookup ----------


def find_agent(api: ApiClient, query: str) -> dict:
    """An agent by id or by name (any case)."""
    agents = api.get("/agents")
    by_id = [a for a in agents if str(a["id"]) == query]
    match = by_id or [a for a in agents if a["name"].lower() == query.strip().lower()]
    if not match:
        raise ApiError(f'No agent called "{query}". List them with: shai agentvault ls')
    if len(match) > 1:
        ids = ", ".join(str(a["id"]) for a in match)
        raise ApiError(f'Several agents are called "{query}". Use the id ({ids}).')
    return match[0]


def _check_name_is_free(name: str, agents: list[dict], import_hint: bool = False) -> None:
    if any(a["name"].lower() == name.strip().lower() for a in agents):
        hint = " Give it another with --name." if import_hint else ""
        raise ApiError(f'You already have an agent called "{name.strip()}".{hint}')


# ---------- terminal helpers ----------


def _ask_text(prompt: str, required: bool = False) -> str:
    while True:
        try:
            text = prompting.prompt_user(prompt).strip()
        except EOFError as error:
            raise Cancelled from error
        if text or not required:
            return text
        print(paint("This one is needed.", "red"))


def _ask_yes_no(question: str, default: bool) -> bool:
    choices = "[Y/n]" if default else "[y/N]"
    while True:
        answer = parse_yes_no(_ask_text(f"{question} {choices} "), default)
        if answer is not None:
            return answer
        print(paint("Please answer y or n.", "red"))


def _ask_number(prompt: str, convert, current: int, kind: type) -> int:
    while True:
        text = _ask_text(prompt)
        if not text:
            return current
        try:
            return convert(kind(text))
        except ValueError as error:
            print(paint(str(error) if "must" in str(error) else "That is not a number.", "red"))


def _skills_from(text: str) -> list[str]:
    try:
        return parse_skill_picks(text, list(SKILLS))
    except ValueError as error:
        raise ApiError(str(error)) from error


def _checked(convert, value) -> int:
    """Run a core.agent_builder rule, turning its ValueError into a terminal error."""
    try:
        return convert(value)
    except ValueError as error:
        raise ApiError(str(error)) from error


def _advise_on_skill_count(skills: list[str]) -> None:
    if len(skills) > RECOMMENDED_MAX_SKILLS:
        print(paint(f"Note: small models can get confused with more than "
                    f"{RECOMMENDED_MAX_SKILLS} skills. If it picks the wrong tool, give it "
                    "fewer.", "yellow"))  # fmt: skip


def _numbered_skills(chosen: list[str]) -> str:
    rows = [
        [
            f"{number}{'*' if name in chosen else ''}",
            paint(name, "bold") if name in chosen else name,
            spec.category.value,
            "asks" if spec.needs_approval else "",
            spec.description,
        ]
        for number, (name, spec) in enumerate(SKILLS.items(), start=1)
    ]
    table = render.table(["#", "skill", "kind", "", "what it does"], rows)
    return table + ("\n" + paint("* = has it now", "dim") if chosen else "")


def _sheet(agent: dict) -> str:
    skills = ", ".join(agent["skills"]) or "none"
    lines = [
        paint(agent["name"], "bold", "brass") + (f"  (id {agent['id']})" if "id" in agent else ""),
        f"  job          {agent['personality'] or paint('none given', 'dim')}",
        f"  skills       {skills}",
        f"  temperature  {temperature_for_creativity(agent['creativity'])}",
        f"  max steps    {max_steps_for_diligence(agent['diligence'])} per task",
    ]
    return "\n".join(lines)


def _body_of(agent: dict) -> dict:
    keys = ("name", "portrait", "personality", "skills", "creativity", "diligence")
    return {key: agent[key] for key in keys}


def _short(text: str, limit: int = 50) -> str:
    return text if len(text) <= limit else text[:limit] + "…"


def _quoted(name: str) -> str:
    return f'"{name}"' if " " in name else name


# ---------- argument parsing ----------


def add_agent_commands(commands: argparse._SubParsersAction) -> None:
    """`shai agentvault ...`, plus the short forms `shai agents` and `shai quest`."""
    vault = commands.add_parser(
        "agentvault", help="your agents: build, change, share and run them"
    ).add_subparsers(dest="action", required=True)
    vault.add_parser("ls", help="list your agents").set_defaults(handler=agents_ls)
    vault.add_parser("skills", help="every skill an agent can take").set_defaults(handler=skills_ls)
    _add_new(vault)
    _add_edit(vault)
    one = vault.add_parser("show", help="an agent's job, skills, settings and notes")
    one.add_argument("agent", help="agent name or id")
    one.set_defaults(handler=show)
    for name, handler, text in (
        ("add-skill", add_skill, "give an agent one or more skills"),
        ("remove-skill", remove_skill, "take skills away from an agent"),
    ):
        sub = vault.add_parser(name, help=text)
        sub.add_argument("agent", help="agent name or id")
        sub.add_argument(
            "skills", nargs="+", help="skill names or numbers (see: agentvault skills)"
        )
        sub.set_defaults(handler=handler)
    delete = vault.add_parser("rm", help="delete an agent and its notes (asks first)")
    delete.add_argument("agent", help="agent name or id")
    delete.add_argument("--yes", action="store_true", help="do not ask")
    delete.set_defaults(handler=rm)
    _add_share(vault)
    _add_run(vault.add_parser("run", help="give an agent a task (asks y/n before every action)"))

    commands.add_parser("agents", help="same as: agentvault ls").set_defaults(handler=agents_ls)
    _add_run(commands.add_parser("quest", help="same as: agentvault run"))


def _add_new(vault: argparse._SubParsersAction) -> None:
    sub = vault.add_parser(
        "new", help="build an agent: answer a few yes/no questions, or give everything as flags"
    )
    sub.add_argument("name", nargs="?", help="the agent's name (asked if left out)")
    _add_sheet_flags(sub)
    sub.set_defaults(handler=new)


def _add_edit(vault: argparse._SubParsersAction) -> None:
    sub = vault.add_parser("edit", help="change an agent (walks through it if no flag is given)")
    sub.add_argument("agent", help="agent name or id")
    sub.add_argument("--name", help="a new name")
    _add_sheet_flags(sub)
    sub.set_defaults(handler=edit)


def _add_sheet_flags(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--job", help="job and instructions, in a sentence or two")
    sub.add_argument(
        "--skills", help='comma separated names or numbers, e.g. "read_file,web_search" ("" = none)'
    )
    sub.add_argument("--temperature", type=float, help="0.1 (exact) to 1.0 (inventive)")
    sub.add_argument("--max-steps", type=int, help="model replies per task, 4 to 22")


def _add_share(vault: argparse._SubParsersAction) -> None:
    out = vault.add_parser("export", help="save an agent to a .agent.json file to share or back up")
    out.add_argument("agent", help="agent name or id")
    out.add_argument("file", nargs="?", help="where to save (default: NAME.agent.json here)")
    out.add_argument("--force", action="store_true", help="overwrite the file if it exists")
    out.set_defaults(handler=export)
    into = vault.add_parser("import", help="add an agent from a .agent.json file")
    into.add_argument("file")
    into.add_argument("--name", help="give it another name (if you already have one called that)")
    into.set_defaults(handler=import_card)


def _add_run(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("agent", help="agent name or id")
    sub.add_argument("task", nargs="+")
    sub.add_argument("--reviewer", help="a second agent who checks the work (a team of two)")
    sub.set_defaults(handler=run_task)
