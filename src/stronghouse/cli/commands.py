"""`shai`: every StronghouseAI feature as a terminal command.

    shai vault ls [search]          shai discover qwen3 --max-gb 8 --moe
    shai vault load qwen3-8b        shai files Qwen/Qwen3-8B-GGUF
    shai chat                       shai get Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf
    shai quest Newsie "..."         shai status
    shai job add "..." --model glm  shai job ls / show / cancel / retry

Each command is a small function that calls the app's HTTP API and prints the
answer. See docs/adr/0009-terminal-client.md.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from stronghouse import __version__
from stronghouse.cli import render
from stronghouse.cli.client import ApiClient, ApiError, connect, stop_background_server
from stronghouse.cli.reference import HELP_WORDS, reference_text
from stronghouse.cli.render import format_bytes, paint
from stronghouse.config import Settings, load_settings
from stronghouse.core.log_view import LOG_SOURCES, find_source, report_text, shorten_home
from stronghouse.services.log_files import read_all_logs, read_log

GIB = 1024**3
POLL_SECONDS = 1.0
LOAD_TIMEOUT_SECONDS = 600
FILES_CHECKED_FOR_MAX_GB = 10  # --max-gb looks inside this many search results
KINDS = {
    "chat": "text-generation",
    "vision": "image-text-to-text",
    "embedding": "feature-extraction",
}
SORTS = {
    "downloads": "downloads",
    "likes": "likes",
    "trending": "trendingScore",
    "updated": "lastModified",
}
FIT_TEXT = {
    "gpu": ("fits in GPU", "green"),
    "offload": ("GPU + RAM", "yellow"),
    "too_big": ("too big", "red"),
}


def prompt_user(prompt: str) -> str:
    """Read one line from the person at the terminal (replaced in tests)."""
    return input(prompt)


# ---------- vault ----------


def vault_ls(api: ApiClient, args: argparse.Namespace) -> None:
    entries = api.get("/library", search=args.search or "")
    if not entries:
        print("Your vault is empty. Find a model with: shai discover <name>")
        return
    rows = [
        [
            str(e["id"]),
            paint(e["filename"], "bold"),
            format_bytes(e["size_bytes"]),
            e["architecture"] or "?",
            paint("MoE", "brass") if e["is_moe"] else "",
            e["repo_id"],
        ]
        for e in entries
    ]
    print(render.table(["id", "file", "size", "arch", "moe", "repo"], rows))


def vault_rescan(api: ApiClient, _args: argparse.Namespace) -> None:
    result = api.post("/library/rescan")
    print(f"Indexed {result['indexed']} model(s).")


def vault_plan(api: ApiClient, args: argparse.Namespace) -> None:
    entry = _find_model(api, args.model)
    result = api.post(f"/library/{entry['id']}/plan", _plan_body(args))
    _print_plan(entry, result["plan"], result["command"])
    _print_forecast(result["forecast"])


def vault_load(api: ApiClient, args: argparse.Namespace) -> None:
    entry = _find_model(api, args.model)
    # The server unloads the current model before planning, so print the plan it used.
    started = api.post("/engine/start", {**_plan_body(args), "model_id": entry["id"]})
    _print_plan(entry, started["plan"], started["command"])
    _print_forecast(started["forecast"])
    print(paint("Loading", "yellow") + " (big models take a minute)", end="", flush=True)
    deadline = time.monotonic() + LOAD_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        status = api.get("/engine")
        if status["state"] == "ready":
            print("\n" + paint("Ready.", "green") + " Talk to it with: shai chat")
            return
        if status["state"] in ("crashed", "stopped"):
            print("\n" + paint("The engine stopped while loading. Last log lines:", "red"))
            print("\n".join(status["log_tail"][-10:]))
            raise SystemExit(1)
        print(".", end="", flush=True)
        time.sleep(POLL_SECONDS)
    raise ApiError("Still loading after 10 minutes. Check: shai status")


def vault_unload(api: ApiClient, _args: argparse.Namespace) -> None:
    api.post("/engine/stop")
    print("Model unloaded. GPU memory is free.")


def bench(api: ApiClient, _args: argparse.Namespace) -> None:
    print("Running speed test…")
    result = api.post("/engine/benchmark")
    print(f"Generation: {paint(str(result['generated_tokens_per_second']), 'bold')} tokens/s")
    print(f"Prompt:     {result['prompt_tokens_per_second']} tokens/s")


# ---------- discover ----------


def discover(api: ApiClient, args: argparse.Namespace) -> None:
    params = {
        "q": " ".join(args.query),
        "size": args.size,
        "kind": KINDS.get(args.type, "any"),
        "sort": SORTS[args.sort],
    }
    if args.moe:
        params["moe"] = "true"
    models = api.get("/hub/search", **params)
    if args.max_gb:
        models = _with_best_file(api, models[:FILES_CHECKED_FOR_MAX_GB], args.max_gb)
    if not models:
        print("No GGUF models match. Try fewer filters.")
        return
    rows = []
    for m in models:
        row = [
            paint(m["repo_id"], "bold"),
            m["size_label"] or "?",
            paint("MoE", "brass") if m["is_moe"] else "",
            f"{m['downloads']:,}",
        ]
        if args.max_gb:
            best = m["best_file"]
            row.append(f"{best['filename']} ({format_bytes(best['size_bytes'])})")
        rows.append(row)
    headers = ["repo", "params", "moe", "downloads"] + (
        ["best file under limit"] if args.max_gb else []
    )
    print(render.table(headers, rows))
    print(paint("\nFiles in a repo: shai files <repo>    Download: shai get <repo> <file>", "dim"))


def files(api: ApiClient, args: argparse.Namespace) -> None:
    listed = api.get("/hub/files", repo_id=args.repo)
    limit = args.max_gb * GIB if args.max_gb else None
    shown = [
        f
        for f in listed
        if (args.all or f["fit"] != "too_big") and (not limit or f["size_bytes"] <= limit)
    ]
    rows = []
    for f in shown:
        label, colour = FIT_TEXT[f["fit"]]
        rows.append([f["filename"], format_bytes(f["size_bytes"]), paint(label, colour)])
    print(render.table(["file", "size", "on this computer"], rows) or "No matching .gguf files.")
    hidden = len(listed) - len(shown)
    if hidden:
        print(
            paint(
                f"\n{hidden} file(s) hidden (too big or over --max-gb). Show all with --all.", "dim"
            )
        )


def get(api: ApiClient, args: argparse.Namespace) -> None:
    job = api.post("/downloads", {"repo_id": args.repo, "filename": args.filename})
    print(f"Archiving {args.filename}")
    while True:
        job = next((j for j in api.get("/downloads") if j["id"] == job["id"]), job)
        if job["state"] == "done":
            print(
                "\r" + render.progress_bar(1, 1) + "  " + paint("archived in your vault", "green")
            )
            return
        if job["state"] == "failed":
            print("\n" + paint(f"Download failed: {job['error']}", "red"))
            raise SystemExit(1)
        done, total = job["bytes_done"], job["bytes_total"]
        sizes = f"{format_bytes(done)} of {format_bytes(total)}"
        text = f"{render.progress_bar(done, total)}  {sizes}  {job['state']}"
        print("\r" + text, end="", flush=True)
        time.sleep(POLL_SECONDS)


# ---------- chat ----------


def ask(api: ApiClient, args: argparse.Namespace) -> None:
    messages = [{"role": "user", "content": " ".join(args.prompt)}]
    _stream_reply(api, messages, think=not args.no_think, show_thinking=args.show_thinking)


def chat(api: ApiClient, args: argparse.Namespace) -> None:
    print(paint("Chat with the loaded model. /clear starts over, /exit or Ctrl+D quits.", "dim"))
    messages: list[dict] = []
    first = " ".join(args.prompt).strip()  # `shai chat how are you` opens with that message
    while True:
        try:
            if first:
                text, first = first, ""
                print(paint("you> ", "brass") + text)
            else:
                text = prompt_user(paint("you> ", "brass")).strip()
        except EOFError:
            print()
            return
        if text in ("/exit", "/quit"):
            return
        if text == "/clear":
            messages.clear()
            print(paint("Started a new conversation.", "dim"))
            continue
        if not text:
            continue
        messages.append({"role": "user", "content": text})
        reply = _stream_reply(
            api, messages, think=not args.no_think, show_thinking=args.show_thinking
        )
        messages.append({"role": "assistant", "content": reply})


def _stream_reply(api: ApiClient, messages: list[dict], think: bool, show_thinking: bool) -> str:
    """Print the reply as it arrives. Thinking shows as a counter, or dim text if asked."""
    body: dict = {"messages": messages}
    if not think:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    started, thought_words, answer = time.monotonic(), 0, ""
    try:
        for delta in api.stream_chat(body):
            if delta.get("reasoning_content"):
                thought_words += len(delta["reasoning_content"].split())
                if show_thinking:
                    print(paint(delta["reasoning_content"], "dim"), end="", flush=True)
                else:
                    seconds = int(time.monotonic() - started)
                    print(
                        f"\r{paint(f'thinking… {seconds}s · {thought_words} words', 'dim')}",
                        end="",
                        flush=True,
                    )
            if delta.get("content"):
                if not answer and thought_words:
                    seconds = int(time.monotonic() - started)
                    print(f"\r{paint(f'thought for {seconds}s ({thought_words} words)', 'dim')}\n")
                answer += delta["content"]
                print(delta["content"], end="", flush=True)
    except KeyboardInterrupt:
        print(paint("\n(stopped)", "dim"))
    print()
    return answer


# ---------- guild ----------


def agents_ls(api: ApiClient, _args: argparse.Namespace) -> None:
    agents = api.get("/agents")
    if not agents:
        print("No agents yet. Create one in the Guild tab of the app.")
        return
    rows = [
        [
            str(a["id"]),
            f"{a['portrait'] or '👤'} {a['name']}",
            str(a["level"]),
            ", ".join(a["skills"]),
        ]
        for a in agents
    ]
    print(render.table(["id", "agent", "level", "skills"], rows))


def quest(api: ApiClient, args: argparse.Namespace) -> None:
    """Run a quest and ask y/n in the terminal for every action, like the app does."""
    worker = _find_agent(api, args.agent)
    party = [worker["id"]]
    if args.reviewer:
        party.append(_find_agent(api, args.reviewer)["id"])
    current = api.post("/quests", {"task": " ".join(args.task), "agent_ids": party})
    print(paint(f"Quest #{current['id']} started with {worker['name']}. Ctrl+C cancels.", "dim"))
    printed = 0
    answered = None  # the request already decided, until the quest moves on from it
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
        print(paint("\nQuest cancelled.", "yellow"))
        return
    if current["state"] == "done" and current["result"]:
        print("\n" + paint("Result", "green", "bold") + "\n" + current["result"])
    else:
        print(paint(f"\nQuest {current['state']}.", "yellow"))


def _decide(api: ApiClient, current: dict) -> dict:
    pending = current["pending"]
    print(paint(f"\n{pending['agent_name']} wants to use {pending['skill']}:", "yellow", "bold"))
    for key, value in pending["arguments"].items():
        shown = str(value) if len(str(value)) < 400 else str(value)[:400] + "…"
        print(f"  {key}: {shown}")
    answer = prompt_user(paint("Allow once? [y/N] ", "yellow")).strip().lower()
    approve = answer in ("y", "yes")
    print(paint("Allowed." if approve else "Denied.", "green" if approve else "red"))
    return api.post(f"/quests/{current['id']}/decision", {"approve": approve})


def _step_text(step: dict) -> str:
    colour = {"tool_call": "cyan", "error": "red", "review": "green"}.get(step["kind"])
    return paint(step["text"], colour) if colour else step["text"]


# ---------- background jobs ----------

JOB_COLOURS = {"done": "green", "running": "yellow", "loading": "yellow", "failed": "red"}


def job_add(api: ApiClient, args: argparse.Namespace) -> None:
    prompt = " ".join(args.prompt)
    if args.file:
        prompt = (prompt + "\n\n" if prompt else "") + _read_prompt_file(args.file)
    body = {**_plan_body(args), "prompt": prompt, "think": not args.no_think}
    if args.model:
        body["model_id"] = _find_model(api, args.model)["id"]
    if args.max_tokens:
        body["max_tokens"] = args.max_tokens
    job = api.post("/jobs", body)
    print(f"Queued job {paint(str(job['id']), 'bold')}: {job['title']}")
    print(
        "It runs in the background; you get a desktop notification when it is done.\n"
        f"Check on it with: shai job show {job['id']}"
    )


def job_ls(api: ApiClient, _args: argparse.Namespace) -> None:
    jobs = api.get("/jobs")
    if not jobs:
        print('No jobs yet. Queue one with: shai job add "your prompt" --model NAME')
        return
    rows = [
        [
            str(j["id"]),
            paint(j["state"], JOB_COLOURS.get(j["state"], "dim")),
            j["title"],
            str(j["generated_tokens"] or ""),
            f"{j['tokens_per_second']:g}" if j["tokens_per_second"] else "",
        ]
        for j in jobs
    ]
    print(render.table(["id", "state", "prompt", "tokens", "tok/s"], rows))


def job_show(api: ApiClient, args: argparse.Namespace) -> None:
    job = api.get(f"/jobs/{args.id}")
    print(f"Job {job['id']}: {paint(job['state'], JOB_COLOURS.get(job['state'], 'dim'), 'bold')}")
    if job["error"]:
        print(paint(job["error"], "red"))
    if job["output_path"]:
        print(paint(f"Answer file: {job['output_path']}", "dim"))
    print()
    print(job["output"] or "(nothing written yet)")


def job_cancel(api: ApiClient, args: argparse.Namespace) -> None:
    api.post(f"/jobs/{args.id}/cancel")
    print(f"Stopping job {args.id}.")


def job_retry(api: ApiClient, args: argparse.Namespace) -> None:
    job = api.post(f"/jobs/{args.id}/retry")
    print(f"Queued again as job {job['id']}.")


def _read_prompt_file(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except OSError as error:
        raise ApiError(f"Could not read {path}: {error.strerror}") from error


# ---------- system ----------


def status(api: ApiClient, _args: argparse.Namespace) -> None:
    engine = api.get("/engine")
    colour = {"ready": "green", "loading": "yellow", "crashed": "red"}.get(engine["state"], "dim")
    print(f"Engine: {paint(engine['state'], colour, 'bold')}")
    if engine["model_path"]:
        print(f"Model:  {engine['model_path']}")
    if engine["plan"]:
        plan = engine["plan"]
        print(
            f"Plan:   {plan['strategy']} · context {plan['context_size']}"
            f" · GPU layers {plan['n_gpu_layers']} · experts on CPU {plan['n_cpu_moe']}"
        )
    system(api, _args)


def system(api: ApiClient, _args: argparse.Namespace) -> None:
    hw = api.get("/hardware")
    gpu = (
        f"{hw['gpu_name']} · {format_bytes(hw['gpu_free_bytes'])} free"
        f" of {format_bytes(hw['gpu_total_bytes'])}"
        if hw["gpu_name"]
        else "none found (CPU only)"
    )
    print(f"GPU:    {gpu}")
    print(
        f"Memory: {format_bytes(hw['ram_available_bytes'])} free"
        f" of {format_bytes(hw['ram_total_bytes'])}"
    )
    print(f"CPU:    {hw['cpu_threads']} threads")


# ---------- helpers ----------


def _find_model(api: ApiClient, query: str) -> dict:
    """A vault model by id, or by a unique piece of its file or repo name."""
    entries = api.get("/library")
    if query.isdigit():
        match = [e for e in entries if e["id"] == int(query)]
    else:
        needle = _loose(query)
        match = [
            e for e in entries if needle in _loose(e["filename"]) or needle in _loose(e["repo_id"])
        ]
    if len(match) == 1:
        return match[0]
    if not match:
        raise ApiError(f'No model in your vault matches "{query}". List them with: shai vault ls')
    names = "\n".join(f"  {e['id']}  {e['filename']}" for e in match)
    raise ApiError(f'"{query}" matches several models. Use the id:\n{names}')


def _loose(name: str) -> str:
    """Lower case without separators, so qwen3_8b, Qwen3-8B and "qwen3 8b" all match."""
    return "".join(char for char in name.lower() if char.isalnum())


def _find_agent(api: ApiClient, query: str) -> dict:
    agents = api.get("/agents")
    match = [a for a in agents if str(a["id"]) == query or a["name"].lower() == query.lower()]
    if not match:
        raise ApiError(f'No agent called "{query}". List them with: shai agents')
    return match[0]


def _with_best_file(api: ApiClient, models: list[dict], max_gb: float) -> list[dict]:
    """Keep repos with a file under the limit, noting the largest such file (best quality)."""
    limit = max_gb * GIB
    kept = []
    for model in models:
        try:
            listed = api.get("/hub/files", repo_id=model["repo_id"])
        except ApiError:
            continue
        under = [f for f in listed if f["size_bytes"] <= limit]
        if under:
            kept.append({**model, "best_file": max(under, key=lambda f: f["size_bytes"])})
    return kept


def _plan_body(args: argparse.Namespace) -> dict:
    body: dict = {"context": args.ctx, "kv_cache_type": args.kv}
    optional = {
        "n_cpu_moe": args.n_cpu_moe,
        "n_gpu_layers": args.ngl,
        "threads": args.threads,
        "batch_size": args.batch,
        "ubatch_size": args.ubatch,
        "extra_args": args.extra,
    }
    body.update({key: value for key, value in optional.items() if value is not None})
    return body


def _print_plan(entry: dict, plan: dict, command: list[str]) -> None:
    print(f"{paint(entry['filename'], 'bold')}  ({format_bytes(entry['size_bytes'])})")
    print(f"  strategy   {paint(plan['strategy'], 'brass')}")
    print(
        f"  GPU        {format_bytes(plan['estimated_gpu_bytes'])} · {plan['n_gpu_layers']} layers"
    )
    print(
        f"  RAM        {format_bytes(plan['estimated_cpu_bytes'])}"
        f" · experts of {plan['n_cpu_moe']} blocks"
    )
    print(f"  context    {plan['context_size']} tokens · KV cache {plan['kv_cache_type']}")
    if plan["memory_mode"] == "stream_from_disk":
        print(f"  memory     {paint('streams from disk (big model mode)', 'yellow')}")
    for note in plan["notes"]:
        print(paint(f"  note: {note}", "dim"))
    print(paint("  $ " + " ".join(command), "dim"))


def _print_forecast(forecast: dict) -> None:
    print(f"  speed      {forecast['wait']} (estimate)")
    if forecast["quant_tip"]:
        print(paint(f"  {forecast['quant_tip']}", "brass"))
    options = forecast["quant_options"]
    if options:
        cells = [
            f"{o['label']}{'*' if o['is_current'] else ''} {o['tokens_per_second']:g}"
            + ("" if o["memory_mode"] == "in_memory" else " (disk)")
            for o in options
        ]
        print(paint("  tokens/s by quant: " + " · ".join(cells), "dim"))


# ---------- argument parsing ----------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="shai",
        description="StronghouseAI in the terminal. Starts the app in the background if needed.",
        epilog="Examples:  shai vault ls · shai vault load qwen3-8b"
        " · shai discover qwen3 --max-gb 8 --moe"
        ' · shai chat · shai quest Newsie "UK headlines to ~/news.md"',
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for add_section in (
        _add_vault,
        _add_discover,
        _add_chat,
        _add_guild,
        _add_job_commands,
        _add_system,
    ):
        add_section(commands)
    return parser


def _add_vault(commands: argparse._SubParsersAction) -> None:
    vault = commands.add_parser("vault", help="models on your disk").add_subparsers(
        dest="action", required=True
    )
    ls = vault.add_parser("ls", help="list models in your vault")
    ls.add_argument("search", nargs="?", help="filter by name, repo or tag")
    ls.set_defaults(handler=vault_ls)
    vault.add_parser("rescan", help="re-index the archive folder").set_defaults(
        handler=vault_rescan
    )
    for name, handler, text in (
        ("plan", vault_plan, "show how a model would be split between GPU and RAM"),
        ("load", vault_load, "load a model (Auto-fit, then start the engine)"),
    ):
        sub = vault.add_parser(name, help=text)
        sub.add_argument("model", help="vault id, or part of the file name (e.g. qwen3-8b)")
        _add_plan_flags(sub)
        sub.set_defaults(handler=handler)
    vault.add_parser("unload", help="unload the model and free GPU memory").set_defaults(
        handler=vault_unload
    )

    commands.add_parser("load", help="same as: vault load").set_defaults(handler=vault_load)
    _add_plan_flags(commands.choices["load"], with_model=True)
    commands.add_parser("unload", help="same as: vault unload").set_defaults(handler=vault_unload)


def _add_discover(commands: argparse._SubParsersAction) -> None:
    disc = commands.add_parser("discover", help="search GGUF models on Hugging Face")
    disc.add_argument("query", nargs="+", help="what to search for, e.g. qwen3 30b")
    disc.add_argument(
        "--size",
        choices=["any", "tiny", "small", "medium", "large", "huge"],
        default="any",
        help="tiny <4B, small 4-9B, medium 9-20B, large 20-40B, huge >40B",
    )
    disc.add_argument("--type", choices=sorted(KINDS), help="chat, vision or embedding")
    disc.add_argument("--moe", action="store_true", help="Mixture-of-Experts models only")
    disc.add_argument("--sort", choices=sorted(SORTS), default="downloads")
    disc.add_argument(
        "--max-gb", type=float, metavar="GB", help="only repos with a file of at most this size"
    )
    disc.set_defaults(handler=discover)

    fls = commands.add_parser("files", help="list a repo's .gguf files and whether they fit")
    fls.add_argument("repo", help="e.g. Qwen/Qwen3-8B-GGUF")
    fls.add_argument("--max-gb", type=float, metavar="GB", help="hide files larger than this")
    fls.add_argument("--all", action="store_true", help="also show files too big for this computer")
    fls.set_defaults(handler=files)

    dl = commands.add_parser("get", help="download a file into your vault")
    dl.add_argument("repo")
    dl.add_argument("filename")
    dl.set_defaults(handler=get)


def _add_chat(commands: argparse._SubParsersAction) -> None:
    for name, handler, text in (
        ("chat", chat, "talk to the loaded model"),
        ("ask", ask, "ask one question"),
    ):
        sub = commands.add_parser(name, help=text)
        sub.add_argument(
            "prompt", nargs="+" if name == "ask" else "*", help="your message (optional for chat)"
        )
        sub.add_argument(
            "--no-think", action="store_true", help="answer directly, skip thinking (Qwen3)"
        )
        sub.add_argument("--show-thinking", action="store_true", help="print the model's thinking")
        sub.set_defaults(handler=handler)

    commands.add_parser("bench", help="speed test the loaded model").set_defaults(handler=bench)


def _add_guild(commands: argparse._SubParsersAction) -> None:
    commands.add_parser("agents", help="list your agents").set_defaults(handler=agents_ls)
    qst = commands.add_parser("quest", help="send an agent on a quest (asks before every action)")
    qst.add_argument("agent", help="agent name or id")
    qst.add_argument("task", nargs="+")
    qst.add_argument("--reviewer", help="a second agent who checks the work")
    qst.set_defaults(handler=quest)


def _add_system(commands: argparse._SubParsersAction) -> None:
    commands.add_parser("status", help="engine, model and hardware").set_defaults(handler=status)
    commands.add_parser("system", help="GPU, memory and CPU").set_defaults(handler=system)
    commands.add_parser("stop", help="stop the background app (unloads the model)")
    logs = commands.add_parser(
        "logs", help="show the app's logs, secrets hidden (works even if the app is down)"
    )
    logs.add_argument(
        "source",
        nargs="?",
        choices=[source.key for source in LOG_SOURCES],
        help="which log (default: every log, with a few facts about this computer)",
    )
    logs.add_argument("--lines", type=int, default=200, help="last N lines (default 200)")
    logs.add_argument("--problems", action="store_true", help="only warnings and errors")
    serve = commands.add_parser("serve", help="open the web UI (starts the app if needed)")
    serve.add_argument("--no-browser", action="store_true")


def _add_job_commands(commands) -> None:
    job = commands.add_parser(
        "job", help="background jobs: queue prompts for a big model and walk away"
    ).add_subparsers(dest="action", required=True)
    add = job.add_parser("add", help="queue a prompt (runs in the background)")
    add.add_argument("prompt", nargs="*", help="the prompt (or use --file)")
    add.add_argument("--file", help="read the prompt (or extra text) from this file")
    add.add_argument("--model", help="load this vault model first (default: the loaded one)")
    add.add_argument("--no-think", action="store_true", help="answer directly, skip thinking")
    add.add_argument("--max-tokens", type=int, help="stop after this many tokens")
    _add_plan_flags(add, with_model=False)
    add.set_defaults(handler=job_add)
    job.add_parser("ls", help="list jobs, newest first").set_defaults(handler=job_ls)
    for name, handler, text in (
        ("show", job_show, "show a job's answer so far"),
        ("cancel", job_cancel, "stop a running job or remove a queued one"),
        ("retry", job_retry, "queue a finished job again"),
    ):
        sub = job.add_parser(name, help=text)
        sub.add_argument("id", type=int)
        sub.set_defaults(handler=handler)


def _add_plan_flags(sub: argparse.ArgumentParser, with_model: bool = False) -> None:
    if with_model:
        sub.add_argument("model", help="vault id, or part of the file name")
    sub.add_argument("--ctx", type=int, default=8192, help="context length (default 8192)")
    sub.add_argument("--kv", choices=["f16", "q8_0"], default="f16", help="KV cache type")
    sub.add_argument("--n-cpu-moe", type=int, help="experts of the first N blocks on CPU")
    sub.add_argument("--ngl", type=int, help="layers on the GPU (--n-gpu-layers)")
    sub.add_argument("--threads", type=int, help="CPU threads (default: llama.cpp decides)")
    sub.add_argument("--batch", type=int, help="llama-server --batch-size")
    sub.add_argument("--ubatch", type=int, help="llama-server --ubatch-size")
    sub.add_argument(
        "--extra",
        metavar="FLAGS",
        help='more llama-server flags, e.g. --extra="-b 4096 -ub 1024" (or a pasted command)',
    )


def logs_text(settings: Settings, key: str | None, lines: int, problems: bool) -> str:
    """Read the log files directly, so this works when the app will not start."""
    lines = max(1, lines)
    if key is not None:
        source = find_source(key)
        assert source is not None  # argparse only allows known keys
        return read_log(settings, source, lines, problems).copy_text
    views = read_all_logs(settings, lines, problems)
    facts = [f"Data folder: {shorten_home(str(settings.data_dir), str(Path.home()))}"]
    return report_text(__version__, facts, [view.copy_text for view in views])


def run(argv: list[str], settings: Settings, api: ApiClient | None = None) -> int:
    """Parse and run one command. Returns the process exit code."""
    if not argv or argv[0] in HELP_WORDS:
        print(reference_text(paint))
        return 0
    if argv[-1] == "-help":  # `shai vault load -help`, the single-dash habit from nmap
        argv = [*argv[:-1], "--help"]
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "serve":
        from stronghouse.__main__ import serve

        serve(no_browser=args.no_browser)
        return 0
    if args.command == "logs":
        print(logs_text(settings, args.source, args.lines, args.problems))
        return 0
    if args.command == "stop":
        print("Stopped." if stop_background_server(settings) else "No background app was running.")
        return 0
    try:
        api = api or connect(settings)
        args.handler(api, args)
    except ApiError as error:
        print(paint(f"Error: {error}", "red"), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


def main() -> None:
    raise SystemExit(run(sys.argv[1:], load_settings()))
