"""The full command reference shown by `shai --help`, `shai -help`, `shai -h` and `shai help`.

Kept in one place so the terminal and the README stay in step. Per-command details
(`shai vault load --help`) still come from argparse in commands.py.
"""

from __future__ import annotations

HELP_WORDS = frozenset({"help", "-h", "-help", "--help", "-?"})

# (section title, [(usage, what it does), ...])
SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "VAULT  models on your disk",
        [
            ("vault ls [SEARCH]", "List your models (id, file, size, architecture, MoE)"),
            ("vault rescan", "Re-index the archive folder, e.g. after copying models in"),
            ("vault plan MODEL", "Show how MODEL would be split between GPU and RAM"),
            ("load MODEL", "Unload the current model, auto-fit MODEL, wait until ready"),
            ("unload", "Unload the model and free GPU memory"),
        ],
    ),
    (
        "LOAD OPTIONS  for load and vault plan",
        [
            ("MODEL", "Vault id (shai load 3) or part of the name (qwen3-8b, qwen3_8b)"),
            ("--ctx N", "Context length in tokens (default 8192)"),
            ("--kv q8_0", "Half-size KV cache, for longer context (default f16)"),
            ("--n-cpu-moe N", "Put the experts of the first N blocks in RAM (MoE models)"),
            ("--ngl N", "Number of layers on the GPU (--n-gpu-layers)"),
            ("--threads N", "CPU threads (default: llama.cpp decides)"),
            ("--batch N / --ubatch N", "llama-server --batch-size / --ubatch-size"),
            ('--extra="FLAGS"', "Any other llama-server flags, or paste your whole command"),
        ],
    ),
    (
        "JOBS  big models in the background: queue, walk away, get notified",
        [
            ('job add "PROMPT"', "Queue a prompt for the loaded model"),
            ("  --model MODEL", "Load this model first (with Auto-fit and the load options)"),
            ("  --file PATH", "Read the prompt from a file"),
            ("  --no-think / --max-tokens N", "Skip thinking / stop after N tokens"),
            ("job ls", "All jobs: queued, running, done, failed"),
            ("job show ID", "The answer so far (also saved as a Markdown file)"),
            ("job cancel ID / job retry ID", "Stop or remove a job / run it again"),
        ],
    ),
    (
        "DISCOVER  Hugging Face",
        [
            ("discover QUERY", "Search GGUF models, most downloaded first"),
            ("  --max-gb N", "Only repos with a file of at most N GB, and show that file"),
            ("  --size S", "tiny <4B, small 4-9B, medium 9-20B, large 20-40B, huge >40B"),
            ("  --type T", "chat, vision or embedding"),
            ("  --moe", "Mixture-of-Experts models only"),
            ("  --sort S", "downloads, likes, trending or updated"),
            ("files REPO", "A repo's files: fits in GPU, GPU + RAM, or too big"),
            ("  --max-gb N / --all", "Hide files over N GB / also show files too big"),
            ("get REPO FILE", "Download FILE into your vault, with a progress bar"),
        ],
    ),
    (
        "CHAT  with the loaded model",
        [
            ("chat [MESSAGE]", "Talk back and forth. /clear starts over, /exit or Ctrl+D quits"),
            ("  /web, /web on, /web off", "Inside chat: turn internet access on or off"),
            ('ask "QUESTION"', "One question, one answer"),
            ("  --no-think", "Answer straight away, skip thinking (Qwen3)"),
            ("  --show-thinking", "Print the model's thinking as it goes"),
            ("  --web", "Let the model search the web and read pages. Asks y/n every time"),
            ("bench", "Speed test: tokens per second"),
        ],
    ),
    (
        "GUILD  agents (create them in the web app's Guild tab)",
        [
            ("agents", "List your agents and their skills"),
            ('quest AGENT "TASK"', "Send an agent on a quest. Asks y/n before every action"),
            ("  --reviewer AGENT", "A second agent checks the work"),
        ],
    ),
    (
        "SYSTEM",
        [
            ("status", "Engine state, loaded model and its plan, GPU and memory"),
            ("system", "GPU, memory and CPU"),
            ("serve", "Open the web UI, starting the app here if it is not running"),
            ("stop", "Stop the background app (also unloads the model)"),
            ("logs [LOG]", "All logs with secrets hidden, ready to paste to Claude"),
            ("  --lines N / --problems", "Last N lines (default 200) / only warnings and errors"),
            ("help", "This page. Details of one command: shai COMMAND --help"),
        ],
    ),
]

EXAMPLES = [
    ("shai discover qwen3 --max-gb 8", "find Qwen3 models that fit in 8 GB"),
    ("shai files Qwen/Qwen3-8B-GGUF", "see which of its files fit your computer"),
    ("shai get Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf", "download one"),
    ("shai load qwen3-8b", "load it"),
    ('shai ask "What is 17 times 23?" --no-think', "quick answer"),
    ('shai ask --web "What changed in the latest llama.cpp?"', "answer with a web search"),
    ("shai load deepseek --ctx 16384 --kv q8_0", "bigger model, longer context"),
    ("shai vault plan glm-4.5-air", "speed estimate and which quantization fits best"),
    ('shai job add "Write a business plan for ..." --model glm', "queue it and walk away"),
    ('shai quest Newsie "UK headlines into ~/Documents/news.md"', "run an agent"),
]


def reference_text(paint=lambda text, *_styles: text) -> str:
    """The whole reference as plain text; pass render.paint for colours."""
    width = max(len(usage) for _, rows in SECTIONS for usage, _ in rows) + 2
    lines = [
        paint("StronghouseAI terminal (shai)", "bold", "brass"),
        "Usage: shai COMMAND [OPTIONS]    The app starts in the background if needed;",
        "       the web UI (uv run stronghouse) shares the same models and agents.",
    ]
    for title, rows in SECTIONS:
        lines += ["", paint(title, "bold")]
        lines += [f"  {usage.ljust(width)}{text}" for usage, text in rows]
    lines += ["", paint("EXAMPLES", "bold")]
    lines += [f"  {command}\n      {paint(note, 'dim')}" for command, note in EXAMPLES]
    return "\n".join(lines)
