"""Turn a LaunchPlan into llama-server command-line arguments.

Kept separate from the planner so a change in llama.cpp's flag names touches
only this file.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from stronghouse.core.models import KvCacheType, LaunchPlan, MemoryMode

# The app owns these: changing them would load another model or move the engine
# somewhere the app cannot reach.
RESERVED_FLAGS = frozenset({"-m", "--model", "--host", "--port", "-hf", "--hf-repo", "-mu",
                            "--model-url"})  # fmt: skip
# These load the whole file into RAM, which defeats streaming a model bigger than RAM.
NO_STREAMING_FLAGS = frozenset({"--mlock", "--no-mmap"})


class ExtraArgsError(ValueError):
    """The extra flags cannot be read (for example an unclosed quote)."""


def parse_extra_args(text: str) -> tuple[str, ...]:
    """Split extra llama-server flags typed by the person, e.g. "-b 4096 -ub 1024".

    A whole pasted terminal command works too: everything up to the first flag
    (the executable) is dropped, and so are the flags the app sets itself
    (--model, --host, --port) with their values.

    Raises:
        ExtraArgsError: for unbalanced quotes.
    """
    try:
        words = shlex.split(text)
    except ValueError as error:
        raise ExtraArgsError(f"Could not read the extra flags: {error}") from error
    while words and not words[0].startswith("-"):
        words.pop(0)
    return tuple(_without_reserved(words))


def streaming_conflicts(plan: LaunchPlan) -> list[str]:
    """Extra flags that would stop a too-big-for-RAM plan from streaming from disk."""
    if plan.memory_mode is not MemoryMode.STREAM_FROM_DISK:
        return []
    return [arg for arg in plan.tuning.extra_args if arg in NO_STREAMING_FLAGS]


def build_server_args(
    executable: str, model_path: Path, plan: LaunchPlan, host: str, port: int
) -> list[str]:
    """Build the full argv for llama-server, executable first."""
    args = [
        executable,
        "--model", str(model_path),
        "--host", host,
        "--port", str(port),
        "--ctx-size", str(plan.context_size),
        "--n-gpu-layers", str(plan.n_gpu_layers),
        # Use the chat template stored in the model file.
        "--jinja",
    ]  # fmt: skip
    if plan.n_cpu_moe > 0:
        args += ["--n-cpu-moe", str(plan.n_cpu_moe)]
    if plan.kv_cache_type is not KvCacheType.F16:
        # A quantized V cache requires flash attention in llama.cpp.
        args += [
            "--cache-type-k", plan.kv_cache_type.value,
            "--cache-type-v", plan.kv_cache_type.value,
            "--flash-attn", "on",
        ]  # fmt: skip
    tuning = plan.tuning
    for flag, value in (
        ("--threads", tuning.threads),
        ("--batch-size", tuning.batch_size),
        ("--ubatch-size", tuning.ubatch_size),
    ):
        if value is not None:
            args += [flag, str(value)]
    # Last, so the person's own flags win over the app's (llama.cpp keeps the last value).
    return args + list(tuning.extra_args)


def _without_reserved(words: list[str]) -> list[str]:
    kept: list[str] = []
    skip_value = False
    for word in words:
        if skip_value:
            skip_value = False
            if not word.startswith("-"):
                continue
        if word.split("=", 1)[0] in RESERVED_FLAGS:
            skip_value = "=" not in word
            continue
        kept.append(word)
    return kept
