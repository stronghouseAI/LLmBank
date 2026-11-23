"""Turn a LaunchPlan into llama-server command-line arguments.

Kept separate from the planner so a change in llama.cpp's flag names touches
only this file.
"""

from __future__ import annotations

from pathlib import Path

from stronghouse.core.models import KvCacheType, LaunchPlan


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
    return args
