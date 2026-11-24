"""Suggest which quantization of a model fits this machine best.

The same model is published at many quantizations (Q2_K, Q4_K_M, Q8_0...). Fewer
bits per weight means a smaller file, slightly lower quality, and for big MoE
models often a *much* faster run, because it can fit in RAM instead of streaming
from disk. This module rescales a model's profile to each common quantization and
runs the normal planner and speed estimate on it. No files are read.
"""

from __future__ import annotations

import re
from dataclasses import replace

from stronghouse.core.models import (
    HardwareProfile,
    KvCacheType,
    MemoryMode,
    MemorySpeeds,
    ModelProfile,
    QuantOption,
)
from stronghouse.core.offload_planner import plan_offload
from stronghouse.core.speed_estimator import estimate_speed, format_rate

# Approximate average bits per weight, from llama.cpp's quantize table. Real files
# mix types per tensor, so these are only good for comparisons.
BITS_PER_WEIGHT = {
    "IQ1_S": 1.56, "IQ1_M": 1.75, "IQ2_XXS": 2.06, "IQ2_XS": 2.31, "IQ2_S": 2.5,
    "IQ2_M": 2.7, "Q2_K": 3.0, "IQ3_XXS": 3.06, "IQ3_XS": 3.3, "Q3_K_S": 3.5,
    "IQ3_M": 3.66, "Q3_K_M": 3.9, "IQ4_XS": 4.25, "MXFP4": 4.25, "Q3_K_L": 4.3,
    "IQ4_NL": 4.5, "Q4_0": 4.55, "Q4_K_S": 4.6, "Q4_K_M": 4.85, "Q5_K_S": 5.5,
    "Q5_K_M": 5.7, "Q6_K": 6.6, "Q8_0": 8.5, "BF16": 16.0, "F16": 16.0,
}  # fmt: skip
# The ones worth suggesting; most repos publish these.
SUGGESTED = ("IQ1_M", "IQ2_XXS", "IQ2_M", "Q2_K", "Q3_K_M", "IQ4_XS", "Q4_K_M", "Q5_K_M",
             "Q6_K", "Q8_0")  # fmt: skip
# Below this, quality drops a lot; say so rather than recommend it silently.
LOW_QUALITY_BITS = 2.5


def detect_quant(filename: str) -> str | None:
    """The quantization label in a GGUF file name, e.g. "Q4_K_M", or None."""
    name = filename.upper()
    found = [label for label in BITS_PER_WEIGHT if re.search(rf"(?<![A-Z0-9]){label}", name)]
    return max(found, key=len) if found else None


def quant_options(
    model: ModelProfile,
    filename: str,
    hardware: HardwareProfile,
    context: int,
    kv_cache_type: KvCacheType,
    speeds: MemorySpeeds,
) -> list[QuantOption]:
    """Each suggested quantization with its size, memory mode and estimated speed.

    Returns an empty list when the file's own quantization cannot be recognised.
    """
    current = detect_quant(filename)
    if current is None:
        return []
    labels = dict.fromkeys([*SUGGESTED, current])
    options = []
    for label in sorted(labels, key=BITS_PER_WEIGHT.__getitem__):
        scaled = scale_profile(model, BITS_PER_WEIGHT[label] / BITS_PER_WEIGHT[current])
        plan = plan_offload(scaled, hardware, context, kv_cache_type)
        speed = estimate_speed(scaled, plan, hardware, speeds)
        options.append(
            QuantOption(
                label=label,
                bits_per_weight=BITS_PER_WEIGHT[label],
                estimated_file_bytes=scaled.total_weight_bytes,
                memory_mode=plan.memory_mode,
                tokens_per_second=speed.tokens_per_second,
                is_current=label == current,
            )
        )
    return options


def quant_tip(options: list[QuantOption]) -> str | None:
    """One sentence pointing at a better-fitting quantization, or None if this one is right."""
    current = next((option for option in options if option.is_current), None)
    if current is None:
        return None
    in_memory = [o for o in options if o.memory_mode is MemoryMode.IN_MEMORY]
    best = max(in_memory, key=lambda o: o.bits_per_weight, default=None)
    if current.memory_mode is MemoryMode.STREAM_FROM_DISK:
        if best is None:
            return (
                "Even the smallest common quantization would not fit in RAM, so this "
                "model streams from disk whatever you pick. Queue jobs and let it work."
            )
        warning = " (noticeably lower quality)" if best.bits_per_weight < LOW_QUALITY_BITS else ""
        return (
            f"Tip: {best.label}{warning} is about {_gib(best.estimated_file_bytes)} and would "
            f"fit in RAM: about {format_rate(best.tokens_per_second)} tokens/s instead of "
            f"{format_rate(current.tokens_per_second)}."
        )
    if best is not None and best.bits_per_weight > current.bits_per_weight:
        return (
            f"Tip: you have room for {best.label} (about {_gib(best.estimated_file_bytes)}), "
            f"which is higher quality and should still run at about "
            f"{format_rate(best.tokens_per_second)} tokens/s."
        )
    return None


def scale_profile(model: ModelProfile, factor: float) -> ModelProfile:
    """The same model with every weight resized by factor (the KV cache is unchanged)."""
    return replace(
        model,
        expert_bytes_per_layer=[int(b * factor) for b in model.expert_bytes_per_layer],
        dense_bytes_per_layer=[int(b * factor) for b in model.dense_bytes_per_layer],
        output_bytes=int(model.output_bytes * factor),
        token_embedding_bytes=int(model.token_embedding_bytes * factor),
    )


def _gib(size: int) -> str:
    return f"{size / 1024**3:.0f} GB"
