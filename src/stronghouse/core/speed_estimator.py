"""Forecast generation speed for a plan, so a big model's slowness is known up front.

Generating one token reads every *active* weight once: all shared weights, plus
only the routed experts the model picks (experts_used of expert_count). On home
hardware that reading, not the maths, sets the speed. So the estimate is simply

    seconds per token = overhead + GPU bytes / VRAM speed + RAM bytes / RAM speed
                        + disk bytes / disk speed

where "disk bytes" are the expert weights that do not fit in RAM and are read
back from the memory-mapped file. It is deliberately rough (within about 2x) and
is shown as "about". The speed test after loading gives the real number.
See docs/adr/0011-big-models-on-small-machines.md.
"""

from __future__ import annotations

from stronghouse.core.models import (
    HardwareProfile,
    LaunchPlan,
    MemoryMode,
    MemorySpeeds,
    ModelProfile,
    SpeedEstimate,
)

# Kernel launches, GPU/CPU hand-overs and sampling cost a few milliseconds per token
# whatever the model size; without this, tiny models would be forecast absurdly fast.
PER_TOKEN_OVERHEAD_SECONDS = 0.008
# Linux needs some RAM for itself and the app; the rest can cache the model file.
RAM_HEADROOM_SHARE = 0.1


def estimate_speed(
    model: ModelProfile, plan: LaunchPlan, hardware: HardwareProfile, speeds: MemorySpeeds
) -> SpeedEstimate:
    """Estimate tokens per second for generating with this plan on this machine."""
    gpu_bytes, cpu_bytes = active_bytes_per_token(model, plan)
    disk_bytes = int(cpu_bytes * _disk_share(plan, hardware))
    ram_bytes = cpu_bytes - disk_bytes
    seconds = (
        PER_TOKEN_OVERHEAD_SECONDS
        + gpu_bytes / speeds.gpu_bytes_per_second
        + ram_bytes / speeds.ram_bytes_per_second
        + disk_bytes / speeds.disk_bytes_per_second
    )
    tokens_per_second = 1 / seconds
    return SpeedEstimate(
        tokens_per_second=round(tokens_per_second, 2),
        gpu_bytes_per_token=gpu_bytes,
        ram_bytes_per_token=ram_bytes,
        disk_bytes_per_token=disk_bytes,
        seconds_per_1000_tokens=round(seconds * 1000, 1),
    )


def active_bytes_per_token(model: ModelProfile, plan: LaunchPlan) -> tuple[int, int]:
    """Bytes read per generated token as (on the GPU, on the CPU side)."""
    expert_share = model.experts_used / model.expert_count if model.is_moe else 1.0
    gpu_blocks = min(plan.n_gpu_layers, model.block_count)
    first_gpu_block = model.block_count - gpu_blocks
    gpu = cpu = 0
    for block, (dense, experts) in enumerate(
        zip(model.dense_bytes_per_layer, model.expert_bytes_per_layer, strict=True)
    ):
        active_experts = int(experts * expert_share)
        block_on_gpu = block >= first_gpu_block
        experts_on_gpu = block_on_gpu and block >= plan.n_cpu_moe
        gpu += dense if block_on_gpu else 0
        cpu += 0 if block_on_gpu else dense
        gpu += active_experts if experts_on_gpu else 0
        cpu += 0 if experts_on_gpu else active_experts
    # llama.cpp puts the output layer on the GPU once every block is offloaded.
    if plan.n_gpu_layers > model.block_count:
        gpu += model.output_bytes
    else:
        cpu += model.output_bytes
    return gpu, cpu


def _disk_share(plan: LaunchPlan, hardware: HardwareProfile) -> float:
    """Share of CPU-side weights that must come from disk because RAM is too small.

    Experts are picked roughly at random per token, so the chance a needed expert is
    already cached in RAM is about (RAM that can cache the file) / (CPU-side bytes).
    """
    if plan.memory_mode is not MemoryMode.STREAM_FROM_DISK or plan.estimated_cpu_bytes <= 0:
        return 0.0
    cache = hardware.ram_available_bytes * (1 - RAM_HEADROOM_SHARE)
    return max(0.0, 1 - cache / plan.estimated_cpu_bytes)


def describe_wait(estimate: SpeedEstimate, tokens: int = 1000) -> str:
    """Plain words for how long an answer of this many tokens takes (about 750 words)."""
    seconds = estimate.seconds_per_1000_tokens * tokens / 1000
    if seconds < 90:
        span = f"{round(seconds)} seconds"
    elif seconds < 90 * 60:
        span = f"{round(seconds / 60)} minutes"
    else:
        span = f"{seconds / 3600:.1f} hours"
    rate = format_rate(estimate.tokens_per_second)
    return f"about {rate} tokens/s, so a {tokens}-token answer takes {span}"


def format_rate(tokens_per_second: float) -> str:
    """Round a speed to what is meaningful for an estimate: 125, 4.2 or 0.24."""
    if tokens_per_second >= 10:
        return f"{tokens_per_second:.0f}"
    if tokens_per_second >= 1:
        return f"{tokens_per_second:.1f}"
    return f"{tokens_per_second:.2f}"
