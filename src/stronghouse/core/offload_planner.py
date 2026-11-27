"""Decide how to split a model between GPU memory and system RAM.

This is the heart of StronghouseAI. For Mixture-of-Experts (MoE) models only a
few experts run per token, so the big expert weights can sit in system RAM while
the small always-used parts (attention, norms, shared layers, KV cache) stay on
the GPU. llama.cpp exposes this as `--n-cpu-moe N`: keep the expert weights of
the first N blocks on the CPU. See docs/adr/0002-offload-planner.md.

The same plan works for CUDA (NVIDIA), Vulkan and ROCm (AMD): --n-cpu-moe and
--n-gpu-layers mean the same thing on every backend. Only the memory each backend
keeps for itself differs (see GPU_RESERVE_BY_BACKEND).

All numbers here are estimates. The real usage also depends on the llama.cpp
build, driver version and batch size, which is why the app offers a speed test
after planning instead of trusting the plan blindly.
"""

from __future__ import annotations

from dataclasses import replace

from stronghouse.core.models import (
    GpuBackend,
    HardwareProfile,
    KvCacheType,
    LaunchPlan,
    MemoryMode,
    ModelProfile,
    OffloadStrategy,
)

MIB = 1024 * 1024

# CUDA context, cuBLAS workspace and llama.cpp compute buffers. Measured values on
# consumer cards are roughly 400-900 MiB; we take the cautious end so a plan does
# not crash with out-of-memory on first load.
GPU_RESERVE_BYTES = 900 * MIB
# Vulkan drivers (RADV on AMD) and the ROCm/HIP runtime keep a little more for
# themselves and their compute buffers than CUDA does, so plan a bit more cautiously.
GPU_RESERVE_BY_BACKEND = {
    GpuBackend.CUDA: GPU_RESERVE_BYTES,
    GpuBackend.VULKAN: 1200 * MIB,
    GpuBackend.ROCM: 1100 * MIB,
    GpuBackend.CPU: 0,
}

# Below this, chat becomes impractical, so we stop shrinking context and move
# whole layers off the GPU instead.
MIN_CONTEXT = 4096

# q8_0 stores 32 values in 34 bytes versus 64 bytes for f16.
_KV_SIZE_FACTOR = {KvCacheType.F16: 1.0, KvCacheType.Q8_0: 34 / 64}


def plan_offload(
    model: ModelProfile,
    hardware: HardwareProfile,
    requested_context: int,
    kv_cache_type: KvCacheType = KvCacheType.F16,
) -> LaunchPlan:
    """Return the fastest layout that should fit in this machine's memory.

    Preference order: everything on GPU, then MoE expert split, then whole
    layers split between GPU and CPU, then CPU only.

    Raises:
        ValueError: if requested_context is not positive.
    """
    if requested_context <= 0:
        raise ValueError(f"requested_context must be positive, got {requested_context}")

    context = min(requested_context, model.context_length or requested_context)
    gpu_budget = hardware.gpu_free_bytes - gpu_reserve_bytes(hardware) if hardware.has_gpu else 0

    if gpu_budget <= 0:
        plan = _cpu_only(model, context, kv_cache_type, note=_no_gpu_note(hardware))
    else:
        plan = (
            _try_full_gpu(model, gpu_budget, context, kv_cache_type)
            or _try_moe_split(model, gpu_budget, context, kv_cache_type)
            or _partial_layers(model, gpu_budget, context, kv_cache_type)
        )
        if plan.strategy is not OffloadStrategy.CPU_ONLY and hardware.gpu_device:
            plan = replace(plan, device=hardware.gpu_device)
    return _with_ram_check(plan, hardware)


def gpu_reserve_bytes(hardware: HardwareProfile) -> int:
    """GPU memory the backend keeps for itself, which a plan must leave free."""
    return GPU_RESERVE_BY_BACKEND[hardware.gpu_backend] if hardware.has_gpu else 0


def _no_gpu_note(hardware: HardwareProfile) -> str:
    if hardware.has_gpu_hardware and hardware.gpu_backend is GpuBackend.CPU:
        return (
            f"{hardware.gpu_name} is not used: this llama-server has no GPU support "
            "(install a Vulkan build, see README)."
        )
    return "No usable GPU memory found."


def kv_cache_bytes(model: ModelProfile, context: int, kv_cache_type: KvCacheType) -> int:
    """Estimated KV cache size for a given context length."""
    return int(model.kv_bytes_per_token_f16 * context * _KV_SIZE_FACTOR[kv_cache_type])


def _try_full_gpu(
    model: ModelProfile, budget: int, context: int, kv: KvCacheType
) -> LaunchPlan | None:
    gpu_bytes = model.total_weight_bytes - model.token_embedding_bytes
    gpu_bytes += kv_cache_bytes(model, context, kv)
    if gpu_bytes > budget:
        return None
    return LaunchPlan(
        strategy=OffloadStrategy.FULL_GPU,
        n_gpu_layers=model.block_count + 1,
        n_cpu_moe=0,
        context_size=context,
        kv_cache_type=kv,
        estimated_gpu_bytes=gpu_bytes,
        estimated_cpu_bytes=model.token_embedding_bytes,
        notes=["The whole model fits on the GPU."],
    )


def _try_moe_split(
    model: ModelProfile, budget: int, context: int, kv: KvCacheType
) -> LaunchPlan | None:
    if not model.is_moe:
        return None

    notes: list[str] = []
    always_on_gpu = model.total_dense_bytes + model.output_bytes
    while always_on_gpu + kv_cache_bytes(model, context, kv) > budget and context > MIN_CONTEXT:
        context = max(MIN_CONTEXT, context // 2)
        notes.append(f"Context reduced to {context} tokens so the shared layers fit on the GPU.")
    base = always_on_gpu + kv_cache_bytes(model, context, kv)
    if base > budget:
        return None

    # llama.cpp's --n-cpu-moe N moves the experts of the FIRST N blocks to the
    # CPU, so the GPU keeps experts from the last blocks. Fill from the end.
    remaining = budget - base
    experts_on_gpu = 0
    for layer_bytes in reversed(model.expert_bytes_per_layer):
        if layer_bytes > remaining:
            break
        remaining -= layer_bytes
        experts_on_gpu += 1

    n_cpu_moe = model.block_count - experts_on_gpu
    gpu_expert_bytes = sum(model.expert_bytes_per_layer[n_cpu_moe:])
    cpu_bytes = model.total_expert_bytes - gpu_expert_bytes + model.token_embedding_bytes
    notes.append(
        f"Experts of {n_cpu_moe} of {model.block_count} blocks run from system RAM; "
        f"attention and shared weights stay on the GPU."
    )
    if kv is KvCacheType.F16:
        notes.append("Tip: a q8_0 KV cache roughly halves cache memory with little quality loss.")
    return LaunchPlan(
        strategy=OffloadStrategy.MOE_SPLIT,
        n_gpu_layers=model.block_count + 1,
        n_cpu_moe=n_cpu_moe,
        context_size=context,
        kv_cache_type=kv,
        estimated_gpu_bytes=base + gpu_expert_bytes,
        estimated_cpu_bytes=cpu_bytes,
        notes=notes,
    )


def _partial_layers(model: ModelProfile, budget: int, context: int, kv: KvCacheType) -> LaunchPlan:
    # llama.cpp offloads the LAST n blocks to the GPU, so count from the end.
    kv_per_layer = kv_cache_bytes(model, context, kv) // max(model.block_count, 1)
    remaining = budget - model.output_bytes
    layers_on_gpu = 0
    gpu_bytes = model.output_bytes
    for dense, experts in zip(
        reversed(model.dense_bytes_per_layer), reversed(model.expert_bytes_per_layer), strict=True
    ):
        layer_bytes = dense + experts + kv_per_layer
        if layer_bytes > remaining:
            break
        remaining -= layer_bytes
        gpu_bytes += layer_bytes
        layers_on_gpu += 1

    if layers_on_gpu == 0:
        return _cpu_only(model, context, kv, note="Not even one layer fits on the GPU.")

    total = model.total_weight_bytes + kv_cache_bytes(model, context, kv)
    return LaunchPlan(
        strategy=OffloadStrategy.PARTIAL_LAYERS,
        n_gpu_layers=layers_on_gpu,
        n_cpu_moe=0,
        context_size=context,
        kv_cache_type=kv,
        estimated_gpu_bytes=gpu_bytes,
        estimated_cpu_bytes=total - gpu_bytes,
        notes=[f"{layers_on_gpu} of {model.block_count} blocks run on the GPU."],
    )


def _cpu_only(
    model: ModelProfile, context: int, kv: KvCacheType, note: str = "No usable GPU memory found."
) -> LaunchPlan:
    return LaunchPlan(
        strategy=OffloadStrategy.CPU_ONLY,
        n_gpu_layers=0,
        n_cpu_moe=0,
        context_size=context,
        kv_cache_type=kv,
        estimated_gpu_bytes=0,
        estimated_cpu_bytes=model.total_weight_bytes + kv_cache_bytes(model, context, kv),
        notes=[note + " Running on CPU only."],
    )


def _with_ram_check(plan: LaunchPlan, hardware: HardwareProfile) -> LaunchPlan:
    """Switch to streaming from disk when the CPU part is bigger than free RAM.

    llama.cpp memory-maps the model file, so weights that do not fit in RAM are read
    back from disk when needed. That is slow but works, which beats not running at
    all. See docs/adr/0011-big-models-on-small-machines.md.
    """
    if plan.estimated_cpu_bytes <= hardware.ram_available_bytes:
        return plan
    note = (
        "Big model mode: the CPU part is larger than free RAM, so the rest streams "
        "from disk (memory-mapped). It works, but slowly; an NVMe drive helps a lot. "
        "Queue a job and walk away, or try a smaller quantization."
    )
    return replace(plan, memory_mode=MemoryMode.STREAM_FROM_DISK, notes=[*plan.notes, note])
