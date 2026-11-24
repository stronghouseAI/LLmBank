import pytest

from stronghouse.core.models import (
    HardwareProfile,
    KvCacheType,
    MemoryMode,
    ModelProfile,
    OffloadStrategy,
)
from stronghouse.core.offload_planner import GPU_RESERVE_BYTES, MIN_CONTEXT, plan_offload

GIB = 1024**3
MIB = 1024**2


def moe_model(blocks: int = 48, expert_gib_per_layer: float = 0.35) -> ModelProfile:
    """Roughly shaped like a 30B-A3B MoE at 4-bit."""
    return ModelProfile(
        architecture="qwen3moe",
        block_count=blocks,
        context_length=40960,
        expert_count=128,
        experts_used=8,
        expert_bytes_per_layer=[int(expert_gib_per_layer * GIB)] * blocks,
        dense_bytes_per_layer=[20 * MIB] * blocks,
        output_bytes=300 * MIB,
        token_embedding_bytes=300 * MIB,
        kv_bytes_per_token_f16=48 * 4 * 128 * 2 * 2,  # 48 blocks, 4 kv heads, dim 128
    )


def dense_model(blocks: int = 32, gib_per_layer: float = 0.2) -> ModelProfile:
    return ModelProfile(
        architecture="llama",
        block_count=blocks,
        context_length=8192,
        expert_count=0,
        experts_used=0,
        expert_bytes_per_layer=[0] * blocks,
        dense_bytes_per_layer=[int(gib_per_layer * GIB)] * blocks,
        output_bytes=200 * MIB,
        token_embedding_bytes=200 * MIB,
        kv_bytes_per_token_f16=32 * 8 * 128 * 2 * 2,
    )


def hardware(gpu_gib: float = 8, ram_gib: float = 32) -> HardwareProfile:
    has_gpu = gpu_gib > 0
    return HardwareProfile(
        gpu_name="Test GPU" if has_gpu else None,
        gpu_total_bytes=int(gpu_gib * GIB),
        gpu_free_bytes=int(gpu_gib * GIB),
        ram_total_bytes=int(ram_gib * GIB),
        ram_available_bytes=int(ram_gib * GIB),
        cpu_threads=16,
    )


def test_small_model_goes_fully_on_gpu():
    plan = plan_offload(dense_model(blocks=8), hardware(gpu_gib=24), 4096)
    assert plan.strategy is OffloadStrategy.FULL_GPU
    assert plan.n_cpu_moe == 0
    assert plan.n_gpu_layers == 9


def test_big_moe_on_8gb_gpu_keeps_shared_weights_on_gpu_and_splits_experts():
    model = moe_model()
    plan = plan_offload(model, hardware(gpu_gib=8), 8192)

    assert plan.strategy is OffloadStrategy.MOE_SPLIT
    assert plan.n_gpu_layers == model.block_count + 1
    assert 0 < plan.n_cpu_moe < model.block_count
    assert plan.estimated_gpu_bytes <= 8 * GIB - GPU_RESERVE_BYTES


def test_more_vram_means_fewer_experts_on_cpu():
    model = moe_model()
    small = plan_offload(model, hardware(gpu_gib=8), 8192)
    large = plan_offload(model, hardware(gpu_gib=16), 8192)
    assert large.n_cpu_moe < small.n_cpu_moe


def test_moe_shrinks_context_before_giving_up_on_split():
    model = moe_model()
    plan = plan_offload(model, hardware(gpu_gib=3), 40960)
    assert plan.strategy is OffloadStrategy.MOE_SPLIT
    assert MIN_CONTEXT <= plan.context_size < 40960
    assert any("Context reduced" in note for note in plan.notes)


def test_context_is_capped_at_what_the_model_was_trained_for():
    plan = plan_offload(dense_model(blocks=4), hardware(gpu_gib=24), 1_000_000)
    assert plan.context_size == 8192


def test_q8_kv_cache_frees_room_for_more_experts_on_gpu():
    model = moe_model()
    f16 = plan_offload(model, hardware(gpu_gib=8), 32768, KvCacheType.F16)
    q8 = plan_offload(model, hardware(gpu_gib=8), 32768, KvCacheType.Q8_0)
    assert q8.n_cpu_moe <= f16.n_cpu_moe
    assert q8.n_cpu_moe < f16.n_cpu_moe or q8.estimated_gpu_bytes < f16.estimated_gpu_bytes


def test_dense_model_too_big_for_gpu_splits_whole_layers():
    model = dense_model(blocks=32, gib_per_layer=0.5)
    plan = plan_offload(model, hardware(gpu_gib=8), 4096)
    assert plan.strategy is OffloadStrategy.PARTIAL_LAYERS
    assert 0 < plan.n_gpu_layers < 32
    assert plan.n_cpu_moe == 0


def test_no_gpu_runs_on_cpu():
    plan = plan_offload(moe_model(), hardware(gpu_gib=0), 4096)
    assert plan.strategy is OffloadStrategy.CPU_ONLY
    assert plan.n_gpu_layers == 0


def test_model_bigger_than_ram_streams_from_disk_instead_of_failing():
    plan = plan_offload(moe_model(), hardware(gpu_gib=8, ram_gib=4), 4096)
    assert plan.strategy is OffloadStrategy.MOE_SPLIT
    assert plan.memory_mode is MemoryMode.STREAM_FROM_DISK
    assert any("Big model mode" in note for note in plan.notes)


def test_model_that_fits_in_ram_stays_in_memory():
    plan = plan_offload(moe_model(), hardware(gpu_gib=8, ram_gib=32), 4096)
    assert plan.memory_mode is MemoryMode.IN_MEMORY


def test_rejects_non_positive_context():
    with pytest.raises(ValueError):
        plan_offload(moe_model(), hardware(), 0)
