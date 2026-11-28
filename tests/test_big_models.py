"""Big model mode: streaming plans, speed forecasts, quantization advice and speed flags."""

from dataclasses import replace
from pathlib import Path

import pytest

from stronghouse.config import load_settings
from stronghouse.core.launch_args import (
    ExtraArgsError,
    build_server_args,
    parse_extra_args,
    streaming_conflicts,
)
from stronghouse.core.models import (
    HardwareProfile,
    KvCacheType,
    MemoryMode,
    MemorySpeeds,
    ModelProfile,
    PlanSettings,
    RunTuning,
)
from stronghouse.core.offload_planner import plan_offload
from stronghouse.core.quant_advisor import detect_quant, quant_options, quant_tip, scale_profile
from stronghouse.core.run_setup import forecast, plan_with_settings
from stronghouse.core.speed_estimator import active_bytes_per_token, describe_wait, estimate_speed

GIB = 1024**3
MIB = 1024**2
GB = 1_000_000_000
SPEEDS = MemorySpeeds(180 * GB, 40 * GB, 2 * GB)


def big_moe(expert_gib: float = 60, blocks: int = 46) -> ModelProfile:
    """Shaped like a 106B-A12B MoE (GLM-4.5-Air class) at about 4 bits."""
    return ModelProfile(
        architecture="glm4moe",
        block_count=blocks,
        context_length=131072,
        expert_count=128,
        experts_used=8,
        expert_bytes_per_layer=[int(expert_gib * GIB / blocks)] * blocks,
        dense_bytes_per_layer=[int(4 * GIB / blocks)] * blocks,
        output_bytes=600 * MIB,
        token_embedding_bytes=600 * MIB,
        kv_bytes_per_token_f16=blocks * 8 * 128 * 4,
    )


def laptop(ram_free_gib: float = 12) -> HardwareProfile:
    """phil's laptop: RTX 4060 Laptop 8 GB, 16 GB RAM, 20 threads."""
    return HardwareProfile("RTX 4060 Laptop", 8 * GIB, int(7.6 * GIB), 16 * GIB,
                           int(ram_free_gib * GIB), 20)  # fmt: skip


# ----- planning and speed -----


def test_model_far_bigger_than_ram_still_gets_a_working_plan():
    plan = plan_offload(big_moe(), laptop(), 8192)
    assert plan.memory_mode is MemoryMode.STREAM_FROM_DISK
    assert plan.n_gpu_layers == 47  # attention and shared weights stay on the GPU
    assert plan.n_cpu_moe > 0


def test_only_active_experts_count_towards_bytes_per_token():
    model = big_moe()
    plan = plan_offload(model, laptop(ram_free_gib=200), 8192)
    gpu, cpu = active_bytes_per_token(model, plan)
    per_token = gpu + cpu
    # 8 of 128 experts: about 1/16 of the expert bytes, plus every shared weight.
    expected = model.total_expert_bytes / 16 + model.total_dense_bytes + model.output_bytes
    assert per_token == pytest.approx(expected, rel=0.01)


def test_streaming_from_disk_is_forecast_much_slower_than_ram():
    model = big_moe()
    in_ram = plan_offload(model, laptop(ram_free_gib=200), 8192)
    streaming = plan_offload(model, laptop(ram_free_gib=12), 8192)
    fast = estimate_speed(model, in_ram, laptop(ram_free_gib=200), SPEEDS)
    slow = estimate_speed(model, streaming, laptop(ram_free_gib=12), SPEEDS)
    assert fast.disk_bytes_per_token == 0
    assert slow.disk_bytes_per_token > 0
    assert slow.tokens_per_second < fast.tokens_per_second / 3
    assert 0.1 < slow.tokens_per_second < 2  # slow, but it runs


def test_small_models_are_not_forecast_absurdly_fast():
    tiny = scale_profile(big_moe(expert_gib=2, blocks=8), 0.1)
    plan = plan_offload(tiny, laptop(), 4096)
    assert estimate_speed(tiny, plan, laptop(), SPEEDS).tokens_per_second < 130


def test_describe_wait_uses_friendly_units():
    model = big_moe()
    plan = plan_offload(model, laptop(), 8192)
    text = describe_wait(estimate_speed(model, plan, laptop(), SPEEDS))
    assert "tokens/s" in text and ("minutes" in text or "hours" in text)


# ----- quantization advice -----


@pytest.mark.parametrize(
    ("filename", "quant"),
    [
        ("GLM-4.5-Air-Q4_K_M.gguf", "Q4_K_M"),
        ("Qwen3-235B-A22B-UD-Q2_K_XL-00001-of-00002.gguf", "Q2_K"),
        ("DeepSeek-V3-IQ2_XXS.gguf", "IQ2_XXS"),
        ("model-BF16.gguf", "BF16"),
        ("Qwen3-8B.gguf", None),
    ],
)
def test_detect_quant(filename, quant):
    assert detect_quant(filename) == quant


def test_quant_options_mark_current_and_shrink_with_bits():
    options = quant_options(big_moe(), "GLM-Q4_K_M.gguf", laptop(), 8192, KvCacheType.F16, SPEEDS)
    labels = [o.label for o in options]
    assert labels.index("Q2_K") < labels.index("Q4_K_M") < labels.index("Q8_0")
    assert [o.label for o in options if o.is_current] == ["Q4_K_M"]
    sizes = [o.estimated_file_bytes for o in options]
    assert sizes == sorted(sizes)


def test_tip_points_to_a_quant_that_fits_in_ram():
    model = big_moe(expert_gib=20)  # about 25 GB at Q4: streams on 12 GB free RAM
    options = quant_options(model, "m-Q4_K_M.gguf", laptop(), 8192, KvCacheType.F16, SPEEDS)
    tip = quant_tip(options)
    assert tip is not None and "would fit in RAM" in tip


def test_tip_says_so_when_nothing_fits():
    options = quant_options(
        big_moe(expert_gib=400), "m-Q4_K_M.gguf", laptop(), 8192, KvCacheType.F16, SPEEDS
    )
    assert "streams from disk whatever you pick" in quant_tip(options)


def test_unknown_quant_gives_no_options():
    assert quant_options(big_moe(), "m.gguf", laptop(), 8192, KvCacheType.F16, SPEEDS) == []


# ----- speed flags -----


def test_extra_args_from_a_pasted_terminal_command():
    pasted = "~/llama.cpp/build/bin/llama-server -m /models/glm.gguf --port 8080 -ngl 99 -b 4096"
    assert parse_extra_args(pasted) == ("-ngl", "99", "-b", "4096")
    assert parse_extra_args("--host=0.0.0.0 --no-warmup") == ("--no-warmup",)
    assert parse_extra_args("") == ()


def test_extra_args_reject_broken_quotes():
    with pytest.raises(ExtraArgsError):
        parse_extra_args('--alias "unclosed')


def test_launch_args_add_tuning_and_put_extra_flags_last():
    plan = plan_offload(big_moe(), laptop(ram_free_gib=200), 8192)
    plan = replace(plan, tuning=RunTuning(threads=6, batch_size=4096, ubatch_size=1024,
                                          extra_args=("-ngl", "40")))  # fmt: skip
    args = build_server_args("llama-server", Path("/m.gguf"), plan, "127.0.0.1", 8766)
    assert args[args.index("--threads") + 1] == "6"
    assert args[args.index("--batch-size") + 1] == "4096"
    assert args[args.index("--ubatch-size") + 1] == "1024"
    assert args[-2:] == ["-ngl", "40"]


def test_default_launch_leaves_threads_and_batch_to_llama_cpp():
    plan = plan_offload(big_moe(), laptop(), 8192)
    args = build_server_args("llama-server", Path("/m.gguf"), plan, "h", 1)
    assert "--threads" not in args and "--batch-size" not in args


def test_mlock_is_flagged_when_the_model_must_stream():
    model = big_moe()
    plan = plan_with_settings(model, laptop(), PlanSettings(extra_args="--mlock"))
    assert streaming_conflicts(plan) == ["--mlock"]
    assert any("--mlock" in note for note in plan.notes)


def test_plan_settings_overrides_and_default_extra_args():
    model = big_moe()
    plan = plan_with_settings(
        model, laptop(), PlanSettings(n_cpu_moe=999, threads=8), default_extra_args="-b 2048"
    )
    assert plan.n_cpu_moe == model.block_count
    assert plan.tuning.threads == 8
    assert plan.tuning.extra_args == ("-b", "2048")
    # An explicit empty string means "no extra flags", not "use the default".
    plan = plan_with_settings(model, laptop(), PlanSettings(extra_args=""), "-b 2048")
    assert plan.tuning.extra_args == ()


def test_forecast_bundles_speed_and_quant_advice():
    model = big_moe()
    plan = plan_offload(model, laptop(), 8192)
    outlook = forecast(model, plan, laptop(), "GLM-4.5-Air-Q4_K_M.gguf", SPEEDS)
    assert outlook.speed.tokens_per_second > 0
    assert outlook.quant_options and "tokens/s" in outlook.wait


# ----- settings -----


def test_speed_and_job_settings_from_environment():
    settings = load_settings(
        {
            "STRONGHOUSE_DATA_DIR": "/data",
            "STRONGHOUSE_DISK_GBPS": "5.5",
            "STRONGHOUSE_EXTRA_ARGS": "-b 4096",
            "STRONGHOUSE_NOTIFY": "off",
        }
    )
    assert settings.speed_overrides.disk_bytes_per_second == 5.5 * GB
    assert settings.extra_args == "-b 4096"
    assert settings.notify is False
    assert settings.jobs_output_dir == Path("/data/jobs")


def test_bad_speed_setting_is_explained():
    with pytest.raises(ValueError, match="DISK_GBPS"):
        load_settings({"STRONGHOUSE_DISK_GBPS": "fast"})
