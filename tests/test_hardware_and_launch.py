from pathlib import Path

from stronghouse.config import load_settings
from stronghouse.core.launch_args import build_server_args
from stronghouse.core.models import KvCacheType, LaunchPlan, OffloadStrategy
from stronghouse.services.hardware_probe import parse_nvidia_smi, probe_hardware, read_meminfo

MIB = 1024**2

MEMINFO = """MemTotal:       32768000 kB
MemFree:         1000000 kB
MemAvailable:   20000000 kB
"""


def test_parse_nvidia_smi_first_gpu():
    name, total, free = parse_nvidia_smi("NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7900\n")
    assert name == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert total == 8188 * MIB
    assert free == 7900 * MIB


def test_parse_nvidia_smi_garbage_means_no_gpu():
    assert parse_nvidia_smi("") == (None, 0, 0)
    assert parse_nvidia_smi("No devices were found") == (None, 0, 0)


def test_read_meminfo():
    assert read_meminfo(MEMINFO) == (32768000 * 1024, 20000000 * 1024)


def test_probe_survives_failing_nvidia_smi(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(MEMINFO)

    def failing(_command):
        raise OSError("no driver")

    profile = probe_hardware(run_command=failing, meminfo_path=meminfo, drm_dir=tmp_path)
    assert not profile.has_gpu
    assert profile.ram_available_bytes == 20000000 * 1024


def plan(**changes) -> LaunchPlan:
    values = dict(
        strategy=OffloadStrategy.MOE_SPLIT,
        n_gpu_layers=49,
        n_cpu_moe=30,
        context_size=8192,
        kv_cache_type=KvCacheType.F16,
        estimated_gpu_bytes=0,
        estimated_cpu_bytes=0,
    )
    values.update(changes)
    return LaunchPlan(**values)


def test_launch_args_include_moe_offload():
    args = build_server_args("llama-server", Path("/m.gguf"), plan(), "127.0.0.1", 8766)
    assert args[0] == "llama-server"
    assert args[args.index("--n-cpu-moe") + 1] == "30"
    assert args[args.index("--n-gpu-layers") + 1] == "49"
    assert "--flash-attn" not in args


def test_launch_args_skip_moe_flag_when_not_needed_and_enable_flash_attn_for_q8():
    args = build_server_args(
        "llama-server", Path("/m.gguf"), plan(n_cpu_moe=0, kv_cache_type=KvCacheType.Q8_0), "h", 1
    )
    assert "--n-cpu-moe" not in args
    assert args[args.index("--cache-type-k") + 1] == "q8_0"
    assert args[args.index("--flash-attn") + 1] == "on"


def test_settings_from_environment():
    settings = load_settings(
        {"STRONGHOUSE_DATA_DIR": "/data", "STRONGHOUSE_PORT": "9000", "HF_TOKEN": "secret"}
    )
    assert settings.archive_dir == Path("/data/archive")
    assert settings.port == 9000
    assert settings.huggingface_token == "secret"


def test_settings_reject_bad_port():
    import pytest

    with pytest.raises(ValueError, match="PORT"):
        load_settings({"STRONGHOUSE_PORT": "abc"})
