"""AMD GPUs (Vulkan / ROCm), backend detection and machine profiles. See ADR 0018."""

import os
import stat
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stronghouse.app import build_services, create_app
from stronghouse.config import load_settings
from stronghouse.core.gpu_backend import combine_hardware, parse_engine_devices, resolve_backend
from stronghouse.core.hub_filters import FileFit, file_fit
from stronghouse.core.launch_args import build_server_args
from stronghouse.core.machine_profile import machine_profile
from stronghouse.core.models import (
    DiskKind,
    GpuBackend,
    GpuReading,
    GpuVendor,
    HardwareProfile,
    MemorySpeedOverrides,
    OffloadStrategy,
)
from stronghouse.core.offload_planner import GPU_RESERVE_BY_BACKEND, plan_offload
from stronghouse.services.disk_probe import classify_disk, disk_kind
from stronghouse.services.engine import EngineController
from stronghouse.services.hardware_probe import probe_hardware, read_amd_gpu
from tests.test_offload_planner import moe_model

GIB = 1024**3
MIB = 1024**2
GB = 1_000_000_000

VULKAN_LIST = """ggml_vulkan: Found 1 Vulkan devices:
ggml_vulkan: 0 = AMD Radeon RX 9060 XT (RADV GFX1200) (radv) | uma: 0 | fp16: 1
Available devices:
  Vulkan0: AMD Radeon RX 9060 XT (RADV GFX1200) (16304 MiB, 15800 MiB free)
"""
TWO_VULKAN = """Available devices:
  Vulkan0: AMD Radeon Graphics (RADV RAPHAEL_MENDOCINO) (2048 MiB, 1900 MiB free)
  Vulkan1: AMD Radeon RX 9060 XT (RADV GFX1200) (16304 MiB, 15800 MiB free)
"""
CUDA_LIST = """Available devices:
  CUDA0: NVIDIA GeForce RTX 4060 Laptop GPU (8188 MiB, 7800 MiB free)
"""
ROCM_LIST = """Available devices:
  ROCm0: AMD Radeon RX 9060 XT (16304 MiB, 16000 MiB free)
"""
CPU_ONLY_LIST = "Available devices:\n"
MEMINFO = "MemTotal:       32768000 kB\nMemAvailable:   28000000 kB\n"

RX_9060_XT = GpuReading(GpuVendor.AMD, "AMD Radeon RX 9060 XT", 16 * GIB, 15 * GIB)


def desktop(backend: GpuBackend = GpuBackend.VULKAN) -> HardwareProfile:
    """phil's desktop: RX 9060 XT 16 GB, 32 GB DDR4, Ryzen 9 3900XT."""
    return HardwareProfile(
        "AMD Radeon RX 9060 XT", 16 * GIB, 16 * GIB, 32 * GIB, 32 * GIB, 24,
        gpu_vendor=GpuVendor.AMD, gpu_backend=backend,
    )  # fmt: skip


# ---------- llama-server --list-devices ----------


def test_parse_vulkan_device_list():
    [device] = parse_engine_devices(VULKAN_LIST)
    assert device.name == "Vulkan0"
    assert device.backend is GpuBackend.VULKAN
    assert device.description == "AMD Radeon RX 9060 XT (RADV GFX1200)"
    assert device.total_bytes == 16304 * MIB
    assert device.free_bytes == 15800 * MIB


@pytest.mark.parametrize(
    ("listing", "backend"),
    [(CUDA_LIST, GpuBackend.CUDA), (ROCM_LIST, GpuBackend.ROCM), (VULKAN_LIST, GpuBackend.VULKAN)],
)
def test_backend_comes_from_the_llama_server_build(listing, backend):
    assert resolve_backend(GpuVendor.AMD, parse_engine_devices(listing)) is backend


def test_cpu_only_build_lists_no_devices():
    assert parse_engine_devices(CPU_ONLY_LIST) == []
    assert resolve_backend(GpuVendor.AMD, []) is GpuBackend.CPU


def test_unreadable_listing_means_unknown():
    assert parse_engine_devices("error: invalid argument: --list-devices") is None


def test_unknown_build_assumes_the_usual_one_per_vendor():
    assert resolve_backend(GpuVendor.NVIDIA, None) is GpuBackend.CUDA
    assert resolve_backend(GpuVendor.AMD, None) is GpuBackend.VULKAN


def test_setting_wins_over_the_build():
    devices = parse_engine_devices(VULKAN_LIST)
    assert resolve_backend(GpuVendor.AMD, devices, GpuBackend.ROCM) is GpuBackend.ROCM


# ---------- combining the OS view and the engine view ----------


def test_amd_card_with_vulkan_build_is_used():
    hardware = combine_hardware(
        RX_9060_XT, parse_engine_devices(VULKAN_LIST), 32 * GIB, 28 * GIB, 24
    )
    assert hardware.has_gpu
    assert hardware.gpu_vendor is GpuVendor.AMD
    assert hardware.gpu_backend is GpuBackend.VULKAN
    assert hardware.gpu_free_bytes == 15 * GIB  # live sysfs number, not the cached listing
    assert hardware.gpu_device is None  # only one GPU, nothing to pin


def test_amd_card_with_cpu_only_build_is_not_used():
    hardware = combine_hardware(RX_9060_XT, [], 32 * GIB, 28 * GIB, 24)
    assert hardware.has_gpu_hardware
    assert not hardware.has_gpu
    plan = plan_offload(moe_model(), hardware, 8192)
    assert plan.strategy is OffloadStrategy.CPU_ONLY
    assert "has no GPU support" in plan.notes[0]


def test_unnamed_card_takes_the_engine_name():
    unnamed = GpuReading(GpuVendor.AMD, None, 16 * GIB, 15 * GIB)
    hardware = combine_hardware(unnamed, parse_engine_devices(VULKAN_LIST), 1, 1, 1)
    assert hardware.gpu_name == "AMD Radeon RX 9060 XT (RADV GFX1200)"
    assert combine_hardware(unnamed, None, 1, 1, 1).gpu_name == "AMD GPU"


def test_card_only_llama_server_knows_about_still_counts():
    hardware = combine_hardware(None, parse_engine_devices(VULKAN_LIST), 1, 1, 1)
    assert hardware.has_gpu
    assert hardware.gpu_vendor is GpuVendor.AMD
    assert hardware.gpu_total_bytes == 16304 * MIB


def test_no_gpu_anywhere_is_cpu_only():
    hardware = combine_hardware(None, None, 32 * GIB, 28 * GIB, 24)
    assert not hardware.has_gpu
    assert hardware.gpu_backend is GpuBackend.CPU


def test_several_gpus_pin_the_biggest_one():
    hardware = combine_hardware(RX_9060_XT, parse_engine_devices(TWO_VULKAN), 1, 1, 1)
    assert hardware.gpu_device == "Vulkan1"
    plan = plan_offload(moe_model(blocks=4), hardware, 4096)
    assert plan.device == "Vulkan1"
    args = build_server_args("llama-server", Path("/m.gguf"), plan, "127.0.0.1", 8766)
    assert args[args.index("--device") + 1] == "Vulkan1"


def test_single_gpu_needs_no_device_flag():
    plan = plan_offload(moe_model(blocks=4), desktop(), 4096)
    args = build_server_args("llama-server", Path("/m.gguf"), plan, "127.0.0.1", 8766)
    assert "--device" not in args


# ---------- sysfs (amdgpu driver) ----------


def write_card(drm: Path, card: str, vendor: str, device: str, total: int, used: int) -> None:
    folder = drm / card / "device"
    folder.mkdir(parents=True)
    (folder / "vendor").write_text(vendor + "\n")
    (folder / "device").write_text(device + "\n")
    (folder / "mem_info_vram_total").write_text(f"{total}\n")
    (folder / "mem_info_vram_used").write_text(f"{used}\n")


def test_read_amd_gpu_picks_the_biggest_amd_card(tmp_path: Path):
    write_card(tmp_path, "card0", "0x1002", "0x164e", 512 * MIB, 100 * MIB)  # iGPU
    write_card(tmp_path, "card1", "0x1002", "0x7590", 16 * GIB, 1 * GIB)
    write_card(tmp_path, "card2", "0x10de", "0x28e0", 24 * GIB, 0)  # not AMD
    (tmp_path / "card1-DP-1").mkdir()  # a connector, not a card
    reading = read_amd_gpu(tmp_path)
    assert reading == GpuReading(GpuVendor.AMD, "AMD Radeon RX 9060 XT", 16 * GIB, 15 * GIB)


def test_read_amd_gpu_without_driver_files(tmp_path: Path):
    (tmp_path / "card0" / "device").mkdir(parents=True)
    (tmp_path / "card0" / "device" / "vendor").write_text("0x1002\n")
    assert read_amd_gpu(tmp_path) is None
    assert read_amd_gpu(tmp_path / "missing") is None


def test_probe_finds_amd_desktop(tmp_path: Path):
    write_card(tmp_path / "drm", "card1", "0x1002", "0x7590", 16 * GIB, 1 * GIB)
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(MEMINFO)

    def no_nvidia(_command):
        raise OSError("nvidia-smi: not found")

    hardware = probe_hardware(
        run_command=no_nvidia,
        meminfo_path=meminfo,
        drm_dir=tmp_path / "drm",
        list_engine_devices=lambda: VULKAN_LIST,
    )
    assert hardware.gpu_name == "AMD Radeon RX 9060 XT"
    assert hardware.gpu_backend is GpuBackend.VULKAN
    assert hardware.gpu_free_bytes == 15 * GIB


def test_probe_prefers_nvidia_smi(tmp_path: Path):
    meminfo = tmp_path / "meminfo"
    meminfo.write_text(MEMINFO)
    hardware = probe_hardware(
        run_command=lambda _command: "NVIDIA GeForce RTX 4060 Laptop GPU, 8188, 7900\n",
        meminfo_path=meminfo,
        drm_dir=tmp_path / "none",
    )
    assert hardware.gpu_vendor is GpuVendor.NVIDIA
    assert hardware.gpu_backend is GpuBackend.CUDA


def test_engine_lists_devices_once(tmp_path: Path):
    calls = tmp_path / "calls"
    fake = tmp_path / "llama-server"
    fake.write_text(f"#!/bin/sh\necho x >> {calls}\nprintf 'Available devices:\\n'\n")
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    engine = EngineController(str(fake), "127.0.0.1", 1, tmp_path / "engine.log")
    assert engine.list_devices() == "Available devices:\n"
    assert engine.list_devices() == "Available devices:\n"
    assert calls.read_text().count("x") == 1


def test_engine_list_devices_without_llama_server(tmp_path: Path):
    engine = EngineController(str(tmp_path / "missing"), "127.0.0.1", 1, tmp_path / "engine.log")
    assert engine.list_devices() is None


# ---------- planning and machine profiles ----------


def test_vulkan_plan_keeps_its_bigger_reserve_free():
    plan = plan_offload(moe_model(), desktop(), 8192)
    assert plan.strategy is OffloadStrategy.MOE_SPLIT
    assert plan.estimated_gpu_bytes <= 16 * GIB - GPU_RESERVE_BY_BACKEND[GpuBackend.VULKAN]


def test_desktop_keeps_more_experts_on_gpu_than_laptop():
    laptop = HardwareProfile("RTX 4060 Laptop", 8 * GIB, 8 * GIB, 16 * GIB, 16 * GIB, 20)
    on_desktop = plan_offload(moe_model(), desktop(), 8192)
    on_laptop = plan_offload(moe_model(), laptop, 8192)
    assert on_desktop.n_cpu_moe < on_laptop.n_cpu_moe


def test_desktop_size_limits():
    profile = machine_profile(desktop())
    assert profile.backend is GpuBackend.VULKAN
    assert 13 * GIB < profile.gpu_limit_bytes < 14 * GIB
    assert 41 * GIB < profile.in_memory_limit_bytes < 42 * GIB
    assert profile.summary == "AMD Radeon RX 9060 XT (17.2 GB) via Vulkan · 34.4 GB RAM"
    assert profile.warnings == []


def test_file_fit_on_the_desktop():
    assert file_fit(12 * GIB, desktop()) is FileFit.GPU
    assert file_fit(40 * GIB, desktop()) is FileFit.OFFLOAD  # too big for the laptop
    assert file_fit(60 * GIB, desktop()) is FileFit.TOO_BIG  # still runs, from disk


def test_speeds_are_picked_per_card_unless_set():
    assert machine_profile(desktop()).memory_speeds.gpu_bytes_per_second == 250 * GB
    laptop = HardwareProfile("NVIDIA GeForce RTX 4060 Laptop GPU", 8 * GIB, 8 * GIB, 1, 1, 1)
    assert machine_profile(laptop).memory_speeds.gpu_bytes_per_second == 180 * GB
    overrides = MemorySpeedOverrides(gpu_bytes_per_second=300 * GB)
    speeds = machine_profile(desktop(), overrides).memory_speeds
    assert speeds.gpu_bytes_per_second == 300 * GB
    assert speeds.ram_bytes_per_second == 40 * GB


def test_warnings_explain_an_unusable_gpu():
    [cpu_build] = machine_profile(desktop(GpuBackend.CPU)).warnings
    assert "no GPU support" in cpu_build and "Vulkan" in cpu_build
    [wrong] = machine_profile(desktop(GpuBackend.CUDA)).warnings
    assert "STRONGHOUSE_GPU_BACKEND=vulkan" in wrong
    assert machine_profile(desktop(GpuBackend.ROCM)).warnings == []


# ---------- settings and API ----------


def test_gpu_backend_setting(tmp_path: Path):
    def backend(value: str):
        env = {"STRONGHOUSE_DATA_DIR": str(tmp_path), "STRONGHOUSE_GPU_BACKEND": value}
        return load_settings(env).gpu_backend

    assert backend("auto") is None
    assert backend("Vulkan") is GpuBackend.VULKAN
    assert backend("rocm") is GpuBackend.ROCM
    with pytest.raises(ValueError, match="GPU_BACKEND"):
        backend("metal")


def test_speeds_unset_by_default(tmp_path: Path):
    settings = load_settings({"STRONGHOUSE_DATA_DIR": str(tmp_path)})
    assert settings.speed_overrides == MemorySpeedOverrides()


def test_hardware_endpoint_includes_the_machine_profile(tmp_path: Path):
    services = build_services(load_settings({"STRONGHOUSE_DATA_DIR": str(tmp_path)}))
    services.probe_hardware = desktop
    with TestClient(create_app(services)) as client:
        body = client.get("/api/hardware").json()
    assert body["gpu_vendor"] == "amd"
    assert body["machine"]["backend"] == "vulkan"
    assert body["machine"]["in_memory_limit_bytes"] > 41 * GIB


# ---------- the vault's drive ----------


def fake_block_device(tmp_path: Path, folder: Path, target: str) -> Path:
    """sysfs where the device behind folder resolves to devices/<target>."""
    block = tmp_path / "sys" / "dev" / "block"
    block.mkdir(parents=True, exist_ok=True)
    device = os.stat(folder).st_dev
    link = block / f"{os.major(device)}:{os.minor(device)}"
    link.symlink_to(tmp_path / "sys" / "devices" / target)
    return block


def make_disk(tmp_path: Path, name: str, rotational: str, partition: str | None = None) -> None:
    disk = tmp_path / "sys" / "devices" / name
    (disk / "queue").mkdir(parents=True)
    (disk / "queue" / "rotational").write_text(rotational + "\n")
    if partition:
        (disk / partition).mkdir()
        (disk / partition / "partition").write_text("1\n")


@pytest.mark.parametrize(
    ("name", "rotational", "kind"),
    [
        ("nvme0n1", "0", DiskKind.NVME),
        ("sda", "0", DiskKind.SATA_SSD),
        ("sdb", "1", DiskKind.HDD),
        ("sdc", None, DiskKind.UNKNOWN),
    ],
)
def test_classify_disk(name, rotational, kind):
    assert classify_disk(name, rotational) is kind


def test_vault_on_a_sata_ssd_partition(tmp_path: Path):
    vault = tmp_path / "mnt" / "shai" / "stronghouse"
    vault.mkdir(parents=True)
    make_disk(tmp_path, "sda", "0", partition="sda1")
    block = fake_block_device(tmp_path, vault, "sda/sda1")
    assert disk_kind(vault, block) is DiskKind.SATA_SSD
    # A vault not created yet is judged by its nearest existing parent.
    assert disk_kind(vault / "archive" / "new", block) is DiskKind.SATA_SSD


def test_vault_on_an_encrypted_nvme_volume(tmp_path: Path):
    make_disk(tmp_path, "nvme0n1", "0", partition="nvme0n1p3")
    mapped = tmp_path / "sys" / "devices" / "dm-0"
    (mapped / "slaves").mkdir(parents=True)
    (mapped / "slaves" / "nvme0n1p3").symlink_to(tmp_path / "sys/devices/nvme0n1/nvme0n1p3")
    block = fake_block_device(tmp_path, tmp_path, "dm-0")
    assert disk_kind(tmp_path, block) is DiskKind.NVME


def test_vault_without_a_block_device(tmp_path: Path):
    assert disk_kind(tmp_path, tmp_path / "no-sysfs") is DiskKind.UNKNOWN


# ---------- phil's desktop, end to end ----------


def phils_desktop() -> HardwareProfile:
    """RX 9060 XT 16 GB (Vulkan), 32 GB DDR4, Ryzen 9 3900XT, vault on a SATA SSD."""
    return HardwareProfile(
        "AMD Radeon RX 9060 XT", 16304 * MIB, 15500 * MIB, 32 * GIB, 28 * GIB, 24,
        gpu_vendor=GpuVendor.AMD, gpu_backend=GpuBackend.VULKAN,
        archive_disk=DiskKind.SATA_SSD,
    )  # fmt: skip


def test_phils_desktop_profile():
    profile = machine_profile(phils_desktop())
    assert profile.summary == (
        "AMD Radeon RX 9060 XT (17.1 GB) via Vulkan · 34.4 GB RAM · vault on SATA SSD"
    )
    assert profile.memory_speeds.gpu_bytes_per_second == 250 * GB
    assert profile.memory_speeds.disk_bytes_per_second == 0.45 * GB
    assert 44 * GB < profile.in_memory_limit_bytes < 45 * GB
    assert profile.warnings == []


def test_phils_desktop_runs_a_30b_moe_mostly_on_the_gpu():
    plan = plan_offload(moe_model(), phils_desktop(), 8192)
    assert plan.strategy is OffloadStrategy.MOE_SPLIT
    assert plan.n_cpu_moe < 24  # over half the expert blocks stay on the GPU
    assert plan.device is None


def test_hard_disk_vault_gets_a_warning():
    [warning] = machine_profile(replace(phils_desktop(), archive_disk=DiskKind.HDD)).warnings
    assert "hard disk" in warning
