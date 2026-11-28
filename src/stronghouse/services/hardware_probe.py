"""Detect the GPU (NVIDIA via nvidia-smi, AMD via sysfs), system RAM and CPU threads on Linux.

Which backend llama-server uses for that GPU (CUDA, Vulkan, ROCm) is decided in
core/gpu_backend.py from what llama-server itself lists.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from stronghouse.core.gpu_backend import combine_hardware, parse_engine_devices
from stronghouse.core.models import GpuBackend, GpuReading, GpuVendor, HardwareProfile
from stronghouse.services.disk_probe import disk_kind

log = logging.getLogger(__name__)
MIB = 1024 * 1024
KIB = 1024
NVIDIA_SMI_TIMEOUT_SECONDS = 5
AMD_PCI_VENDOR = "0x1002"
DRM_DIR = Path("/sys/class/drm")
# PCI device ids of AMD cards worth naming; others show as "AMD GPU" or llama-server's name.
AMD_DEVICE_NAMES = {
    "0x7590": "AMD Radeon RX 9060 XT",
    "0x7550": "AMD Radeon RX 9070 / 9070 XT",
    "0x744c": "AMD Radeon RX 7900 XT / XTX",
    "0x7480": "AMD Radeon RX 7600 / 7600 XT",
    "0x73bf": "AMD Radeon RX 6800 / 6900 XT",
    "0x73df": "AMD Radeon RX 6700 XT",
}
_CARD = re.compile(r"^card\d+$")

CommandRunner = Callable[[list[str]], str]
EngineDeviceLister = Callable[[], str | None]


def probe_hardware(
    run_command: CommandRunner | None = None,
    meminfo_path: Path = Path("/proc/meminfo"),
    drm_dir: Path = DRM_DIR,
    list_engine_devices: EngineDeviceLister | None = None,
    forced_backend: GpuBackend | None = None,
    archive_dir: Path | None = None,
) -> HardwareProfile:
    """Return what this machine has right now. Never raises for a missing GPU.

    Args:
        list_engine_devices: returns `llama-server --list-devices` output, or None.
        forced_backend: STRONGHOUSE_GPU_BACKEND when set.
        archive_dir: the vault folder, to find which kind of drive it is on.
    """
    runner = run_command or _run
    gpu = _probe_nvidia(runner) or read_amd_gpu(drm_dir)
    listing = list_engine_devices() if list_engine_devices else None
    ram_total, ram_available = read_meminfo(meminfo_path.read_text())
    hardware = combine_hardware(
        gpu,
        parse_engine_devices(listing) if listing is not None else None,
        ram_total,
        ram_available,
        os.cpu_count() or 1,
        forced_backend,
    )
    if archive_dir is None:
        return hardware
    return replace(hardware, archive_disk=disk_kind(archive_dir))


def parse_nvidia_smi(output: str) -> tuple[str | None, int, int]:
    """Parse `nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits`.

    Only the first GPU is used; laptops normally have one discrete GPU.
    """
    first_line = output.strip().splitlines()[0] if output.strip() else ""
    parts = [part.strip() for part in first_line.split(",")]
    if len(parts) != 3:
        return None, 0, 0
    name, total_mib, free_mib = parts
    return name, int(float(total_mib)) * MIB, int(float(free_mib)) * MIB


def read_amd_gpu(drm_dir: Path = DRM_DIR) -> GpuReading | None:
    """The AMD card with the most VRAM, from the amdgpu driver's sysfs files.

    amdgpu exposes mem_info_vram_total and mem_info_vram_used in bytes, live, with no
    extra tools installed. The biggest card wins, so a desktop card beats an iGPU.
    """
    readings = [reading for card in _cards(drm_dir) if (reading := _read_amd_card(card))]
    return max(readings, key=lambda reading: reading.total_bytes, default=None)


def read_meminfo(text: str) -> tuple[int, int]:
    """Return (MemTotal, MemAvailable) in bytes from /proc/meminfo contents."""
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            values[key] = int(rest.split()[0]) * KIB
    return values.get("MemTotal", 0), values.get("MemAvailable", 0)


def _cards(drm_dir: Path) -> list[Path]:
    if not drm_dir.is_dir():
        return []
    return sorted(path / "device" for path in drm_dir.iterdir() if _CARD.match(path.name))


def _read_amd_card(device: Path) -> GpuReading | None:
    try:
        if (device / "vendor").read_text().strip().lower() != AMD_PCI_VENDOR:
            return None
        total = int((device / "mem_info_vram_total").read_text())
        used = int((device / "mem_info_vram_used").read_text())
        device_id = (device / "device").read_text().strip().lower()
    except (OSError, ValueError):
        return None
    if total <= 0:
        return None
    name = AMD_DEVICE_NAMES.get(device_id)
    return GpuReading(GpuVendor.AMD, name, total, max(0, total - used))


def _probe_nvidia(runner: CommandRunner) -> GpuReading | None:
    if runner is _run and shutil.which("nvidia-smi") is None:
        log.info("nvidia-smi not found; looking for other GPUs")
        return None
    try:
        output = runner(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ]
        )
    except (OSError, subprocess.SubprocessError) as error:
        log.warning("nvidia-smi failed", extra={"error": str(error)})
        return None
    name, total, free = parse_nvidia_smi(output)
    return GpuReading(GpuVendor.NVIDIA, name, total, free) if name and total > 0 else None


def _run(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_SECONDS
    ).stdout
