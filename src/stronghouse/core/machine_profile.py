"""Work out what this computer can run: backend, model size limits and memory speeds.

So each of phil's machines (RTX 4060 laptop, Radeon RX 9060 XT desktop, a 64 GB
mini PC) gets sensible numbers without hand-tuning. Settings still win: any speed
set with STRONGHOUSE_*_GBPS replaces the guess here.
See docs/adr/0018-amd-gpus-and-machine-profiles.md.
"""

from __future__ import annotations

from stronghouse.core.hub_filters import fit_limits
from stronghouse.core.models import (
    GpuBackend,
    GpuVendor,
    HardwareProfile,
    MachineProfile,
    MemorySpeedOverrides,
    MemorySpeeds,
)
from stronghouse.core.offload_planner import gpu_reserve_bytes

GB = 1_000_000_000
# Effective (not peak) VRAM read speed in GB/s for cards we know; about 70-80% of
# the spec sheet. Checked in order, so put longer names first.
KNOWN_GPU_GBPS = (
    ("RTX 4060 Laptop", 180.0),  # 256 GB/s peak
    ("RX 9070", 500.0),  # 9070 and 9070 XT, 640 GB/s peak
    ("RX 9060 XT", 250.0),  # 320 GB/s peak
)
DEFAULT_GPU_GBPS = 180.0
# Dual-channel DDR4-3200 and DDR5-5600 both land near this in practice.
DEFAULT_RAM_GBPS = 40.0
DEFAULT_DISK_GBPS = 2.0

BACKEND_LABELS = {
    GpuBackend.CUDA: "CUDA",
    GpuBackend.VULKAN: "Vulkan",
    GpuBackend.ROCM: "ROCm",
    GpuBackend.CPU: "CPU only",
}
# Backends that can drive a vendor's cards at all.
_WORKS_WITH = {
    GpuBackend.CUDA: {GpuVendor.NVIDIA},
    GpuBackend.ROCM: {GpuVendor.AMD},
}


def machine_profile(
    hardware: HardwareProfile, overrides: MemorySpeedOverrides | None = None
) -> MachineProfile:
    """Everything the app assumes about this machine, in one place."""
    gpu_limit, in_memory_limit = fit_limits(hardware)
    return MachineProfile(
        summary=_summary(hardware),
        backend=hardware.gpu_backend if hardware.has_gpu else GpuBackend.CPU,
        gpu_reserve_bytes=gpu_reserve_bytes(hardware),
        gpu_limit_bytes=gpu_limit,
        in_memory_limit_bytes=in_memory_limit,
        memory_speeds=memory_speeds(hardware, overrides),
        warnings=_warnings(hardware),
    )


def memory_speeds(
    hardware: HardwareProfile, overrides: MemorySpeedOverrides | None = None
) -> MemorySpeeds:
    """Read speeds for the speed forecast: the person's settings, else a guess for this card."""
    given = overrides or MemorySpeedOverrides()
    return MemorySpeeds(
        gpu_bytes_per_second=given.gpu_bytes_per_second or _gpu_gbps(hardware) * GB,
        ram_bytes_per_second=given.ram_bytes_per_second or DEFAULT_RAM_GBPS * GB,
        disk_bytes_per_second=given.disk_bytes_per_second or DEFAULT_DISK_GBPS * GB,
    )


def _gpu_gbps(hardware: HardwareProfile) -> float:
    name = hardware.gpu_name or ""
    for known, gbps in KNOWN_GPU_GBPS:
        if known.lower() in name.lower():
            return gbps
    return DEFAULT_GPU_GBPS


def _summary(hardware: HardwareProfile) -> str:
    ram = f"{_gb(hardware.ram_total_bytes)} RAM"
    if not hardware.has_gpu:
        return f"CPU only · {ram}"
    gpu = f"{hardware.gpu_name} ({_gb(hardware.gpu_total_bytes)})"
    return f"{gpu} via {BACKEND_LABELS[hardware.gpu_backend]} · {ram}"


def _warnings(hardware: HardwareProfile) -> list[str]:
    """Problems that stop the GPU from being used well, with what to do about them."""
    notes: list[str] = []
    backend = hardware.gpu_backend
    if hardware.has_gpu_hardware and backend is GpuBackend.CPU:
        notes.append(
            f"{hardware.gpu_name} is not used: this llama-server has no GPU support. "
            "Install a Vulkan build of llama.cpp (see README)."
        )
    elif hardware.has_gpu and not _backend_fits(backend, hardware.gpu_vendor):
        notes.append(
            f"{BACKEND_LABELS[backend]} does not work with {hardware.gpu_name}. "
            "Set STRONGHOUSE_GPU_BACKEND=vulkan or install a matching llama.cpp build."
        )
    return notes


def _backend_fits(backend: GpuBackend, vendor: GpuVendor) -> bool:
    allowed = _WORKS_WITH.get(backend)
    return allowed is None or vendor in allowed


def _gb(size: int) -> str:
    return f"{size / GB:.1f} GB"
