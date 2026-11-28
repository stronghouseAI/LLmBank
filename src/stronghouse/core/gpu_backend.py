"""Decide which GPU llama-server will use and through which backend (CUDA, Vulkan, ROCm).

The operating system tells us what card is in the machine and how much memory is
free (nvidia-smi for NVIDIA, sysfs for AMD). llama-server tells us what it was
built with (`llama-server --list-devices`). Both matter: an AMD card is useless
to a CUDA-only build, so a plan that assumes it would be wrong.
See docs/adr/0018-amd-gpus-and-machine-profiles.md.
"""

from __future__ import annotations

import re

from stronghouse.core.models import (
    EngineDevice,
    GpuBackend,
    GpuReading,
    GpuVendor,
    HardwareProfile,
)

MIB = 1024 * 1024

# Prefix of llama.cpp's device names ("Vulkan0", "CUDA0", "ROCm0") to backend.
_DEVICE_PREFIXES = {"CUDA": GpuBackend.CUDA, "Vulkan": GpuBackend.VULKAN, "ROCm": GpuBackend.ROCM}
# "  Vulkan0: AMD Radeon RX 9060 XT (RADV GFX1200) (16304 MiB, 15800 MiB free)"
_DEVICE_LINE = re.compile(
    r"^\s*(?P<name>(?P<prefix>[A-Za-z]+)\d+):\s+(?P<description>.+?)\s+"
    r"\((?P<total>\d+) MiB, (?P<free>\d+) MiB free\)\s*$"
)
_DEVICE_LIST_HEADER = "Available devices"

# What to assume when llama-server cannot be asked (not installed yet, or too old).
_DEFAULT_BACKEND = {
    GpuVendor.NVIDIA: GpuBackend.CUDA,
    GpuVendor.AMD: GpuBackend.VULKAN,
    GpuVendor.INTEL: GpuBackend.VULKAN,
    GpuVendor.OTHER: GpuBackend.VULKAN,
}

_VENDOR_WORDS = (
    ("nvidia", GpuVendor.NVIDIA),
    ("geforce", GpuVendor.NVIDIA),
    ("amd", GpuVendor.AMD),
    ("radeon", GpuVendor.AMD),
    ("intel", GpuVendor.INTEL),
)


def parse_engine_devices(output: str) -> list[EngineDevice] | None:
    """Parse `llama-server --list-devices`.

    Returns None when the output is not a device list (old llama-server, or it failed),
    and an empty list for a CPU-only build, which lists no devices.
    """
    if _DEVICE_LIST_HEADER not in output:
        return None
    devices = []
    for line in output.splitlines():
        match = _DEVICE_LINE.match(line)
        if match is None or match.group("prefix") not in _DEVICE_PREFIXES:
            continue
        devices.append(
            EngineDevice(
                name=match.group("name"),
                description=match.group("description"),
                backend=_DEVICE_PREFIXES[match.group("prefix")],
                total_bytes=int(match.group("total")) * MIB,
                free_bytes=int(match.group("free")) * MIB,
            )
        )
    return devices


def vendor_from_name(name: str) -> GpuVendor:
    """Guess the card maker from a device description such as "AMD Radeon RX 9060 XT"."""
    lowered = name.lower()
    for word, vendor in _VENDOR_WORDS:
        if word in lowered:
            return vendor
    return GpuVendor.OTHER


def combine_hardware(
    gpu: GpuReading | None,
    engine_devices: list[EngineDevice] | None,
    ram_total_bytes: int,
    ram_available_bytes: int,
    cpu_threads: int,
    forced_backend: GpuBackend | None = None,
) -> HardwareProfile:
    """Put the operating system's view and llama-server's view into one profile.

    Args:
        gpu: the card the OS reports, with live free memory; None if none was found.
        engine_devices: what llama-server lists; None if it could not be asked.
        forced_backend: STRONGHOUSE_GPU_BACKEND, when the person set one.
    """
    chosen = _largest(engine_devices or [])
    if gpu is None and chosen is not None:
        # No vendor tool reported the card (e.g. Intel), so use the engine's numbers.
        vendor = vendor_from_name(chosen.description)
        gpu = GpuReading(vendor, chosen.description, chosen.total_bytes, chosen.free_bytes)
    if gpu is None:
        return HardwareProfile(
            gpu_name=None,
            gpu_total_bytes=0,
            gpu_free_bytes=0,
            ram_total_bytes=ram_total_bytes,
            ram_available_bytes=ram_available_bytes,
            cpu_threads=cpu_threads,
            gpu_vendor=GpuVendor.NONE,
            gpu_backend=GpuBackend.CPU,
        )
    name = gpu.name or (chosen.description if chosen else None) or _unnamed(gpu.vendor)
    several = engine_devices is not None and len(engine_devices) > 1
    return HardwareProfile(
        gpu_name=name,
        gpu_total_bytes=gpu.total_bytes,
        gpu_free_bytes=gpu.free_bytes,
        ram_total_bytes=ram_total_bytes,
        ram_available_bytes=ram_available_bytes,
        cpu_threads=cpu_threads,
        gpu_vendor=gpu.vendor,
        gpu_backend=resolve_backend(gpu.vendor, engine_devices, forced_backend),
        gpu_device=chosen.name if several and chosen else None,
    )


def resolve_backend(
    vendor: GpuVendor,
    engine_devices: list[EngineDevice] | None,
    forced_backend: GpuBackend | None = None,
) -> GpuBackend:
    """The backend llama-server will use for this card.

    A setting wins. Otherwise llama-server's own device list decides; if it lists no
    GPU, the build is CPU-only. If llama-server could not be asked, assume the usual
    build for the vendor (CUDA for NVIDIA, Vulkan for everyone else).
    """
    if forced_backend is not None:
        return forced_backend
    if engine_devices is not None:
        chosen = _largest(engine_devices)
        return chosen.backend if chosen else GpuBackend.CPU
    return _DEFAULT_BACKEND.get(vendor, GpuBackend.CPU)


def _largest(devices: list[EngineDevice]) -> EngineDevice | None:
    """The device with the most memory: the discrete card, not an integrated one."""
    return max(devices, key=lambda device: device.total_bytes, default=None)


def _unnamed(vendor: GpuVendor) -> str:
    makers = {GpuVendor.NVIDIA: "NVIDIA", GpuVendor.AMD: "AMD", GpuVendor.INTEL: "Intel"}
    return f"{makers[vendor]} GPU" if vendor in makers else "GPU"
