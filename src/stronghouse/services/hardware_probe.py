"""Detect GPU memory (NVIDIA via nvidia-smi), system RAM and CPU threads on Linux."""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

from stronghouse.core.models import HardwareProfile

log = logging.getLogger(__name__)
MIB = 1024 * 1024
KIB = 1024
NVIDIA_SMI_TIMEOUT_SECONDS = 5

CommandRunner = Callable[[list[str]], str]


def probe_hardware(
    run_command: CommandRunner | None = None, meminfo_path: Path = Path("/proc/meminfo")
) -> HardwareProfile:
    """Return what this machine has right now. Never raises for a missing GPU."""
    runner = run_command or _run
    gpu_name, gpu_total, gpu_free = _probe_nvidia(runner)
    ram_total, ram_available = read_meminfo(meminfo_path.read_text())
    return HardwareProfile(
        gpu_name=gpu_name,
        gpu_total_bytes=gpu_total,
        gpu_free_bytes=gpu_free,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        cpu_threads=os.cpu_count() or 1,
    )


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


def read_meminfo(text: str) -> tuple[int, int]:
    """Return (MemTotal, MemAvailable) in bytes from /proc/meminfo contents."""
    values = {}
    for line in text.splitlines():
        key, _, rest = line.partition(":")
        if key in ("MemTotal", "MemAvailable"):
            values[key] = int(rest.split()[0]) * KIB
    return values.get("MemTotal", 0), values.get("MemAvailable", 0)


def _probe_nvidia(runner: CommandRunner) -> tuple[str | None, int, int]:
    if runner is _run and shutil.which("nvidia-smi") is None:
        log.info("nvidia-smi not found; assuming no NVIDIA GPU")
        return None, 0, 0
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
        return None, 0, 0
    return parse_nvidia_smi(output)


def _run(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True, timeout=NVIDIA_SMI_TIMEOUT_SECONDS
    ).stdout
