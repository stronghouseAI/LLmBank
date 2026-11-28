"""Find what kind of drive a folder lives on (NVMe, SATA SSD or hard disk) on Linux.

Models bigger than RAM stream from the vault's drive, so its kind sets their speed.
The kernel says it in sysfs: the block device behind the folder, whether it spins
(queue/rotational), and for encrypted or LVM volumes the drive underneath (slaves/).
"""

from __future__ import annotations

import os
from pathlib import Path

from stronghouse.core.models import DiskKind

SYS_DEV_BLOCK = Path("/sys/dev/block")
# Encrypted (LUKS) or LVM volumes sit on another device; follow at most this many layers.
MAX_LAYERS = 4


def disk_kind(folder: Path, sys_dev_block: Path = SYS_DEV_BLOCK) -> DiskKind:
    """The kind of drive holding folder (symlinks followed). UNKNOWN if it cannot tell."""
    existing = _nearest_existing(folder.expanduser())
    if existing is None:
        return DiskKind.UNKNOWN
    device = os.stat(existing).st_dev
    block = sys_dev_block / f"{os.major(device)}:{os.minor(device)}"
    try:
        disk = _whole_disk(block.resolve(strict=True))
    except OSError:
        return DiskKind.UNKNOWN  # e.g. btrfs or a network share, which have no block device
    return classify_disk(disk.name, _read(disk / "queue" / "rotational"))


def classify_disk(name: str, rotational: str | None) -> DiskKind:
    """NVMe by name; otherwise a spinning disk is a hard disk and the rest are SATA SSDs."""
    if name.startswith("nvme"):
        return DiskKind.NVME
    if rotational == "1":
        return DiskKind.HDD
    if rotational == "0":
        return DiskKind.SATA_SSD
    return DiskKind.UNKNOWN


def _whole_disk(device: Path) -> Path:
    """From a partition or a mapped volume (dm-0) to the physical drive."""
    for _ in range(MAX_LAYERS):
        if (device / "partition").exists():
            device = device.parent
        slaves = sorted((device / "slaves").iterdir()) if (device / "slaves").is_dir() else []
        if not slaves:
            return device
        device = slaves[0].resolve(strict=True)
    return device


def _nearest_existing(folder: Path) -> Path | None:
    """The folder itself, or its closest parent that exists (a vault not created yet)."""
    for candidate in (folder, *folder.parents):
        if candidate.exists():
            return candidate
    return None


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None
