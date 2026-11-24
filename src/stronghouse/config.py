"""Application settings: the single source of truth for paths, ports and tunables.

Every value can be overridden with an environment variable prefixed STRONGHOUSE_
(for example STRONGHOUSE_ARCHIVE_DIR=/run/media/me/ext/models). Nothing else in
the codebase reads environment variables directly, except the optional
Hugging Face token (HF_TOKEN), which is a standard name other tools share.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from stronghouse.core.models import MemorySpeeds

ENV_PREFIX = "STRONGHOUSE_"
_DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "stronghouse"
GB = 1_000_000_000
# Effective read speeds for speed estimates, in GB/s. Defaults suit a gaming laptop
# (RTX 4060 Laptop, dual-channel DDR5, PCIe 4 NVMe); override to match yours.
_DEFAULT_GPU_GBPS = "180"
_DEFAULT_RAM_GBPS = "40"
_DEFAULT_DISK_GBPS = "2"


def _default_speeds() -> MemorySpeeds:
    return MemorySpeeds(
        gpu_bytes_per_second=float(_DEFAULT_GPU_GBPS) * GB,
        ram_bytes_per_second=float(_DEFAULT_RAM_GBPS) * GB,
        disk_bytes_per_second=float(_DEFAULT_DISK_GBPS) * GB,
    )


@dataclass(frozen=True)
class Settings:
    """Runtime configuration, built once at start-up and passed down explicitly."""

    data_dir: Path
    archive_dir: Path
    llama_server_path: str
    host: str
    port: int
    engine_port: int
    huggingface_url: str
    huggingface_token: str | None
    # llama-server flags added to every launch, e.g. "-b 4096 -ub 1024" (STRONGHOUSE_EXTRA_ARGS).
    extra_args: str = ""
    memory_speeds: MemorySpeeds = field(default_factory=_default_speeds)
    # Finished background jobs are written here as Markdown files.
    jobs_dir: Path | None = None
    # Show a desktop notification (notify-send) when a background job finishes.
    notify: bool = True

    @property
    def jobs_output_dir(self) -> Path:
        return self.jobs_dir or self.data_dir / "jobs"

    @property
    def jobs_database_path(self) -> Path:
        return self.data_dir / "jobs.sqlite3"

    @property
    def database_path(self) -> Path:
        return self.data_dir / "library.sqlite3"

    @property
    def agents_database_path(self) -> Path:
        return self.data_dir / "agents.sqlite3"


def load_settings(environ: dict[str, str] | None = None) -> Settings:
    """Build Settings from the environment, falling back to sensible defaults.

    Args:
        environ: mapping to read from; defaults to os.environ (injectable for tests).

    Raises:
        ValueError: if a numeric setting is not a valid integer.
    """
    env = dict(os.environ if environ is None else environ)

    def _get(name: str, default: str) -> str:
        return env.get(ENV_PREFIX + name, default)

    data_dir = Path(_get("DATA_DIR", str(_DEFAULT_DATA_DIR))).expanduser()
    return Settings(
        data_dir=data_dir,
        archive_dir=Path(_get("ARCHIVE_DIR", str(data_dir / "archive"))).expanduser(),
        llama_server_path=_get("LLAMA_SERVER", "llama-server"),
        host=_get("HOST", "127.0.0.1"),
        port=_parse_port(_get("PORT", "8765"), "PORT"),
        engine_port=_parse_port(_get("ENGINE_PORT", "8766"), "ENGINE_PORT"),
        huggingface_url=_get("HF_URL", "https://huggingface.co").rstrip("/"),
        huggingface_token=env.get("HF_TOKEN") or None,
        extra_args=_get("EXTRA_ARGS", ""),
        memory_speeds=MemorySpeeds(
            gpu_bytes_per_second=_parse_speed(_get("GPU_GBPS", _DEFAULT_GPU_GBPS), "GPU_GBPS"),
            ram_bytes_per_second=_parse_speed(_get("RAM_GBPS", _DEFAULT_RAM_GBPS), "RAM_GBPS"),
            disk_bytes_per_second=_parse_speed(_get("DISK_GBPS", _DEFAULT_DISK_GBPS), "DISK_GBPS"),
        ),
        jobs_dir=Path(env[ENV_PREFIX + "JOBS_DIR"]).expanduser()
        if env.get(ENV_PREFIX + "JOBS_DIR")
        else None,
        notify=_get("NOTIFY", "1").strip().lower() not in ("0", "false", "no", "off"),
    )


def _parse_speed(raw: str, name: str) -> float:
    try:
        speed = float(raw)
    except ValueError as error:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number in GB/s, got {raw!r}") from error
    if speed <= 0:
        raise ValueError(f"{ENV_PREFIX}{name} must be above 0, got {speed}")
    return speed * GB


def _parse_port(raw: str, name: str) -> int:
    try:
        port = int(raw)
    except ValueError as error:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from error
    if not 1 <= port <= 65535:
        raise ValueError(f"{ENV_PREFIX}{name} must be between 1 and 65535, got {port}")
    return port
