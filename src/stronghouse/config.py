"""Application settings: the single source of truth for paths, ports and tunables.

Every value can be overridden with an environment variable prefixed STRONGHOUSE_
(for example STRONGHOUSE_ARCHIVE_DIR=/run/media/me/ext/models). Nothing else in
the codebase reads environment variables directly, except the optional
Hugging Face token (HF_TOKEN), which is a standard name other tools share.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ENV_PREFIX = "STRONGHOUSE_"
_DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "stronghouse"


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

    @property
    def database_path(self) -> Path:
        return self.data_dir / "library.sqlite3"


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
    )


def _parse_port(raw: str, name: str) -> int:
    try:
        port = int(raw)
    except ValueError as error:
        raise ValueError(f"{ENV_PREFIX}{name} must be a number, got {raw!r}") from error
    if not 1 <= port <= 65535:
        raise ValueError(f"{ENV_PREFIX}{name} must be between 1 and 65535, got {port}")
    return port
