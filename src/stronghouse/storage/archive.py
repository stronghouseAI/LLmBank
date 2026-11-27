"""On-disk archive layout.

    <archive_dir>/<owner>/<repo>/<file>.gguf
    <archive_dir>/<owner>/<repo>/README.md          (model card, if published)
    <archive_dir>/<owner>/<repo>/stronghouse.json   (our sidecar, see below)

The sidecar makes each folder self-describing: where it came from, checksums,
license, and which llama.cpp build was used. That way the archive survives
being copied to another drive or machine even without the SQLite index, which
can be rebuilt with `scan_archive`. See docs/adr/0003-archive-layout.md.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

SIDECAR_NAME = "stronghouse.json"
SIDECAR_SCHEMA_VERSION = 1
_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


class UnsafePathError(ValueError):
    """A repo id or filename would escape the archive folder."""


@dataclass
class ArchivedFile:
    sha256: str | None
    size_bytes: int
    downloaded_at: str


@dataclass
class Sidecar:
    repo_id: str
    license: str | None = None
    engine_version: str | None = None
    files: dict[str, ArchivedFile] = field(default_factory=dict)
    schema_version: int = SIDECAR_SCHEMA_VERSION


def repo_dir(archive_dir: Path, repo_id: str) -> Path:
    """Folder for a Hugging Face repo like 'unsloth/Qwen3-30B-A3B-GGUF'."""
    if not _REPO_ID.match(repo_id) or ".." in repo_id:
        raise UnsafePathError(f"Not a valid repo id: {repo_id!r}")
    return archive_dir / repo_id


def file_path(archive_dir: Path, repo_id: str, filename: str) -> Path:
    """Where a repo file lives in the archive. Filenames may include subfolders."""
    base = repo_dir(archive_dir, repo_id).resolve()
    target = (base / filename).resolve()
    if base not in target.parents:
        raise UnsafePathError(f"Filename escapes the archive: {filename!r}")
    return target


def read_sidecar(folder: Path, repo_id: str) -> Sidecar:
    """Load a folder's sidecar, or an empty one if there is none yet."""
    path = folder / SIDECAR_NAME
    if not path.exists():
        return Sidecar(repo_id=repo_id)
    raw = json.loads(path.read_text(encoding="utf-8"))
    files = {name: ArchivedFile(**info) for name, info in raw.get("files", {}).items()}
    return Sidecar(
        repo_id=raw.get("repo_id", repo_id),
        license=raw.get("license"),
        engine_version=raw.get("engine_version"),
        files=files,
        schema_version=raw.get("schema_version", SIDECAR_SCHEMA_VERSION),
    )


def write_sidecar(folder: Path, sidecar: Sidecar) -> None:
    """Write the sidecar atomically so a crash never leaves half a JSON file."""
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / (SIDECAR_NAME + ".tmp")
    temporary.write_text(json.dumps(asdict(sidecar), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(folder / SIDECAR_NAME)


@dataclass(frozen=True)
class FoundModel:
    repo_id: str
    filename: str
    path: Path


def scan_archive(archive_dir: Path) -> list[FoundModel]:
    """Find every .gguf file under <owner>/<repo>/ folders in the archive.

    Symlinked folders are followed, so models on another drive (a second SSD) can be
    linked into the archive. Each real folder is visited once, so a link loop is safe.
    """
    found = []
    if not archive_dir.exists():
        return found
    for path in sorted(_gguf_files(archive_dir)):
        relative = path.relative_to(archive_dir).parts
        if len(relative) < 3:
            continue  # not inside an <owner>/<repo>/ folder
        found.append(
            FoundModel(
                repo_id=f"{relative[0]}/{relative[1]}",
                filename="/".join(relative[2:]),
                path=path,
            )
        )
    return found


def _gguf_files(root: Path) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for folder, subfolders, names in os.walk(root, followlinks=True):
        real = os.path.realpath(folder)
        if real in seen:
            subfolders.clear()
            continue
        seen.add(real)
        files += [Path(folder) / name for name in names if name.endswith(".gguf")]
    return files
