"""A fingerprint of the installed code, so `shai` can spot an app left running from an
older install (it would quietly ignore newer options such as `--files`)."""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path

_PACKAGE = Path(__file__).resolve().parent.parent
_SOURCE_SUFFIXES = {".py", ".js", ".html", ".css"}


@cache
def build_id() -> str:
    """A short hash of the package's source files; it changes with every install that
    changes the code."""
    digest = hashlib.sha256()
    for path in sorted(_PACKAGE.rglob("*")):
        if path.suffix in _SOURCE_SUFFIXES and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(_PACKAGE)).encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()[:12]
