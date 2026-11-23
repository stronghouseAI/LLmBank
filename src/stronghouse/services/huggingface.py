"""Talk to the Hugging Face Hub: search GGUF models, list files, download with resume.

Only public HTTP endpoints are used, so no extra SDK dependency is needed. An
optional HF_TOKEN unlocks gated repos; it is sent as a header and never logged.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)

SEARCH_LIMIT = 30
CHUNK_BYTES = 4 * 1024 * 1024
HASH_CHUNK_BYTES = 16 * 1024 * 1024
REQUEST_TIMEOUT = httpx.Timeout(30.0, read=120.0)

ProgressCallback = Callable[[int, int], None]  # (bytes_done, bytes_total)


class DownloadError(RuntimeError):
    """A download failed or the file did not match its published checksum."""


@dataclass(frozen=True)
class RemoteModel:
    repo_id: str
    downloads: int
    likes: int
    last_modified: str | None
    tags: list[str]


@dataclass(frozen=True)
class RemoteFile:
    filename: str
    size_bytes: int
    sha256: str | None


class HuggingFaceClient:
    def __init__(self, base_url: str, token: str | None, http: httpx.Client | None = None) -> None:
        headers = {"User-Agent": "stronghouse/0.1"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._base_url = base_url
        self._http = http or httpx.Client(
            headers=headers, timeout=REQUEST_TIMEOUT, follow_redirects=True
        )
        if http is not None:
            self._http.headers.update(headers)

    def search_models(self, query: str, limit: int = SEARCH_LIMIT) -> list[RemoteModel]:
        """Search GGUF repos, most downloaded first."""
        response = self._http.get(
            f"{self._base_url}/api/models",
            params={"search": query, "filter": "gguf", "sort": "downloads", "limit": limit},
        )
        response.raise_for_status()
        return [
            RemoteModel(
                repo_id=item["id"],
                downloads=item.get("downloads", 0),
                likes=item.get("likes", 0),
                last_modified=item.get("lastModified"),
                tags=item.get("tags", []),
            )
            for item in response.json()
        ]

    def list_gguf_files(self, repo_id: str) -> list[RemoteFile]:
        """All .gguf files in a repo (including subfolders) with size and sha256."""
        response = self._http.get(
            f"{self._base_url}/api/models/{repo_id}/tree/main", params={"recursive": "true"}
        )
        response.raise_for_status()
        files = []
        for item in response.json():
            if item.get("type") != "file" or not item["path"].endswith(".gguf"):
                continue
            lfs = item.get("lfs") or {}
            files.append(
                RemoteFile(
                    filename=item["path"],
                    size_bytes=lfs.get("size", item.get("size", 0)),
                    sha256=lfs.get("oid"),
                )
            )
        return files

    def model_license(self, repo_id: str) -> str | None:
        response = self._http.get(f"{self._base_url}/api/models/{repo_id}")
        response.raise_for_status()
        info = response.json()
        card_license = (info.get("cardData") or {}).get("license")
        if card_license:
            return str(card_license)
        for tag in info.get("tags", []):
            if tag.startswith("license:"):
                return tag.split(":", 1)[1]
        return None

    def fetch_model_card(self, repo_id: str) -> str | None:
        response = self._http.get(self._file_url(repo_id, "README.md"))
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        response.raise_for_status()
        return response.text

    def download(
        self,
        repo_id: str,
        remote: RemoteFile,
        destination: Path,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        """Download to destination, resuming a previous partial download if present.

        Returns the sha256 of the finished file.

        Raises:
            DownloadError: on HTTP failure or checksum mismatch. The partial file is
                kept after a network failure (to resume) but deleted on a bad checksum.
        """
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size == remote.size_bytes:
            digest = sha256_of(destination)
            if not remote.sha256 or digest == remote.sha256:
                return digest  # already archived and intact; nothing to download
        partial = destination.with_name(destination.name + ".part")
        done = partial.stat().st_size if partial.exists() else 0
        total = remote.size_bytes

        if done < total or total == 0:
            self._stream_to(
                partial, self._file_url(repo_id, remote.filename), done, total, on_progress
            )

        digest = sha256_of(partial)
        if remote.sha256 and digest != remote.sha256:
            partial.unlink(missing_ok=True)
            raise DownloadError(
                f"Checksum mismatch for {remote.filename}: expected {remote.sha256}, got {digest}"
            )
        partial.replace(destination)
        return digest

    def _stream_to(
        self, partial: Path, url: str, done: int, total: int, on_progress: ProgressCallback | None
    ) -> None:
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            with self._http.stream("GET", url, headers=headers) as response:
                if response.status_code == httpx.codes.OK and done:
                    log.info("server ignored resume request; restarting", extra={"url": url})
                    done = 0
                elif response.status_code not in (httpx.codes.OK, httpx.codes.PARTIAL_CONTENT):
                    raise DownloadError(f"HTTP {response.status_code} downloading {url}")
                with partial.open("ab" if done else "wb") as output:
                    for chunk in response.iter_bytes(CHUNK_BYTES):
                        output.write(chunk)
                        done += len(chunk)
                        if on_progress:
                            on_progress(done, total)
        except httpx.HTTPError as error:
            raise DownloadError(f"Network error downloading {url}: {error}") from error

    def _file_url(self, repo_id: str, filename: str) -> str:
        return f"{self._base_url}/{repo_id}/resolve/main/{quote(filename)}"


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()
