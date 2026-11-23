"""Hugging Face client and download manager, against a fake hub (no network)."""

import hashlib
import time
from pathlib import Path

import httpx
import pytest

from stronghouse.services.download_manager import DownloadManager, JobState
from stronghouse.services.huggingface import DownloadError, HuggingFaceClient, RemoteFile
from stronghouse.services.library_indexer import LibraryIndexer
from stronghouse.storage import archive
from stronghouse.storage.library_db import LibraryStore
from tests.gguf_builder import moe_model_spec, write_gguf

BASE = "https://hub.test"


class FakeHub:
    """Serves a repo with given files, honouring Range requests like the real hub."""

    def __init__(self, files: dict[str, bytes], license_name: str = "apache-2.0"):
        self.files = files
        self.license = license_name
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/models":
            return httpx.Response(200, json=[{"id": "owner/repo", "downloads": 5, "likes": 1}])
        if path == "/api/models/owner/repo/tree/main":
            return httpx.Response(
                200,
                json=[
                    {
                        "type": "file",
                        "path": name,
                        "lfs": {"oid": hashlib.sha256(data).hexdigest(), "size": len(data)},
                    }
                    for name, data in self.files.items()
                ]
                + [{"type": "file", "path": "config.json", "size": 2}],
            )
        if path == "/api/models/owner/repo":
            return httpx.Response(200, json={"cardData": {"license": self.license}})
        if path == "/owner/repo/resolve/main/README.md":
            return httpx.Response(200, text="# Model card")
        prefix = "/owner/repo/resolve/main/"
        if path.startswith(prefix) and path[len(prefix) :] in self.files:
            data = self.files[path[len(prefix) :]]
            if "range" in request.headers:
                start = int(request.headers["range"].split("=")[1].rstrip("-"))
                return httpx.Response(206, content=data[start:])
            return httpx.Response(200, content=data)
        return httpx.Response(404)

    def client(self) -> HuggingFaceClient:
        http = httpx.Client(transport=httpx.MockTransport(self.handler))
        return HuggingFaceClient(BASE, token="hf_secret", http=http)


def test_search_and_list_only_gguf_files():
    hub = FakeHub({"m.gguf": b"abc"})
    client = hub.client()
    assert client.search_models("qwen")[0].repo_id == "owner/repo"
    assert [f.filename for f in client.list_gguf_files("owner/repo")] == ["m.gguf"]
    assert hub.requests[0].headers["authorization"] == "Bearer hf_secret"


def test_download_resumes_from_partial_file(tmp_path: Path):
    data = b"x" * 1000
    hub = FakeHub({"m.gguf": data})
    remote = hub.client().list_gguf_files("owner/repo")[0]
    (tmp_path / "m.gguf.part").write_bytes(data[:400])

    sha = hub.client().download("owner/repo", remote, tmp_path / "m.gguf")

    assert (tmp_path / "m.gguf").read_bytes() == data
    assert sha == hashlib.sha256(data).hexdigest()
    assert hub.requests[-1].headers["range"] == "bytes=400-"


def test_checksum_mismatch_deletes_file(tmp_path: Path):
    hub = FakeHub({"m.gguf": b"real"})
    bad = RemoteFile(filename="m.gguf", size_bytes=4, sha256="0" * 64)
    with pytest.raises(DownloadError, match="Checksum mismatch"):
        hub.client().download("owner/repo", bad, tmp_path / "m.gguf")
    assert not (tmp_path / "m.gguf").exists()
    assert not (tmp_path / "m.gguf.part").exists()


def wait_for(manager: DownloadManager, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = manager.jobs()[0]
        if job.state in (JobState.DONE, JobState.FAILED):
            return job
        time.sleep(0.02)
    raise AssertionError("download did not finish")


def make_manager(tmp_path: Path, hub: FakeHub):
    store = LibraryStore(tmp_path / "library.sqlite3")
    archive_dir = tmp_path / "archive"
    indexer = LibraryIndexer(store, archive_dir)
    manager = DownloadManager(hub.client(), indexer, archive_dir, lambda: "version: 9999 (test)")
    return manager, store, archive_dir


def test_download_manager_archives_split_model_and_indexes_it(tmp_path: Path):
    metadata, tensors = moe_model_spec(blocks=2)
    names = list(tensors)
    first = write_gguf(
        tmp_path / "1.gguf", metadata, {n: tensors[n] for n in names[: len(names) // 2]}
    )
    second = write_gguf(tmp_path / "2.gguf", {}, {n: tensors[n] for n in names[len(names) // 2 :]})
    hub = FakeHub(
        {
            "q4/big-00001-of-00002.gguf": first.read_bytes(),
            "q4/big-00002-of-00002.gguf": second.read_bytes(),
        }
    )
    manager, store, archive_dir = make_manager(tmp_path, hub)

    manager.start("owner/repo", "q4/big-00001-of-00002.gguf")
    job = wait_for(manager)

    assert job.state is JobState.DONE, job.error
    repo_folder = archive_dir / "owner/repo"
    assert (repo_folder / "q4/big-00002-of-00002.gguf").exists()
    assert (repo_folder / "README.md").read_text() == "# Model card"
    sidecar = archive.read_sidecar(repo_folder, "owner/repo")
    assert sidecar.license == "apache-2.0"
    assert sidecar.engine_version == "version: 9999 (test)"
    assert set(sidecar.files) == {"q4/big-00001-of-00002.gguf", "q4/big-00002-of-00002.gguf"}

    (entry,) = store.list_entries()
    assert entry.is_moe and entry.architecture == "qwen3moe"
    assert entry.size_bytes == first.stat().st_size + second.stat().st_size


def test_download_manager_reports_missing_file(tmp_path: Path):
    manager, _, _ = make_manager(tmp_path, FakeHub({"m.gguf": b"abc"}))
    manager.start("owner/repo", "other.gguf")
    job = wait_for(manager)
    assert job.state is JobState.FAILED
    assert "other.gguf" in job.error


def test_download_manager_rejects_unsafe_names(tmp_path: Path):
    manager, _, _ = make_manager(tmp_path, FakeHub({}))
    with pytest.raises(archive.UnsafePathError):
        manager.start("owner/repo", "../../x.gguf")


def test_rescan_rebuilds_index_from_archive(tmp_path: Path):
    metadata, tensors = moe_model_spec(blocks=1)
    folder = tmp_path / "archive/owner/repo"
    folder.mkdir(parents=True)
    write_gguf(folder / "m.gguf", metadata, tensors)
    store = LibraryStore(tmp_path / "library.sqlite3")

    assert LibraryIndexer(store, tmp_path / "archive").rescan() == 1
    assert store.list_entries()[0].repo_id == "owner/repo"
