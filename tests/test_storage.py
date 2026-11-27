from pathlib import Path

import pytest

from stronghouse.core.models import LibraryEntry
from stronghouse.storage import archive
from stronghouse.storage.library_db import LibraryStore


def entry(path: str = "/a/b.gguf", **changes) -> LibraryEntry:
    values = dict(
        path=path,
        repo_id="owner/repo",
        filename="b.gguf",
        size_bytes=10,
        sha256="abc",
        license="apache-2.0",
        architecture="qwen3moe",
        is_moe=True,
        tags=[],
        added_at="2026-11-23T12:00:00+00:00",
    )
    values.update(changes)
    return LibraryEntry(**values)


@pytest.fixture
def store(tmp_path: Path):
    library = LibraryStore(tmp_path / "db" / "library.sqlite3")
    yield library
    library.close()


def test_upsert_then_list(store: LibraryStore):
    saved = store.upsert(entry())
    assert saved.id is not None
    assert store.list_entries() == [saved]


def test_upsert_same_path_updates_instead_of_duplicating(store: LibraryStore):
    store.upsert(entry(size_bytes=10))
    store.upsert(entry(size_bytes=20))
    (only,) = store.list_entries()
    assert only.size_bytes == 20


def test_search_matches_repo_name_and_tags(store: LibraryStore):
    first = store.upsert(entry("/x.gguf", repo_id="unsloth/qwen"))
    store.upsert(entry("/y.gguf", repo_id="bartowski/llama"))
    store.set_tags(first.id, ["Coding", " favourite ", ""])

    assert [e.path for e in store.list_entries("qwen")] == ["/x.gguf"]
    assert [e.path for e in store.list_entries("favourite")] == ["/x.gguf"]
    assert store.get(first.id).tags == ["coding", "favourite"]


def test_remove_forgets_entry(store: LibraryStore):
    saved = store.upsert(entry())
    store.remove(saved.id)
    assert store.get(saved.id) is None


def test_migrations_run_once_when_reopened(tmp_path: Path):
    path = tmp_path / "library.sqlite3"
    LibraryStore(path).upsert(entry())
    reopened = LibraryStore(path)
    assert len(reopened.list_entries()) == 1


@pytest.mark.parametrize("repo_id", ["../etc", "owner", "owner/../x", "/abs/path", "a/b/c"])
def test_unsafe_repo_ids_are_rejected(tmp_path: Path, repo_id: str):
    with pytest.raises(archive.UnsafePathError):
        archive.repo_dir(tmp_path, repo_id)


def test_filenames_cannot_escape_archive(tmp_path: Path):
    with pytest.raises(archive.UnsafePathError):
        archive.file_path(tmp_path, "owner/repo", "../../evil.gguf")
    nested = archive.file_path(tmp_path, "owner/repo", "Q4_K_M/model.gguf")
    assert nested == (tmp_path / "owner/repo/Q4_K_M/model.gguf").resolve()


def test_sidecar_round_trip(tmp_path: Path):
    folder = tmp_path / "owner/repo"
    sidecar = archive.read_sidecar(folder, "owner/repo")
    sidecar.license = "mit"
    sidecar.files["m.gguf"] = archive.ArchivedFile("abc", 10, "2026-11-23")
    archive.write_sidecar(folder, sidecar)

    loaded = archive.read_sidecar(folder, "owner/repo")
    assert loaded.license == "mit"
    assert loaded.files["m.gguf"].sha256 == "abc"


def test_scan_finds_models_in_owner_repo_folders(tmp_path: Path):
    (tmp_path / "owner/repo/sub").mkdir(parents=True)
    (tmp_path / "owner/repo/a.gguf").write_bytes(b"")
    (tmp_path / "owner/repo/sub/b.gguf").write_bytes(b"")
    (tmp_path / "loose.gguf").write_bytes(b"")

    found = archive.scan_archive(tmp_path)
    assert [(f.repo_id, f.filename) for f in found] == [
        ("owner/repo", "a.gguf"),
        ("owner/repo", "sub/b.gguf"),
    ]


def test_scan_follows_folders_linked_from_another_drive(tmp_path: Path):
    other_drive = tmp_path / "ssd2" / "unsloth"
    (other_drive / "GLM-GGUF").mkdir(parents=True)
    (other_drive / "GLM-GGUF" / "glm.gguf").write_bytes(b"")
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "unsloth").symlink_to(other_drive)
    (other_drive / "GLM-GGUF" / "loop").symlink_to(other_drive)  # must not recurse forever

    found = archive.scan_archive(vault)
    assert [(f.repo_id, f.filename) for f in found] == [("unsloth/GLM-GGUF", "glm.gguf")]
