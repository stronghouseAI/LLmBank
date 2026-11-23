"""GGUF reading and model profiling, using tiny generated GGUF files."""

from pathlib import Path

import pytest

from stronghouse.core.model_profiler import profile_model
from stronghouse.services.gguf_reader import (
    MAX_STORED_ARRAY_LENGTH,
    ArrayStub,
    GgufFormatError,
    read_gguf,
    split_shard_paths,
)
from tests.gguf_builder import moe_model_spec, write_gguf


def test_reads_metadata_and_tensor_sizes(tmp_path: Path):
    metadata, tensors = moe_model_spec(blocks=2, expert_bytes=4096)
    contents = read_gguf(write_gguf(tmp_path / "m.gguf", metadata, tensors))

    assert contents.metadata["general.architecture"] == "qwen3moe"
    sizes = {t.name: t.size_bytes for t in contents.tensors}
    # Sizes come from offset gaps, so they include alignment padding (32 bytes).
    assert sizes["blk.0.ffn_up_exps.weight"] == 4096
    assert sizes["output_norm.weight"] == 32


def test_long_arrays_are_skipped_not_loaded(tmp_path: Path):
    vocab = [f"tok{i}" for i in range(MAX_STORED_ARRAY_LENGTH + 5)]
    metadata, tensors = moe_model_spec(blocks=1)
    metadata["tokenizer.ggml.tokens"] = vocab
    contents = read_gguf(write_gguf(tmp_path / "m.gguf", metadata, tensors))

    stub = contents.metadata["tokenizer.ggml.tokens"]
    assert isinstance(stub, ArrayStub)
    assert stub.length == len(vocab)
    assert contents.metadata["qwen3moe.block_count"] == 1  # parsing continued correctly


def test_rejects_non_gguf_file(tmp_path: Path):
    path = tmp_path / "fake.gguf"
    path.write_bytes(b"NOPE" + b"\0" * 64)
    with pytest.raises(GgufFormatError, match="not a GGUF"):
        read_gguf(path)


def test_truncated_file_fails_clearly(tmp_path: Path):
    path = tmp_path / "short.gguf"
    path.write_bytes(b"GGUF\x03\x00")
    with pytest.raises(GgufFormatError, match="ends unexpectedly"):
        read_gguf(path)


def test_split_model_tensors_are_merged(tmp_path: Path):
    metadata, tensors = moe_model_spec(blocks=2)
    names = list(tensors)
    first = {n: tensors[n] for n in names[: len(names) // 2]}
    second = {n: tensors[n] for n in names[len(names) // 2 :]}
    write_gguf(tmp_path / "big-00001-of-00002.gguf", {**metadata, "split.count": 2}, first)
    write_gguf(tmp_path / "big-00002-of-00002.gguf", {"split.count": 2}, second)

    contents = read_gguf(tmp_path / "big-00001-of-00002.gguf")
    assert {t.name for t in contents.tensors} == set(names)


def test_split_model_with_missing_part_fails(tmp_path: Path):
    metadata, tensors = moe_model_spec(blocks=1)
    write_gguf(tmp_path / "big-00001-of-00002.gguf", metadata, tensors)
    with pytest.raises(GgufFormatError, match="Missing split part"):
        read_gguf(tmp_path / "big-00001-of-00002.gguf")


def test_split_shard_paths_for_single_file():
    assert split_shard_paths(Path("model.gguf")) == [Path("model.gguf")]


def test_profile_separates_experts_from_shared_weights(tmp_path: Path):
    metadata, tensors = moe_model_spec(blocks=3, expert_bytes=4096, dense_bytes=512)
    contents = read_gguf(write_gguf(tmp_path / "m.gguf", metadata, tensors))
    profile = profile_model(contents.metadata, contents.tensors)

    assert profile.is_moe
    assert profile.block_count == 3
    assert profile.expert_bytes_per_layer == [3 * 4096] * 3
    assert profile.dense_bytes_per_layer == [512] * 3
    assert profile.token_embedding_bytes == 1024
    # 2 kv heads * (32 + 32 head dim) * 2 bytes * 3 blocks
    assert profile.kv_bytes_per_token_f16 == 2 * 64 * 2 * 3


def test_shared_experts_count_as_dense():
    class T:
        def __init__(self, name, size):
            self.name, self.size_bytes = name, size

    metadata = {"general.architecture": "x", "x.block_count": 1, "x.expert_count": 4}
    profile = profile_model(
        metadata, [T("blk.0.ffn_up_shexp.weight", 100), T("blk.0.ffn_up_exps.weight", 900)]
    )
    assert profile.dense_bytes_per_layer == [100]
    assert profile.expert_bytes_per_layer == [900]


def test_latent_attention_models_use_compressed_kv_size():
    metadata = {
        "general.architecture": "deepseek2",
        "deepseek2.block_count": 2,
        "deepseek2.attention.kv_lora_rank": 512,
        "deepseek2.rope.dimension_count": 64,
    }
    assert profile_model(metadata, []).kv_bytes_per_token_f16 == (512 + 64) * 2 * 2


def test_profile_requires_architecture():
    with pytest.raises(ValueError, match="architecture"):
        profile_model({}, [])
