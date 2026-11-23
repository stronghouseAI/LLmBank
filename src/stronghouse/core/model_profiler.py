"""Build a ModelProfile from GGUF metadata and tensor sizes. Pure logic, no file access."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Protocol

from stronghouse.core.models import ModelProfile

# Matches the routed-expert tensors that llama.cpp's --cpu-moe / --n-cpu-moe move
# to the CPU, e.g. "blk.12.ffn_down_exps.weight". Shared experts ("_shexp") are
# used for every token, so they count as dense and stay on the GPU.
_EXPERT_TENSOR = re.compile(r"^blk\.(\d+)\.ffn_\w+_exps\.")
_BLOCK_TENSOR = re.compile(r"^blk\.(\d+)\.")
_TOKEN_EMBEDDING = "token_embd.weight"
_F16_BYTES = 2


class SizedTensor(Protocol):
    name: str
    size_bytes: int


def profile_model(metadata: Mapping[str, object], tensors: Iterable[SizedTensor]) -> ModelProfile:
    """Summarise where a model's bytes are, per block, for the offload planner.

    Raises:
        ValueError: if required metadata (architecture, block count) is missing.
    """
    architecture = str(metadata.get("general.architecture", ""))
    if not architecture:
        raise ValueError("Model has no general.architecture metadata")
    block_count = _int(metadata, f"{architecture}.block_count")
    if block_count <= 0:
        raise ValueError(f"Model has no {architecture}.block_count metadata")

    expert_bytes = [0] * block_count
    dense_bytes = [0] * block_count
    output_bytes = 0
    token_embedding_bytes = 0
    for tensor in tensors:
        if expert_match := _EXPERT_TENSOR.match(tensor.name):
            expert_bytes[_block_index(expert_match, block_count)] += tensor.size_bytes
        elif block_match := _BLOCK_TENSOR.match(tensor.name):
            dense_bytes[_block_index(block_match, block_count)] += tensor.size_bytes
        elif tensor.name == _TOKEN_EMBEDDING:
            token_embedding_bytes += tensor.size_bytes
        else:
            output_bytes += tensor.size_bytes

    return ModelProfile(
        architecture=architecture,
        block_count=block_count,
        context_length=_int(metadata, f"{architecture}.context_length"),
        expert_count=_int(metadata, f"{architecture}.expert_count"),
        experts_used=_int(metadata, f"{architecture}.expert_used_count"),
        expert_bytes_per_layer=expert_bytes,
        dense_bytes_per_layer=dense_bytes,
        output_bytes=output_bytes,
        token_embedding_bytes=token_embedding_bytes,
        kv_bytes_per_token_f16=_kv_bytes_per_token(metadata, architecture, block_count),
    )


def _kv_bytes_per_token(metadata: Mapping[str, object], arch: str, block_count: int) -> int:
    """Estimate f16 KV cache bytes per token across all blocks.

    Models with multi-head latent attention (DeepSeek style) cache one compressed
    vector per token instead of full keys and values, which is much smaller.
    Sliding-window layers are not modelled, so this errs on the high side.
    """
    kv_lora_rank = _int(metadata, f"{arch}.attention.kv_lora_rank")
    if kv_lora_rank:
        rope_dims = _int(metadata, f"{arch}.rope.dimension_count")
        return (kv_lora_rank + rope_dims) * _F16_BYTES * block_count

    head_count = _first_int(metadata.get(f"{arch}.attention.head_count"))
    embedding = _int(metadata, f"{arch}.embedding_length")
    default_head_dim = embedding // head_count if head_count else 0
    key_length = _int(metadata, f"{arch}.attention.key_length") or default_head_dim
    value_length = _int(metadata, f"{arch}.attention.value_length") or default_head_dim

    kv_heads = metadata.get(f"{arch}.attention.head_count_kv", head_count)
    per_block_heads = (
        [int(h) for h in kv_heads]
        if isinstance(kv_heads, list)
        else [int(kv_heads or 0)] * block_count
    )
    return sum(heads * (key_length + value_length) * _F16_BYTES for heads in per_block_heads)


def _block_index(match: re.Match[str], block_count: int) -> int:
    index = int(match.group(1))
    if index >= block_count:
        raise ValueError(f"Tensor refers to block {index} but model has {block_count} blocks")
    return index


def _int(metadata: Mapping[str, object], key: str) -> int:
    return _first_int(metadata.get(key))


def _first_int(value: object) -> int:
    """Metadata values may be an int or a per-layer list; take the first for lists."""
    if isinstance(value, list):
        value = value[0] if value else 0
    return int(value) if isinstance(value, int | float) else 0
