"""Write tiny but valid GGUF files for tests. Tensor data is zero bytes."""

from __future__ import annotations

import struct
from pathlib import Path

ALIGNMENT = 32


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _value(value: object) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<I?", 7, value)
    if isinstance(value, int):
        return struct.pack("<IQ", 10, value)
    if isinstance(value, float):
        return struct.pack("<If", 6, value)
    if isinstance(value, str):
        return struct.pack("<I", 8) + _string(value)
    if isinstance(value, list):
        if value and isinstance(value[0], str):
            return struct.pack("<IIQ", 9, 8, len(value)) + b"".join(_string(v) for v in value)
        return struct.pack("<IIQ", 9, 5, len(value)) + b"".join(struct.pack("<i", v) for v in value)
    raise TypeError(type(value))


def write_gguf(path: Path, metadata: dict[str, object], tensors: dict[str, int]) -> Path:
    """Write a GGUF v3 file. `tensors` maps tensor name to its size in bytes."""
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        header += _string(key) + _value(value)

    offset = 0
    data_layout = []
    for name, size in tensors.items():
        header += _string(name) + struct.pack("<I", 1) + struct.pack("<Q", size)
        header += struct.pack("<IQ", 0, offset)
        data_layout.append(size)
        offset += size + (-size % ALIGNMENT)

    padding = b"\0" * (-len(header) % ALIGNMENT)
    with path.open("wb") as stream:
        stream.write(header + padding)
        for size in data_layout:
            stream.write(b"\0" * (size + (-size % ALIGNMENT)))
    return path


def moe_model_spec(blocks: int = 4, expert_bytes: int = 4096, dense_bytes: int = 512):
    """Metadata and tensors for a small fake MoE model."""
    metadata = {
        "general.architecture": "qwen3moe",
        "qwen3moe.block_count": blocks,
        "qwen3moe.context_length": 32768,
        "qwen3moe.embedding_length": 256,
        "qwen3moe.attention.head_count": 8,
        "qwen3moe.attention.head_count_kv": 2,
        "qwen3moe.expert_count": 8,
        "qwen3moe.expert_used_count": 2,
    }
    tensors = {"token_embd.weight": 1024, "output.weight": 1024, "output_norm.weight": 32}
    for block in range(blocks):
        tensors[f"blk.{block}.attn_q.weight"] = dense_bytes
        tensors[f"blk.{block}.ffn_gate_exps.weight"] = expert_bytes
        tensors[f"blk.{block}.ffn_up_exps.weight"] = expert_bytes
        tensors[f"blk.{block}.ffn_down_exps.weight"] = expert_bytes
    return metadata, tensors
