"""Read metadata and tensor sizes from GGUF model files, without loading weights.

GGUF layout (version 2 and 3), all little-endian:
    magic "GGUF" | u32 version | u64 tensor_count | u64 kv_count
    kv pairs:     string key | u32 value_type | value
    tensor infos: string name | u32 n_dims | u64 dims[n_dims] | u32 ggml_type | u64 offset
    padding to `general.alignment` (default 32), then the tensor data.

We work out each tensor's size from the gap between data offsets rather than
from ggml type tables, so new quantization types never break the reader.
Spec: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

GGUF_MAGIC = b"GGUF"
SUPPORTED_VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32
# Arrays longer than this (tokenizer vocabularies) are skipped, only their length kept.
MAX_STORED_ARRAY_LENGTH = 1024

_SPLIT_NAME = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")

_SCALAR_FORMATS = {
    0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
    6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d",
}  # fmt: skip
_TYPE_STRING = 8
_TYPE_ARRAY = 9


class GgufFormatError(ValueError):
    """The file is not a GGUF file we can read."""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    size_bytes: int


@dataclass(frozen=True)
class GgufContents:
    metadata: dict[str, object]
    tensors: list[TensorInfo]


def read_gguf(path: Path) -> GgufContents:
    """Read a GGUF model, following split files (-00001-of-0000N) if present.

    Raises:
        GgufFormatError: if any part is missing or malformed.
    """
    first = _read_single_file(path)
    shard_paths = split_shard_paths(path)
    if len(shard_paths) <= 1:
        return first

    tensors = list(first.tensors)
    for shard in shard_paths[1:]:
        if not shard.exists():
            raise GgufFormatError(f"Missing split part: {shard.name}")
        tensors.extend(_read_single_file(shard).tensors)
    return GgufContents(metadata=first.metadata, tensors=tensors)


def split_shard_paths(path: Path) -> list[Path]:
    """All parts of a split model, given any one part. A single file returns [path]."""
    match = _SPLIT_NAME.match(path.name)
    if not match:
        return [path]
    prefix, count = match["prefix"], int(match["count"])
    return [path.with_name(f"{prefix}-{i:05d}-of-{count:05d}.gguf") for i in range(1, count + 1)]


def _read_single_file(path: Path) -> GgufContents:
    try:
        with path.open("rb") as stream:
            return _parse(stream, file_size=path.stat().st_size, source=path.name)
    except struct.error as error:
        raise GgufFormatError(f"{path.name}: file ends unexpectedly") from error


def _parse(stream: BinaryIO, file_size: int, source: str) -> GgufContents:
    if stream.read(4) != GGUF_MAGIC:
        raise GgufFormatError(f"{source}: not a GGUF file")
    (version,) = _unpack(stream, "<I")
    if version not in SUPPORTED_VERSIONS:
        raise GgufFormatError(f"{source}: unsupported GGUF version {version}")
    tensor_count, kv_count = _unpack(stream, "<QQ")

    metadata: dict[str, object] = {}
    for _ in range(kv_count):
        key = _read_string(stream)
        (value_type,) = _unpack(stream, "<I")
        metadata[key] = _read_value(stream, value_type)

    names_and_offsets: list[tuple[str, int]] = []
    for _ in range(tensor_count):
        name = _read_string(stream)
        (n_dims,) = _unpack(stream, "<I")
        stream.read(8 * n_dims)  # dimensions are not needed; sizes come from offsets
        _ggml_type, offset = _unpack(stream, "<IQ")
        names_and_offsets.append((name, offset))

    alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT))  # type: ignore[arg-type]
    data_start = _align(stream.tell(), alignment)
    data_size = file_size - data_start
    return GgufContents(
        metadata=metadata, tensors=_sizes_from_offsets(names_and_offsets, data_size)
    )


def _sizes_from_offsets(
    names_and_offsets: list[tuple[str, int]], data_size: int
) -> list[TensorInfo]:
    ordered = sorted(names_and_offsets, key=lambda item: item[1])
    tensors = []
    for index, (name, offset) in enumerate(ordered):
        end = ordered[index + 1][1] if index + 1 < len(ordered) else data_size
        if end < offset:
            raise GgufFormatError(f"Tensor {name} points past the end of the file")
        tensors.append(TensorInfo(name=name, size_bytes=end - offset))
    return tensors


def _read_value(stream: BinaryIO, value_type: int) -> object:
    if value_type in _SCALAR_FORMATS:
        return _unpack(stream, _SCALAR_FORMATS[value_type])[0]
    if value_type == _TYPE_STRING:
        return _read_string(stream)
    if value_type == _TYPE_ARRAY:
        item_type, length = _unpack(stream, "<IQ")
        if length > MAX_STORED_ARRAY_LENGTH:
            _skip_array(stream, item_type, length)
            return ArrayStub(item_type=item_type, length=length)
        return [_read_value(stream, item_type) for _ in range(length)]
    raise GgufFormatError(f"Unknown metadata value type {value_type}")


@dataclass(frozen=True)
class ArrayStub:
    """Placeholder for a long array we skipped (e.g. a 150k-entry vocabulary)."""

    item_type: int
    length: int


def _skip_array(stream: BinaryIO, item_type: int, length: int) -> None:
    if item_type in _SCALAR_FORMATS:
        stream.seek(struct.calcsize(_SCALAR_FORMATS[item_type]) * length, 1)
        return
    for _ in range(length):
        if item_type == _TYPE_STRING:
            (string_length,) = _unpack(stream, "<Q")
            stream.seek(string_length, 1)
        else:
            _read_value(stream, item_type)


def _read_string(stream: BinaryIO) -> str:
    (length,) = _unpack(stream, "<Q")
    return stream.read(length).decode("utf-8", errors="replace")


def _unpack(stream: BinaryIO, fmt: str) -> tuple:
    size = struct.calcsize(fmt)
    return struct.unpack(fmt, stream.read(size))


def _align(position: int, alignment: int) -> int:
    return position + (-position % alignment)
