"""Plain data types shared across modules. No I/O lives here."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


@dataclass(frozen=True)
class ModelProfile:
    """What we need to know about a model file to plan where its weights go.

    Byte counts come from the tensor table in the GGUF file. Per-layer lists are
    indexed by transformer block number (0 = first block).
    """

    architecture: str
    block_count: int
    context_length: int
    expert_count: int
    experts_used: int
    expert_bytes_per_layer: list[int]
    dense_bytes_per_layer: list[int]
    output_bytes: int
    token_embedding_bytes: int
    kv_bytes_per_token_f16: int

    @property
    def is_moe(self) -> bool:
        return self.expert_count > 1 and sum(self.expert_bytes_per_layer) > 0

    @property
    def total_expert_bytes(self) -> int:
        return sum(self.expert_bytes_per_layer)

    @property
    def total_dense_bytes(self) -> int:
        return sum(self.dense_bytes_per_layer)

    @property
    def total_weight_bytes(self) -> int:
        return (
            self.total_expert_bytes
            + self.total_dense_bytes
            + self.output_bytes
            + self.token_embedding_bytes
        )


@dataclass(frozen=True)
class HardwareProfile:
    """Memory and compute available on this machine right now."""

    gpu_name: str | None
    gpu_total_bytes: int
    gpu_free_bytes: int
    ram_total_bytes: int
    ram_available_bytes: int
    cpu_threads: int

    @property
    def has_gpu(self) -> bool:
        return self.gpu_name is not None and self.gpu_total_bytes > 0


class KvCacheType(StrEnum):
    F16 = "f16"
    Q8_0 = "q8_0"


class OffloadStrategy(StrEnum):
    FULL_GPU = "full_gpu"
    MOE_SPLIT = "moe_split"
    PARTIAL_LAYERS = "partial_layers"
    CPU_ONLY = "cpu_only"


@dataclass(frozen=True)
class LaunchPlan:
    """Where each part of the model should live, and the settings to run it with."""

    strategy: OffloadStrategy
    n_gpu_layers: int
    n_cpu_moe: int
    context_size: int
    kv_cache_type: KvCacheType
    estimated_gpu_bytes: int
    estimated_cpu_bytes: int
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class LibraryEntry:
    """One model file kept in the local archive."""

    path: str
    repo_id: str
    filename: str
    size_bytes: int
    sha256: str | None
    license: str | None
    architecture: str | None
    is_moe: bool
    tags: list[str]
    added_at: str
    id: int | None = None
