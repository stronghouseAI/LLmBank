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


class MemoryMode(StrEnum):
    """Where the CPU part of the model is read from while generating."""

    IN_MEMORY = "in_memory"  # fits in free RAM: normal speed for the split
    STREAM_FROM_DISK = "stream_from_disk"  # bigger than RAM: memory-mapped, read from disk


@dataclass(frozen=True)
class RunTuning:
    """Optional llama-server speed settings. None means "llama.cpp's own default".

    These mirror the flags people use when running llama-server by hand, so a fast
    terminal setup can be reproduced in the app. extra_args is passed through last.
    """

    threads: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    extra_args: tuple[str, ...] = ()


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
    memory_mode: MemoryMode = MemoryMode.IN_MEMORY
    tuning: RunTuning = field(default_factory=RunTuning)


@dataclass(frozen=True)
class MemorySpeeds:
    """Effective read speeds in bytes per second, used only for speed estimates."""

    gpu_bytes_per_second: float
    ram_bytes_per_second: float
    disk_bytes_per_second: float


@dataclass(frozen=True)
class SpeedEstimate:
    """A rough generation-speed forecast for a plan, before loading anything.

    Generating one token reads every active weight once, so speed is bounded by how
    fast those bytes can be read from VRAM, RAM and (when streaming) the disk.
    """

    tokens_per_second: float
    gpu_bytes_per_token: int
    ram_bytes_per_token: int
    disk_bytes_per_token: int
    seconds_per_1000_tokens: float


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


@dataclass(frozen=True)
class QuantOption:
    """How the same model would run at another quantization (a different download)."""

    label: str
    bits_per_weight: float
    estimated_file_bytes: int
    memory_mode: MemoryMode
    tokens_per_second: float
    is_current: bool


@dataclass(frozen=True)
class PlanSettings:
    """What the person asked for on the Run page (or shai flags). None means automatic."""

    context: int = 8192
    kv_cache_type: KvCacheType = KvCacheType.F16
    n_cpu_moe: int | None = None
    n_gpu_layers: int | None = None
    threads: int | None = None
    batch_size: int | None = None
    ubatch_size: int | None = None
    # None uses the app-wide default flags (STRONGHOUSE_EXTRA_ARGS); "" means none.
    extra_args: str | None = None
