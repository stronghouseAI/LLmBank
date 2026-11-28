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


class GpuVendor(StrEnum):
    """Who made the graphics card."""

    NVIDIA = "nvidia"
    AMD = "amd"
    INTEL = "intel"
    OTHER = "other"
    NONE = "none"


class GpuBackend(StrEnum):
    """How llama.cpp talks to the GPU. It depends on how llama-server was built."""

    CUDA = "cuda"  # NVIDIA only
    VULKAN = "vulkan"  # any vendor; the easy choice for AMD on Pop!_OS
    ROCM = "rocm"  # AMD only (llama.cpp's HIP build)
    CPU = "cpu"  # llama-server has no GPU support, or the GPU is switched off


class DiskKind(StrEnum):
    """The drive the vault is on. It sets the speed of models that stream from disk."""

    NVME = "nvme"
    SATA_SSD = "sata_ssd"
    HDD = "hdd"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GpuReading:
    """One graphics card as the operating system reports it (nvidia-smi or sysfs)."""

    vendor: GpuVendor
    name: str | None
    total_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class EngineDevice:
    """One GPU as llama-server sees it (`llama-server --list-devices`)."""

    name: str  # e.g. "Vulkan0", "CUDA0", "ROCm0"
    description: str  # e.g. "AMD Radeon RX 9060 XT (RADV GFX1200)"
    backend: GpuBackend
    total_bytes: int
    free_bytes: int


@dataclass(frozen=True)
class HardwareProfile:
    """Memory and compute available on this machine right now."""

    gpu_name: str | None
    gpu_total_bytes: int
    gpu_free_bytes: int
    ram_total_bytes: int
    ram_available_bytes: int
    cpu_threads: int
    gpu_vendor: GpuVendor = GpuVendor.NONE
    # The backend llama-server will use for the GPU. CPU means the GPU is not used.
    gpu_backend: GpuBackend = GpuBackend.CUDA
    # llama-server device to pin (e.g. "Vulkan0"); set only when it sees several GPUs.
    gpu_device: str | None = None
    # The drive holding the vault (model files), found from the archive folder.
    archive_disk: DiskKind = DiskKind.UNKNOWN

    @property
    def has_gpu(self) -> bool:
        """True when llama-server can put weights on a GPU here."""
        return self.has_gpu_hardware and self.gpu_backend is not GpuBackend.CPU

    @property
    def has_gpu_hardware(self) -> bool:
        """True when a graphics card was found, whether or not llama-server can use it."""
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
    # llama-server --device value; None lets llama.cpp use its only (or default) GPU.
    device: str | None = None


@dataclass(frozen=True)
class MemorySpeeds:
    """Effective read speeds in bytes per second, used only for speed estimates."""

    gpu_bytes_per_second: float
    ram_bytes_per_second: float
    disk_bytes_per_second: float


@dataclass(frozen=True)
class MemorySpeedOverrides:
    """Speeds the person set by hand, in bytes per second. None means "pick for this machine"."""

    gpu_bytes_per_second: float | None = None
    ram_bytes_per_second: float | None = None
    disk_bytes_per_second: float | None = None


@dataclass(frozen=True)
class MachineProfile:
    """What this computer can run, worked out from its hardware. Shown on System.

    The limits are model *file* sizes: up to gpu_limit_bytes runs fully on the GPU,
    up to in_memory_limit_bytes runs from VRAM plus RAM, and anything bigger still
    runs by streaming the rest from disk (slowly).
    """

    summary: str
    backend: GpuBackend
    gpu_reserve_bytes: int
    gpu_limit_bytes: int
    in_memory_limit_bytes: int
    memory_speeds: MemorySpeeds
    warnings: list[str] = field(default_factory=list)


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
