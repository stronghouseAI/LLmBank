"""Filters for the Discover tab: model size, kind, MoE, and whether a file fits.

Pure rules only; services/huggingface.py fetches the data. Hugging Face does not
always report a parameter count, so it is also read from the repo name
("Qwen3-8B", "Qwen3-30B-A3B", "Mixtral-8x7B"). See docs/adr/0008-discover-filters.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from stronghouse.core.models import HardwareProfile
from stronghouse.core.offload_planner import gpu_reserve_bytes

GIB = 1024**3
BILLION = 1_000_000_000

# Room left for the KV cache and compute buffers when judging whether a file fits.
CONTEXT_ALLOWANCE_BYTES = int(1.5 * GIB)
# RAM the desktop, browser and the app itself keep using while a model runs.
SYSTEM_RAM_ALLOWANCE_BYTES = 4 * GIB

# llama.cpp architecture names that are Mixture-of-Experts models.
MOE_ARCHITECTURES = frozenset(
    {
        "deepseek2",
        "deepseek3",
        "qwen2moe",
        "qwen3moe",
        "qwen3vlmoe",
        "gpt-oss",
        "olmoe",
        "granitemoe",
        "llama4",
        "dbrx",
        "arctic",
        "glm4moe",
        "ernie4_5-moe",
        "hunyuan-moe",
        "bailingmoe",
        "grovemoe",
    }
)
MOE_TAGS = frozenset({"moe", "mixture-of-experts"})


class SizeClass(StrEnum):
    ANY = "any"
    TINY = "tiny"  # under 4B parameters
    SMALL = "small"  # 4B to 9B
    MEDIUM = "medium"  # 9B to 20B
    LARGE = "large"  # 20B to 40B
    HUGE = "huge"  # over 40B


# Upper bound in billions of parameters, checked in order.
_SIZE_LIMITS = [
    (SizeClass.TINY, 4.0),
    (SizeClass.SMALL, 9.5),
    (SizeClass.MEDIUM, 20.5),
    (SizeClass.LARGE, 40.5),
    (SizeClass.HUGE, float("inf")),
]


class ModelKind(StrEnum):
    """Maps to Hugging Face pipeline tags."""

    ANY = "any"
    CHAT = "text-generation"
    VISION = "image-text-to-text"
    EMBEDDING = "feature-extraction"


class SortOrder(StrEnum):
    DOWNLOADS = "downloads"
    LIKES = "likes"
    TRENDING = "trendingScore"
    UPDATED = "lastModified"


class FileFit(StrEnum):
    GPU = "gpu"  # the whole file fits in graphics memory
    OFFLOAD = "offload"  # fits when part of it runs from system RAM
    TOO_BIG = "too_big"  # larger than graphics memory and RAM together


@dataclass(frozen=True)
class SearchFilters:
    size: SizeClass = SizeClass.ANY
    kind: ModelKind = ModelKind.ANY
    moe_only: bool = False
    sort: SortOrder = SortOrder.DOWNLOADS

    @property
    def filters_after_fetch(self) -> bool:
        """Size and MoE are judged here, not by Hugging Face, so fetch more results."""
        return self.size is not SizeClass.ANY or self.moe_only


# "8B", "0.6B", "135M", but not the active-parameter part of "30B-A3B".
_PARAMS = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)([BM])(?![A-Za-z])", re.IGNORECASE)
_EXPERT_PARAMS = re.compile(r"(?<![A-Za-z0-9.])(\d+)x(\d+(?:\.\d+)?)B(?![A-Za-z])", re.IGNORECASE)
_ACTIVE_PARAMS = re.compile(r"(?<![A-Za-z0-9])A\d+(?:\.\d+)?B(?![A-Za-z])")


def parameters_from_name(name: str) -> int | None:
    """Parameter count written in a repo name, or None if it has none."""
    model_name = name.rsplit("/", 1)[-1]
    experts = _EXPERT_PARAMS.search(model_name)
    if experts:  # Mixtral-8x7B: roughly experts x size (shared layers make it a bit less)
        return int(int(experts.group(1)) * float(experts.group(2)) * BILLION)
    match = _PARAMS.search(model_name)
    if not match:
        return None
    scale = BILLION if match.group(2).upper() == "B" else 1_000_000
    return int(float(match.group(1)) * scale)


def size_class(parameters: int | None) -> SizeClass | None:
    if parameters is None:
        return None
    billions = parameters / BILLION
    for size, limit in _SIZE_LIMITS:
        if billions < limit:
            return size
    return SizeClass.HUGE


def size_label(parameters: int | None) -> str:
    """Short human label: "8B", "0.6B", "135M", or "" if unknown."""
    if parameters is None:
        return ""
    if parameters < BILLION // 2:
        return f"{round(parameters / 1_000_000)}M"
    billions = parameters / BILLION
    if billions >= 10:
        return f"{round(billions)}B"
    return f"{billions:.1f}".removesuffix(".0") + "B"


def looks_like_moe(name: str, architecture: str | None, tags: list[str]) -> bool:
    if architecture and (architecture in MOE_ARCHITECTURES or "moe" in architecture):
        return True
    if MOE_TAGS & {tag.lower() for tag in tags}:
        return True
    model_name = name.rsplit("/", 1)[-1]
    return bool(
        _ACTIVE_PARAMS.search(model_name)
        or _EXPERT_PARAMS.search(model_name)
        or re.search(r"(?<![a-z])moe(?![a-z])", model_name, re.IGNORECASE)
    )


def matches(filters: SearchFilters, size: SizeClass | None, is_moe: bool) -> bool:
    """Whether a model passes the filters judged after fetching. Unknown sizes pass only "any"."""
    if filters.moe_only and not is_moe:
        return False
    return filters.size is SizeClass.ANY or size is filters.size


def file_fit(size_bytes: int, hardware: HardwareProfile) -> FileFit:
    """Rough answer to "will this file run here?", before downloading it.

    The offload planner gives the exact answer once the file is on disk.
    """
    gpu_limit, in_memory_limit = fit_limits(hardware)
    if hardware.has_gpu and size_bytes <= gpu_limit:
        return FileFit.GPU
    if size_bytes <= in_memory_limit:
        return FileFit.OFFLOAD
    return FileFit.TOO_BIG


def fit_limits(hardware: HardwareProfile) -> tuple[int, int]:
    """Biggest model file that runs (fully on the GPU, from GPU memory plus RAM).

    Uses total memory, not free, so the answer does not change with what is open.
    """
    gpu_limit = 0
    if hardware.has_gpu:
        reserve = gpu_reserve_bytes(hardware) + CONTEXT_ALLOWANCE_BYTES
        gpu_limit = max(0, hardware.gpu_total_bytes - reserve)
    ram_limit = max(0, hardware.ram_total_bytes - SYSTEM_RAM_ALLOWANCE_BYTES)
    return gpu_limit, gpu_limit + ram_limit


_SPLIT_PART = re.compile(r"^(?P<prefix>.+)-(?P<index>\d{5})-of-(?P<count>\d{5})\.gguf$")


def combine_split_parts(files: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Show split models (-00001-of-0000N) as their first part with the total size.

    Downloading the first part fetches every part, so the others are not listed.
    """
    totals: dict[str, int] = {}
    order: list[str] = []
    for filename, size in files:
        match = _SPLIT_PART.match(filename)
        key = filename
        if match:
            count = match.group("count")
            key = f"{match.group('prefix')}-00001-of-{count}.gguf"
        if key not in totals:
            totals[key] = 0
            order.append(key)
        totals[key] += size
    return [(key, totals[key]) for key in order]
