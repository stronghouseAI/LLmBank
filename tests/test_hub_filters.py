"""Discover filters: sizes from names, MoE detection, file fit, split parts, hub search."""

import httpx
import pytest

from stronghouse.core.hub_filters import (
    GIB,
    FileFit,
    ModelKind,
    SearchFilters,
    SizeClass,
    SortOrder,
    combine_split_parts,
    file_fit,
    looks_like_moe,
    parameters_from_name,
    size_class,
    size_label,
)
from stronghouse.core.models import HardwareProfile
from stronghouse.services.huggingface import HuggingFaceClient

B = 1_000_000_000

# phil's laptop: 8 GB GPU, about 16 GB RAM.
LAPTOP = HardwareProfile("RTX 4060 Laptop", 8 * GIB, 7 * GIB, 16 * GIB, 12 * GIB, 20)
NO_GPU = HardwareProfile(None, 0, 0, 16 * GIB, 12 * GIB, 8)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("unsloth/Qwen3-8B-GGUF", 8 * B),
        ("Qwen/Qwen3-0.6B-GGUF", int(0.6 * B)),
        ("unsloth/Qwen3-30B-A3B-GGUF", 30 * B),  # total, not the 3B active
        ("TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF", 56 * B),
        ("HuggingFaceTB/SmolLM2-135M-Instruct-GGUF", 135_000_000),
        ("bartowski/gemma-3-4b-it-GGUF", 4 * B),
        ("lmstudio-community/DeepSeek-V2-Lite-Chat-GGUF", None),
    ],
)
def test_parameters_from_name(name, expected):
    assert parameters_from_name(name) == expected


def test_size_classes_and_labels():
    assert size_class(None) is None
    assert size_class(int(0.6 * B)) is SizeClass.TINY
    assert size_class(8 * B) is SizeClass.SMALL
    assert size_class(14 * B) is SizeClass.MEDIUM
    assert size_class(30 * B) is SizeClass.LARGE
    assert size_class(235 * B) is SizeClass.HUGE
    assert [size_label(n) for n in (135_000_000, int(0.6 * B), 8 * B, 30 * B)] == [
        "135M", "0.6B", "8B", "30B",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("name", "architecture", "tags", "expected"),
    [
        ("unsloth/Qwen3-30B-A3B-GGUF", None, [], True),
        ("x/DeepSeek-V2-Lite-GGUF", "deepseek2", [], True),
        ("x/gpt-oss-20b-GGUF", "gpt-oss", [], True),
        ("x/OLMoE-1B-7B-GGUF", None, [], False),  # name alone is not enough here...
        ("x/OLMoE-1B-7B-GGUF", None, ["moe"], True),  # ...but the tag is
        ("x/Mixtral-8x7B-GGUF", "llama", [], True),
        ("x/Qwen3-8B-GGUF", "qwen3", [], False),
        ("x/Amoeba-7B", None, [], False),
    ],
)
def test_moe_detection(name, architecture, tags, expected):
    assert looks_like_moe(name, architecture, tags) is expected


def test_file_fit_on_phils_laptop():
    assert file_fit(5 * GIB, LAPTOP) is FileFit.GPU  # Qwen3-8B Q4_K_M
    assert file_fit(17 * GIB, LAPTOP) is FileFit.OFFLOAD  # Qwen3-30B-A3B Q4_K_M
    assert file_fit(40 * GIB, LAPTOP) is FileFit.TOO_BIG
    assert file_fit(5 * GIB, NO_GPU) is FileFit.OFFLOAD


def test_split_parts_are_combined_into_the_first_part():
    files = [
        ("big-00002-of-00002.gguf", 3),
        ("big-00001-of-00002.gguf", 5),
        ("small.gguf", 1),
    ]
    assert combine_split_parts(files) == [("big-00001-of-00002.gguf", 8), ("small.gguf", 1)]


# ----- hub search with filters -----

HUB_MODELS = [
    {"id": "unsloth/Qwen3-30B-A3B-GGUF", "downloads": 9, "gguf": {"total": 30_500_000_000,
     "architecture": "qwen3moe"}, "pipeline_tag": "text-generation"},
    {"id": "unsloth/Qwen3-8B-GGUF", "downloads": 8, "pipeline_tag": "text-generation"},
    {"id": "someone/mystery-GGUF", "downloads": 7},
]  # fmt: skip


def make_hub(refuse_expand: bool = False):
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if refuse_expand and "expand%5B%5D" in str(request.url):
            return httpx.Response(400, json={"error": "bad expand"})
        return httpx.Response(200, json=HUB_MODELS)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return HuggingFaceClient("https://hub.test", token=None, http=http), requests


def test_search_sends_sort_and_type_and_reads_gguf_summary():
    hub, requests = make_hub()
    filters = SearchFilters(kind=ModelKind.CHAT, sort=SortOrder.TRENDING)
    models = hub.search_models("qwen", filters)
    params = requests[0].url.params
    assert params["sort"] == "trendingScore" and params["pipeline_tag"] == "text-generation"
    assert "gguf" in params.get_list("expand[]")
    first = models[0]
    assert first.parameters == 30_500_000_000 and first.is_moe and first.size_label == "30B"
    assert models[1].size_label == "8B" and not models[1].is_moe


def test_size_and_moe_filters_and_fallback_without_expand():
    hub, requests = make_hub(refuse_expand=True)
    small = hub.search_models("qwen", SearchFilters(size=SizeClass.SMALL))
    assert [m.repo_id for m in small] == ["unsloth/Qwen3-8B-GGUF"]
    assert requests[0].url.params["limit"] == "200"  # fetch more when filtering here
    assert "expand[]" not in requests[1].url.params  # retried without extra fields
    moe = hub.search_models("qwen", SearchFilters(moe_only=True))
    assert [m.repo_id for m in moe] == ["unsloth/Qwen3-30B-A3B-GGUF"]
