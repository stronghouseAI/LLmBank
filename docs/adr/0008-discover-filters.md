# 0008: Filters on the Discover tab

Date: 2026-11-23. Status: accepted.

## Context

phil asked for filters on Discover: model size, model type and so on. Hugging
Face search returns many GGUF repos, and most are too big for an 8 GB laptop GPU.

## Decision

- **Model type** (chat, vision, embedding) and **sort order** (downloads, likes,
  trending, recently updated) are passed to the Hugging Face search, which filters
  them itself (`pipeline_tag`, `sort`).
- **Model size** (tiny under 4B, small 4-9B, medium 9-20B, large 20-40B, huge) and
  **MoE only** are judged by StronghouseAI, because Hugging Face has no such search
  filter. The search asks for the repo's GGUF summary (`expand[]=gguf`: parameter
  count and architecture). When that is missing, the size is read from the repo name
  ("Qwen3-8B", "30B-A3B", "8x7B"). If the hub ever refuses the expanded fields, the
  search is retried without them and names are used. With these filters on, up to
  200 results are fetched so enough survive.
- MoE is recognised from the llama.cpp architecture (`deepseek2`, `qwen3moe`,
  `gpt-oss`...), a `moe` tag, or the name (`A3B` active-parameter suffix, `8x7B`).
- Each file is marked **fits in GPU**, **GPU + RAM** (runs with offload) or
  **too big**, from its size and this machine's memory, leaving 900 MiB of GPU
  reserve (the planner's), 1.5 GB for context and 4 GB of RAM for the desktop.
  This is a rough guide before downloading; the offload planner gives the exact
  answer afterwards. Too-big files are hidden by default (one checkbox).
- Split models are listed once, as their first part with the total size.
- All rules live in `core/hub_filters.py` and are unit-tested.

## Consequences

Unknown sizes only show under "Any size". Name parsing can be wrong for odd names;
the hub's own parameter count wins whenever it is there.
