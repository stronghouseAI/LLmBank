# 1. Wrap llama.cpp instead of writing an inference engine

Date: 2026-11-23 · Status: accepted

## Context
StronghouseAI's goal is running large models, especially Mixture-of-Experts models,
on moderate hardware. The techniques that make this possible (expert offload with
`--cpu-moe` / `--n-cpu-moe` / `-ot`, quantization, partial GPU offload, quantized KV
cache) already exist in llama.cpp, which is fast, actively developed, runs on CUDA,
Vulkan, ROCm and CPU, and defines the GGUF file format most open models ship in.

## Decision
Run llama.cpp's `llama-server` as a child process and talk to it over HTTP. Our
value is in choosing the right settings (the offload planner), managing the archive,
and the UI. All flag names live in `core/launch_args.py`.

## Consequences
- We get new llama.cpp speedups for free by updating the binary.
- Users must install llama.cpp (documented in the README).
- If a flag is renamed upstream, only `launch_args.py` changes.
- Forks such as ik_llama.cpp can be used by pointing `STRONGHOUSE_LLAMA_SERVER` at them,
  as long as they accept the same flags.
