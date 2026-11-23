# 2. Plan GPU/CPU placement from the GGUF tensor table

Date: 2026-11-23 · Status: accepted

## Context
For MoE models only a few experts run per token. Keeping attention, norms, shared
experts and the KV cache on the GPU while routed experts sit in system RAM gives most
of the GPU speed with a fraction of the VRAM. Picking `--n-cpu-moe` by hand is trial
and error.

## Decision
Read every tensor's size from the GGUF header (no weights loaded), classify tensors
as routed experts (`blk.N.ffn_*_exps.*`, the same set llama.cpp's `--cpu-moe` moves),
per-block dense weights, output weights, or token embeddings, then fill the GPU in
this order:
1. Whole model on GPU if it fits.
2. MoE: all shared weights + KV cache on GPU (halving context down to 4096 if needed),
   then expert blocks from the last block backwards, because `--n-cpu-moe N` keeps
   the experts of the *first* N blocks on the CPU.
3. Otherwise whole layers, last blocks first (how `--n-gpu-layers` works).
4. CPU only.
A fixed 900 MiB is reserved for CUDA context and compute buffers.

## Consequences
- Plans are estimates. Real usage varies by llama.cpp build and batch size, so the UI
  offers a speed test and manual overrides.
- KV cache size is estimated from attention metadata; sliding-window layers are not
  modelled, so long-context estimates err high (safe side).
- Tensor sizes come from offset gaps, so new quantization types never break the reader.
