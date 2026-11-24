# 0011: Big models on small machines: stream from disk, forecast speed, run as jobs

Date: 2026-11-24. Status: accepted.

## Context

StronghouseAI's main selling point is running big MoE models (GLM-4.5-Air,
Qwen3-235B-A22B, DeepSeek) on hardware most people already own, such as phil's
RTX 4060 Laptop (8 GB VRAM, 16 GB RAM). phil's rule: "slow power is better than
no power". Speed is the accepted trade-off, as long as you can do something else
while the model works.

Before this, a model bigger than free RAM only got a warning, there was no way to
know how slow a model would be before loading it, and every prompt needed a
browser or terminal open until the answer finished. phil also finds llama-server
fast when run by hand, so the app must be able to use exactly the same flags.

## Decision

1. **Big model mode.** When the CPU part of a plan is larger than free RAM, the
   plan switches to `stream_from_disk` instead of only warning. llama.cpp already
   memory-maps the GGUF file, so the experts that do not fit in RAM are read back
   from disk when a token needs them. The GPU keeps attention, shared weights and
   the KV cache as before. `--mlock` and `--no-mmap` stop this from working, so a
   plan that streams warns when they are in the extra flags.
2. **Speed forecast** (`core/speed_estimator.py`). Generating a token reads every
   active weight once: shared weights plus `experts_used / expert_count` of the
   expert weights. Seconds per token = a fixed 8 ms overhead + GPU bytes / VRAM
   speed + RAM bytes / RAM speed + disk bytes / disk speed. The share read from
   disk is 1 - (90% of free RAM / CPU-side bytes), assuming experts are picked at
   random. Memory speeds are settings (`STRONGHOUSE_GPU_GBPS`, `_RAM_GBPS`,
   `_DISK_GBPS`, defaults 180 / 40 / 2) because no probe can measure them cheaply.
   The forecast is shown as "about"; the speed test after loading is the truth.
3. **Quantization advice** (`core/quant_advisor.py`). The quant label is read from
   the file name, the profile is rescaled to each common quantization by bits per
   weight, and the normal planner and forecast run on each. A tip names the
   highest-quality quantization that fits in RAM when the current one streams.
4. **Terminal-speed flags.** Threads, batch and micro-batch can be set, and any
   other llama-server flags can be typed or pasted (a whole terminal command is
   fine: the executable, `--model`, `--host` and `--port` are dropped). They go
   last on the command line, so they win over the app's choices.
   `STRONGHOUSE_EXTRA_ARGS` sets default flags for every launch. Threads and batch
   are left to llama.cpp unless set, because its own choice (performance cores on
   hybrid Intel CPUs) is usually right and a wrong guess costs speed.
5. **Background jobs.** A job is one prompt, optionally for a specific vault model.
   One worker thread runs jobs in order: it loads the job's model if needed (with
   the same `plan_with_settings` as the Run page), streams the answer into
   `<data dir>/jobs/NNNN-title.md` as it is written (so `tail -f` works), puts the
   model's thinking in a folded section at the end, then calls `notify-send`.
   Jobs live in their own `jobs.sqlite3`. A job that was running when the app
   stopped is marked failed with a Retry button rather than re-run, so a job that
   crashes the app cannot loop.

## Consequences

- A model far bigger than RAM now loads and runs. On phil's laptop, a 60 GB
  model forecasts well under 1 token/s, so jobs are the intended way to use it.
- Forecasts can be off by about 2x: real expert use is not random (hot experts
  stay cached, which is faster than forecast), and NVMe speed through page
  faults varies a lot between drives. Measured speeds are shown after each job.
- No new dependencies. Notifications need `notify-send` (libnotify), which
  Omarchy ships with mako; without it jobs still work, silently.
- This is inference only. Fine-tuning (training) big MoE models on 8 GB VRAM is
  out of scope.
