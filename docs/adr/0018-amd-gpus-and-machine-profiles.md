# 0018: AMD GPUs (Vulkan, ROCm) and machine profiles

Date: 2026-11-27. Status: accepted.

## Context

Until now the app assumed one kind of machine: an NVIDIA GPU found with
`nvidia-smi` and a CUDA build of llama.cpp. phil's desktop has an AMD Radeon RX
9060 XT (RDNA4, 16 GB VRAM) with 32 GB DDR4 on Pop!_OS 24.04, and there is a 64 GB
mini PC too. On the desktop the app saw no GPU at all and planned for CPU only,
which wastes the card that makes big MoE models usable.

Two separate facts decide what a plan can do:

1. **Which card is in the machine and how much memory is free.** Only the OS knows
   this live: `nvidia-smi` for NVIDIA, the `amdgpu` driver's sysfs files
   (`mem_info_vram_total`, `mem_info_vram_used`) for AMD.
2. **What llama-server can drive.** That depends on how it was built (CUDA, Vulkan,
   ROCm/HIP or CPU only). An AMD card is useless to a CUDA build, and planning GPU
   layers for it would silently run on the CPU.

## Decision

1. **Hardware probe** (`services/hardware_probe.py`) reads NVIDIA first, then AMD
   from sysfs (the card with the most VRAM, so a discrete card beats an iGPU). A
   few AMD PCI ids are named (0x7590 is the RX 9060 XT); others take llama-server's
   name. No new tools or packages are needed.
2. **Backend detection** (`core/gpu_backend.py`). The app runs
   `llama-server --list-devices` once and remembers it (`EngineController.list_devices`).
   Device names say the backend: `CUDA0`, `Vulkan0`, `ROCm0`. An empty list means a
   CPU-only build, and the plan then says why the GPU is not used. If llama-server
   cannot be asked, the usual build is assumed (CUDA for NVIDIA, Vulkan otherwise).
   `STRONGHOUSE_GPU_BACKEND` overrides it.
3. **Vulkan is the default for AMD.** It works with the Mesa driver Pop!_OS ships,
   needs no ROCm install, and is usually as fast for token generation. ROCm stays
   optional and is detected the same way. The README has Pop!_OS steps for both.
4. **Same planner for every backend.** `--n-cpu-moe` and `--n-gpu-layers` mean the
   same thing on CUDA, Vulkan and ROCm. Only the memory each backend keeps for
   itself differs: 900 MiB (CUDA), 1200 MiB (Vulkan), 1100 MiB (ROCm). When
   llama-server sees more than one GPU (an iGPU next to the card), the plan pins
   the biggest with `--device`, so layers are not split onto the iGPU.
5. **Machine profile** (`core/machine_profile.py`): the backend, the largest model
   file that runs fully on the GPU and from GPU + RAM (the same limits the Discover
   page's "fits" badges use), memory speeds for the forecast, and warnings such as
   "this llama-server has no GPU support". VRAM speed is picked from a short table
   of known cards (RTX 4060 Laptop 180 GB/s, RX 9060 XT 250 GB/s effective), else
   180 GB/s. The `STRONGHOUSE_*_GBPS` settings now override instead of being the
   only source. Shown on the System page, `shai status` and `/api/hardware`.
6. **Vault on several drives.** `scan_archive` now follows symlinked folders (Python
   3.13's `rglob` does not), visiting each real folder once, so models kept on extra
   SATA SSDs can be linked into the archive.

## Consequences

- The CUDA path is unchanged: same probe, same reserve, same flags, same speeds.
- On the desktop a 30B-A3B MoE keeps many more expert blocks on the GPU than on the
  laptop, and models up to about 44 GB run from memory; bigger ones stream from disk.
- Reserves for Vulkan and ROCm are cautious estimates; the speed test and the
  engine log are the truth, and manual overrides remain.
- `--list-devices` needs a llama.cpp from 2025 or later. Older builds fall back to
  the vendor default.
- Only one GPU is planned for. Splitting a model across two cards is out of scope.
