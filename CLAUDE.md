# StronghouseAI (LLmBank)

Local model studio: MoE-aware offloading for llama.cpp plus an offline model archive.
Owner runs Omarchy (Arch, Hyprland) with an NVIDIA laptop GPU.

- Follow the owner's "10 Rules for Code That Survives the Long Term" (summarised in README
  "Development"): one job per function, modules by responsibility (core has no I/O),
  contracts via dataclasses/type hints, settings only in `config.py`, tests for every
  behaviour change, structured logging without secrets, ADRs in `docs/adr/` for decisions,
  small commits on branches, few pinned dependencies, ruff + pytest must pass.
- Run checks: `uv run ruff check . && uv run ruff format --check . && uv run pytest`
- The app never deletes model files.
