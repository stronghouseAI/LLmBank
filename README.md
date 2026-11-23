# StronghouseAI

Run big open-weight language models on ordinary hardware, and keep your own
offline archive of them.

StronghouseAI is a local model studio. Its main job is **Mixture-of-Experts
(MoE) offloading**: it reads a model file, checks your GPU and RAM, and works
out how to split the model so the always-used parts sit on the GPU and the
large expert weights run from system RAM. Its second job is **archiving**:
downloads are resumable, checksum-verified, and stored with their model card,
license and engine version so they still work years from now without internet.

Its third part is the **Agent shop**: create agents like RPG characters and let
them do tasks on your computer with the local model, asking your permission for
every action.

Inference itself is done by [llama.cpp](https://github.com/ggml-org/llama.cpp)'s
`llama-server`, which StronghouseAI starts and controls.

## Features (v0.1)

- **Vault**: browse, search and tag the models on your disk; rescan any archive folder (e.g. an external drive).
- **Discover**: search GGUF models on Hugging Face and archive them. Split models (`-00001-of-0000N`) download all parts.
- **Run**: Auto-fit plans GPU/CPU placement (`--n-cpu-moe`, `--n-gpu-layers`, context, KV cache type), shows the exact command, loads the model and runs a speed test. Manual overrides are there for fine-tuning.
- **Chat**: talk to the loaded model.
- **OpenAI-compatible endpoint**: while a model is loaded, `http://127.0.0.1:8766/v1` works with Open WebUI and other tools.
- **Guild (Agent shop)**: create agents like RPG characters (name, portrait, class, personality, up to 5 skills, Creativity and Diligence stats), recruit starter agents, and send them on quests using the loaded model. Every action on your computer (reading, writing, searching files, running commands) waits for your **Allow once** or **Deny**. Pair an agent with a reviewer for a team of two. Agents keep a journal between quests and can be exported and imported as `.agent.json` cards. Agents choose 5 skills from a library of 15 (files, web and news, PDFs, desktop, calculator and clock); see `docs/adr/0006-skill-library.md`. PDF reading needs `poppler` and clipboard copy needs `wl-clipboard` (both one `pacman -S` away).
- **System**: GPU, RAM, CPU and engine version.

## Install (Arch / Omarchy)

```bash
# 1. The inference engine (CUDA build for NVIDIA GPUs). From the AUR, e.g.:
yay -S llama.cpp-cuda          # or build from source; you need the `llama-server` binary
# 2. uv, the Python project tool
sudo pacman -S uv
# 3. StronghouseAI
git clone https://github.com/stronghouseAI/LLmBank.git && cd LLmBank
uv sync
uv run stronghouse             # opens http://127.0.0.1:8765 in your browser
```

`nvidia-smi` must work for GPU detection. Without it, StronghouseAI plans for CPU only.

## Configuration

All settings are environment variables (defined in one place, `src/stronghouse/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `STRONGHOUSE_ARCHIVE_DIR` | `~/.local/share/stronghouse/archive` | Where models are stored. Point it at your external drive. |
| `STRONGHOUSE_DATA_DIR` | `~/.local/share/stronghouse` | Library index and engine log. |
| `STRONGHOUSE_LLAMA_SERVER` | `llama-server` | Path to the llama-server binary. |
| `STRONGHOUSE_PORT` / `STRONGHOUSE_ENGINE_PORT` | `8765` / `8766` | App and engine ports. |
| `HF_TOKEN` | unset | Hugging Face token for gated models. Never logged. |

Example: `STRONGHOUSE_ARCHIVE_DIR=/run/media/$USER/models uv run stronghouse`

## Archive layout

```
<archive>/<owner>/<repo>/<file>.gguf
<archive>/<owner>/<repo>/README.md          model card
<archive>/<owner>/<repo>/stronghouse.json   source, sha256, license, engine version
```

The folder is the record; the SQLite index can be deleted and rebuilt with **Rescan**.
Already have models in another tool? Move or symlink them into this layout and rescan.

## Development

```bash
uv sync                    # install, including dev tools
uv run pytest              # tests
uv run ruff check . && uv run ruff format .
uv run pre-commit install  # run the linters on every commit
```

Code layout (dependencies flow downward only):

```
src/stronghouse/
  api/        HTTP routes for the UI (thin)
  ui/static/  the web UI, plain HTML/CSS/JS
  services/   outside world: GGUF files, Hugging Face, llama-server, hardware
  storage/    SQLite index and archive folder layout
  core/       pure logic: offload planner, model profiling, launch args
  config.py   all settings
  app.py      wires everything together
```

Design decisions and their reasons are in [`docs/adr/`](docs/adr). The code
follows the project's *10 Rules for Code That Survives the Long Term*.
