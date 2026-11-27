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
- **Discover**: search GGUF models on Hugging Face and archive them. Filter by model size, type (chat, vision, embedding) and MoE, and sort by downloads, likes, trending or date. Every file is marked *fits in GPU*, *GPU + RAM* or *too big* for your computer. Split models (`-00001-of-0000N`) download all parts.
- **Run**: Auto-fit plans GPU/CPU placement (`--n-cpu-moe`, `--n-gpu-layers`, context, KV cache type), shows the exact command, loads the model and runs a speed test. Manual overrides are there for fine-tuning.
- **Big model mode**: models bigger than your RAM still run. The experts that do not fit stream from your disk (an NVMe drive helps), slowly but surely. Auto-fit shows the expected speed before you load (e.g. "about 0.6 tokens/s, so a 1000-token answer takes 28 minutes") and which quantization of the model would fit in RAM instead. See `docs/adr/0011-big-models-on-small-machines.md`.
- **Speed flags**: set threads, batch and micro-batch, or paste the llama-server command you already use in the terminal; your flags win over the app's.
- **Jobs**: queue prompts for a big model and walk away. Answers are written to Markdown files in `~/.local/share/stronghouse/jobs/` as they are generated, and a desktop notification tells you when each one is done.
- **Chat**: talk to the loaded model. Every reply has a **Copy** button, **Copy chat** puts the whole conversation on the clipboard, and **Export .txt** saves it as a text file (with the model name, date and any thinking). Chat text can be selected and copied by hand, even while a reply is still streaming. Turn on **Use internet** to let the model search the web, read pages and news feeds when it needs to: it asks you (Allow / Deny) before every single search or page, and nothing goes online while the switch is off. In the terminal the same thing is `shai chat --web` / `shai ask --web`, or `/web` inside a chat, with a y/n question before every search or page. `shai chat --files` / `shai ask --files` (or `/files` inside a chat) lets the model list folders and read files and PDFs on your computer, read only, with a y/n question before every folder or file. Models without native tool calling can still use these by writing the call as text (ADR 0016), and the terminal warns you if a model ignores its tools.
- **OpenAI-compatible endpoint**: while a model is loaded, `http://127.0.0.1:8766/v1` works with Open WebUI and other tools.
- **Agents (Agent shop)**: build your own agents from the bottom up. Give each one a name, a job and instructions, any of the 15 skills it needs (files, web and news, PDFs, desktop, calculator and clock), a Temperature and a Max steps limit. Give them tasks using the loaded model. Every action on your computer (reading, writing, searching files, running commands, going online) waits for your **Allow once** or **Deny**. Pair an agent with a reviewer for a team of two. Agents keep notes between tasks and can be exported and imported as `.agent.json` cards. See `docs/adr/0006-skill-library.md` and `0007-agents-built-from-skills.md`; the look is in `0010-studio-ui-and-error-contract.md`. PDF reading needs `poppler` and clipboard copy needs `wl-clipboard` (both one `pacman -S` away).
- **System**: GPU, RAM, CPU and engine version, plus the **GPU backend** (CUDA for NVIDIA, Vulkan or ROCm for AMD) and this machine's **model size limits**: how big a model runs fully on the GPU, how big runs from GPU + RAM, and that anything bigger still runs from disk. NVIDIA cards are read with `nvidia-smi`, AMD cards straight from the `amdgpu` driver, and `llama-server --list-devices` tells the app which backend your llama.cpp build has. See `docs/adr/0018-amd-gpus-and-machine-profiles.md`.
- **Logs**: the app, engine, jobs, agents and startup logs in one place. **Copy** or **Copy all logs** puts them on the clipboard ready to paste to Claude, with tokens and passwords hidden. `shai logs` does the same in the terminal, even when the app will not start. See `docs/adr/0012-log-window.md`.

## Terminal version (`shai`)

Everything also works from the terminal. The first command starts the app in the
background if it is not already running, so the web UI and the terminal share
the same loaded model. `uv run stronghouse` then just opens the browser to it.

```bash
shai vault ls                          # models on your disk
shai vault load qwen3-8b               # Auto-fit and load (by id or part of the name)
shai vault load deepseek --ctx 16384 --kv q8_0 --n-cpu-moe 10
shai chat "hi"                         # talk to it; --no-think for quick answers
shai ask "What is 17 times 23?"
shai ask --web "What changed in llama.cpp this week?"   # may go online, asks y/n each time
shai chat --web                        # or type /web inside chat to turn internet on/off
shai ask --files "What is in my Downloads folder?"      # may read your files, asks y/n each time
shai chat --files                      # or type /files inside chat to turn file reading on/off
shai discover qwen3 --max-gb 8 --moe   # Hugging Face search with filters
shai files Qwen/Qwen3-8B-GGUF          # which files fit your computer
shai get Qwen/Qwen3-8B-GGUF Qwen3-8B-Q4_K_M.gguf
shai agentvault new                    # build an agent: a few yes/no questions pick its skills
shai agentvault new Newsie --skills "web_search,read_news_feed,write_file" --job "..."
shai agentvault ls / show Newsie       # your agents (shai agents is the short form)
shai agentvault add-skill Newsie read_pdf / remove-skill / edit / rm
shai agentvault export Newsie / import newsie.agent.json   # share or back up
shai agentvault run Newsie "UK headlines into ~/Documents/news.md"   # y/n for every action
shai vault plan glm-4.5-air            # speed estimate and the best quantization for you
shai job add "Write a full business plan for ..." --model glm   # queue it, walk away
shai job ls / shai job show 3          # check on jobs; answers are also saved as files
shai status                            # engine, model, GPU and memory
shai logs --problems                   # warnings and errors from every log, ready to paste
shai unload / shai stop                # free the GPU / stop the background app
```

Run `shai --help` or `shai <command> --help` for every option. Inside the repo use
`uv run shai ...`; to have `shai` everywhere, install it once with
`uv tool install --editable .` from the repo folder.

## Install (Arch / Omarchy, NVIDIA laptop)

```bash
# 1. The inference engine (CUDA build for NVIDIA GPUs). From the AUR, e.g.:
yay -S llama.cpp-cuda          # AMD card on Arch? use llama.cpp-vulkan instead
# 2. uv, the Python project tool
sudo pacman -S uv
# 3. StronghouseAI
git clone https://github.com/stronghouseAI/LLmBank.git && cd LLmBank
uv sync
uv run stronghouse             # opens http://127.0.0.1:8765 in your browser
```

NVIDIA cards are found with `nvidia-smi`. Without a GPU, StronghouseAI plans for CPU only.

## Install (Pop!_OS 24.04, AMD desktop)

For an AMD Radeon card such as the RX 9060 XT. llama.cpp talks to it through
**Vulkan**, which uses the Mesa driver Pop!_OS already ships, so there is no AMD
driver to install. Use `git clone` rather than the GitHub ZIP, so `git pull`
updates it later.

**1. Check the card is ready.** RDNA4 cards (RX 9060 / 9070) need Linux 6.14 or
newer and Mesa 25.0 or newer; update Pop!_OS first if `vulkaninfo` does not list it.

```bash
sudo apt update && sudo apt full-upgrade
sudo apt install vulkan-tools
vulkaninfo --summary | grep deviceName   # should show "AMD Radeon RX 9060 XT (RADV GFX1200)"
```

**2. Build llama.cpp with Vulkan** (about 5 minutes on a 12-core CPU):

```bash
sudo apt install git curl build-essential cmake libvulkan-dev glslc spirv-headers libcurl4-openssl-dev
git clone https://github.com/ggml-org/llama.cpp ~/src/llama.cpp
cd ~/src/llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j "$(nproc)"
./build/bin/llama-server --list-devices  # should show "Vulkan0: AMD Radeon RX 9060 XT ..."
```

To update it later: `cd ~/src/llama.cpp && git pull && cmake --build build --config Release -j "$(nproc)"`.
(llama.cpp's GitHub releases also have a ready-made `ubuntu-vulkan-x64` zip if you
would rather not build; point `STRONGHOUSE_LLAMA_SERVER` at the `llama-server` inside it.)

**3. Install uv and StronghouseAI:**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # then open a new terminal
git clone https://github.com/stronghouseAI/LLmBank.git ~/LLmBank && cd ~/LLmBank
uv sync
echo 'export STRONGHOUSE_LLAMA_SERVER="$HOME/src/llama.cpp/build/bin/llama-server"' >> ~/.bashrc
source ~/.bashrc
uv run stronghouse             # opens http://127.0.0.1:8765 in your browser
```

The **System** page should now say *GPU backend: Vulkan* and show the model size
limits for this machine (with 16 GB VRAM and 32 GB RAM: about 14 GB fully on the
GPU, about 44 GB from GPU + RAM, bigger models stream from disk).
`uv run shai status` shows the same in the terminal.

**ROCm instead of Vulkan (optional).** Vulkan is the easy default and is usually
as fast for generating text. ROCm can be faster at reading long prompts. RDNA4
needs ROCm 6.4.1 or newer (see AMD's install guide for Ubuntu 24.04). Then:

```bash
cd ~/src/llama.cpp
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
  cmake -B build-rocm -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1200 -DCMAKE_BUILD_TYPE=Release
cmake --build build-rocm --config Release -j "$(nproc)"
```

and point `STRONGHOUSE_LLAMA_SERVER` at `build-rocm/bin/llama-server`. The app
notices the ROCm build by itself.

### Keeping models on extra SSDs

Mount each drive somewhere permanent first (the **Disks** app, *Edit Mount
Options*, or a line in `/etc/fstab`), for example `/mnt/ssd1` and `/mnt/ssd2`.
Then either:

- **Put the whole vault on one drive.** New downloads go there too:
  `echo 'export STRONGHOUSE_ARCHIVE_DIR=/mnt/ssd1/models' >> ~/.bashrc`
- **Use several drives at once.** Keep the vault where it is and link
  `<owner>` folders from the other drives into it, then press **Rescan** (or
  `shai vault rescan`):

  ```bash
  mkdir -p /mnt/ssd2/models/unsloth
  ln -s /mnt/ssd2/models/unsloth ~/.local/share/stronghouse/archive/unsloth
  ```

  New downloads still go to the vault folder; move finished ones across and rescan.

SATA SSDs read at about 0.5 GB/s, so a model that streams from one is slower than
from NVMe. Tell the speed forecast with `export STRONGHOUSE_DISK_GBPS=0.5`, and keep
the models you stream most on the fastest drive. The app never deletes model files.

## Configuration

All settings are environment variables (defined in one place, `src/stronghouse/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `STRONGHOUSE_ARCHIVE_DIR` | `~/.local/share/stronghouse/archive` | Where models are stored. Point it at your external drive. |
| `STRONGHOUSE_DATA_DIR` | `~/.local/share/stronghouse` | Library index and logs (`app.log`, `engine.log`, `server.log`). |
| `STRONGHOUSE_LLAMA_SERVER` | `llama-server` | Path to the llama-server binary. |
| `STRONGHOUSE_PORT` / `STRONGHOUSE_ENGINE_PORT` | `8765` / `8766` | App and engine ports. |
| `STRONGHOUSE_EXTRA_ARGS` | unset | llama-server flags added to every launch, e.g. `-b 4096 -ub 1024`. |
| `STRONGHOUSE_GPU_BACKEND` | `auto` | `cuda`, `vulkan`, `rocm` or `cpu`. `auto` asks llama-server which one it was built with. |
| `STRONGHOUSE_GPU_GBPS` / `_RAM_GBPS` / `_DISK_GBPS` | picked per machine | Memory speeds used for speed estimates (otherwise about 180-250 / 40 / 2 GB/s depending on the GPU). Raise `DISK_GBPS` for a fast NVMe, lower it for a SATA SSD. |
| `STRONGHOUSE_JOBS_DIR` | `<data dir>/jobs` | Where job answers are saved. |
| `STRONGHOUSE_NOTIFY` | `1` | Set to `0` to turn off desktop notifications for jobs. |
| `HF_TOKEN` | unset | Hugging Face token for gated models. Never logged. |

Example: `STRONGHOUSE_ARCHIVE_DIR=/run/media/$USER/models uv run stronghouse`

## Archive layout

```
<archive>/<owner>/<repo>/<file>.gguf
<archive>/<owner>/<repo>/README.md          model card
<archive>/<owner>/<repo>/stronghouse.json   source, sha256, license, engine version
```

The folder is the record; the SQLite index can be deleted and rebuilt with **Rescan**.
Already have models in another tool, or on another drive? Move or symlink them into
this layout and rescan (linked folders are followed, see *Keeping models on extra SSDs*).

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
