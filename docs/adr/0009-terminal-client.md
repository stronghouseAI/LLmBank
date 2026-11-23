# 0009: A terminal version (`shai`) that talks to the app's API

Date: 2026-11-23. Status: accepted.

## Context

phil asked for a terminal version with the same features, driven by short
commands in the spirit of nmap: `vault ls`, `vault load <model>`,
`discover <model> <8 GB`, and so on.

## Decision

- A second entry point, `shai` (and `stronghouse <command>`), in `stronghouse/cli/`.
  `stronghouse` with no command still opens the web UI, as before.
- The terminal client is a thin front end over the app's existing HTTP API
  (`cli/client.py`). It does not open the databases or start llama-server itself.
  The app process owns the engine, downloads and quests, so a model loaded from
  the terminal keeps running after the command ends, and the web UI and terminal
  see the same state.
- If the app is not running, the first command starts it in the background
  (`stronghouse serve --no-browser`, logging to `<data dir>/server.log`, PID in
  `server.pid`). `shai stop` stops that background app, which unloads the model.
- Commands are subcommands with flags (`shai vault load qwen3-8b --ctx 16384`),
  like git or docker, so `--help` works at every level. Models can be named by
  vault id or by any unique part of the file or repo name.
- `discover --max-gb N` checks the files of the top 10 results and keeps repos
  with a file of at most N GB, showing the largest such file.
- Every agent action in `shai quest` waits for a typed `y` in the terminal, the
  same rule as the web UI (ADR 0005). Ctrl+C cancels the quest.
- No new dependencies: output uses plain ANSI colours, off when piped or when
  `NO_COLOR` is set.

## Consequences

The terminal version can never drift from the web UI, because both use the same
API. Agents are still created in the web UI; the terminal lists them and runs quests.
