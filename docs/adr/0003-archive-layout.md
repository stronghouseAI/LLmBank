# 3. The archive folder is the source of truth; SQLite is an index

Date: 2026-11-23 · Status: accepted

## Context
The archive should outlive this app, this machine and possibly this database. It
should be copyable to an external drive and readable by other tools.

## Decision
Store files as `<archive>/<owner>/<repo>/<file>` (the Hugging Face repo id), with
the model card (`README.md`) and a `stronghouse.json` sidecar holding source, sha256,
license and the llama.cpp version used. The SQLite library is rebuilt from the folder
by "Rescan". The app never deletes model files; "Forget" only removes the index row.

## Consequences
- Other tools (llama.cpp directly, Ollama via import, etc.) can use the files as-is.
- Moving the archive to a new drive is copy + rescan.
- Tags live only in SQLite for now; if they need to survive a rebuild they should
  move into the sidecar.
