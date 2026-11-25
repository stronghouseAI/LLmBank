# 15. File access in terminal chat

Date: 2026-11-25 · Status: accepted

## Context
phil asked `shai chat` to "list the contents of my Download folder" and the model answered that
it cannot see the computer, which was true: chat only passed text. Agents already have file
skills that work on phil's laptop, and the app's rule is that every action on the computer asks
first (ADR 0006). ADR 0014 gave chat the internet the same way.

## Decision
- `shai chat --files` and `shai ask --files` turn file access on (off by default); `/files`,
  `/files on`, `/files off` switch it inside a chat. It works alongside `--web`.
- Only the **read-only** skills are offered: `list_folder`, `read_file`, `search_files`,
  `read_pdf` (`core/chat_files.py`). Writing, moving and running commands stay with agents,
  where a task and a reviewer give them context. Chat answers questions about files.
- `POST /api/chat` with `files: true` adds those tools and a short system note (paths may start
  with `~`). Each call the model makes is shown in plain words ("List the folder ~/Downloads")
  with `Allow once? [y/N]`; Enter or Ctrl+D means no. Only after a yes does the terminal call
  `POST /api/chat/files`, which runs the agents' `SkillExecutor` and refuses any other skill.
- A tool the model asks for that is not switched on is never run and never asked about.
- Up to 10 tool steps per reply when files are on (listing then opening a file is common), 6
  for internet only. After that the tools are taken away and the model must answer.

## Consequences
- Needs a model whose chat template supports tools, as for the internet.
- The Chat page in the browser does not have a file switch yet; it can reuse the same endpoint.
- The server binds to 127.0.0.1 by default, so the file endpoint is only reachable from the
  laptop itself.
