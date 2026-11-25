# 0017: Build agents in the terminal (`shai agentvault`)

Date: 2026-11-25. Status: accepted.

## Context

Agents could only be created in the web app's Agents tab; the terminal could list
them (`shai agents`) and run them (`shai quest`). phil asked for an agent vault that
works like `shai vault` does for models, and for "a handful of yes and no
questions to narrow and channel the agents".

## Decision

- `shai agentvault` with `ls`, `new`, `skills`, `show`, `edit`, `add-skill`,
  `remove-skill`, `rm`, `export`, `import` and `run`. It is a thin client over the
  existing agent API (ADR 0009), so the terminal and the web app share one set of
  agents. `shai agents` and `shai quest` stay as short forms.
- `agentvault new` walks through: name, job and instructions, then eight yes/no
  questions (read files? change files? PDFs? internet? terminal commands? apps and
  clipboard? numbers and dates? notes between tasks?). Each yes adds only the
  skills for that need, then the person can keep or trim the list. Two more
  questions set behaviour: exact work gives temperature 0.2 (creative 0.7), and
  long tasks give 18 max steps (short 10). The questions live in
  `core/agent_builder.py`, pure and tested.
- No presets or classes (ADR 0007): every answer defaults to no, so an agent
  starts with nothing and gets only what its job needs.
- Everything can also be given as flags (`new NAME --skills ... --job ...
  --temperature ... --max-steps ...`), which skips the questions, for scripts.
- Agent names are kept unique in the terminal (case-insensitive), because
  commands find agents by name. `import --name` renames a card on the way in.
- `rm` asks y/n first (unless `--yes`). `export` never overwrites a file unless
  `--force`. Model files are never touched.
- Agents use whatever model is loaded; there is no per-agent model setting, as in
  the web app.

## Consequences

- The CLI module for agents moved out of `cli/commands.py` into
  `cli/agentvault.py`, and terminal input into `cli/prompting.py`.
- The help page section is now "AGENT VAULT" instead of "GUILD".
