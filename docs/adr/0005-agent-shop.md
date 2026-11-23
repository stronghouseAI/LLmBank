# 5. Agent shop: RPG-style agents that run on the local model, with approval on every action

Date: 2026-11-23 · Status: accepted

## Context
StronghouseAI has three parts and no more: model storage, the low-spec runtime layer,
and the Agent shop. phil wants agents created like RPG characters (name, class,
skills) that do tasks for him on his own computer. His answers to the design questions:
agents are workers (not companions), they may have full access to his computer but
must ask permission for every action, and teams start at two agents.

## Decision
- **Character sheet** (`core/agents.py`): name, portrait, class, personality, up to 5
  skills, and two stats that map to real settings: Creativity -> sampling temperature,
  Diligence -> maximum steps per quest. XP and levels are cosmetic for now.
- **Skill slots are limited to 5** because small local models choose tools worse as
  the tool list grows. It doubles as a game rule.
- **Skills** are OpenAI-style tools sent to llama-server's `/v1/chat/completions`
  (`--jinja` is already on). read_file, list_folder, search_files, write_file and
  run_command all pause the quest until the user allows or denies that exact call.
  take_note only writes to the agent's own journal in our database, so it does not ask.
- **Party of two** = worker + reviewer. The reviewer checks the result and either
  answers `APPROVED` or sends concrete feedback for one revision (two review rounds max).
  Both agents share the one loaded model and take turns, because swapping models is slow
  on a laptop.
- **Storage**: agents, journals and quest logs live in `agents.sqlite3`, separate from the
  library index, because the library index is disposable and agents are not.
- **Sharing**: agents export as `*.agent.json` cards (format `stronghouse-agent`, v1).
  Unknown skills are dropped on import so newer cards still load.

## Consequences
- Agents need a model that supports tool calling (Qwen3, Llama 3.1+, gpt-oss and similar).
  Very small models may produce bad tool calls; those are fed back as errors to retry.
- Approval on every action is slow for long tasks. A later "trust level" could allow
  read-only skills without asking, but only if phil decides to.
- run_command executes the approved command line with the user's own permissions;
  the approval box always shows the exact command first.
- Quests in progress do not survive an app restart; they are marked "interrupted".
