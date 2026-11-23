# 0007: Agents are built from skills, with no classes or starters

Date: 2026-11-23. Status: accepted. Changes parts of 0005 and 0006.

## Context

The Guild shipped with ready-made starter agents (Mira, Quill, Forge, Herald,
Warden), a class for every agent (scholar, scout, builder and so on) and a limit
of 5 skills each. In the first live tests phil kept picking a starter that lacked
the skill the task needed. phil asked to drop the "recruiter types" and build
agents from the bottom up: every agent can do every task, and you add only the
skills you want on that agent.

## Decision

- No starter agents and no classes. An agent is its name, portrait, job and
  personality text, skills and stats.
- Any agent can equip any of the skills; there is no hard limit. The UI suggests
  staying at 5 or fewer (`RECOMMENDED_MAX_SKILLS`), because small local models get
  confused by long tool lists, but it does not block more.
- The free-text "Personality" field becomes "Job and personality" and goes into the
  system prompt, replacing the class description.
- Migration `002_drop_agent_class.sql` drops the column. Existing agents (including
  recruited starters) keep their skills, stats, xp and journal.
- Agent cards move to version 2 without `agent_class`. Version 1 cards still
  import; their class is ignored.

## Consequences

The character creator is simpler and matches how phil wants to work. The cost
is that a new user starts with an empty Guild; the empty-state text says what to
do. Too many skills on a small model is now a soft warning instead of a rule.
