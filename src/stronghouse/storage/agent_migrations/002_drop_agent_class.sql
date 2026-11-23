-- Agents are now built from skills alone, with no class (ADR 0007).
-- Existing agents keep their name, skills, stats, xp and journal.
ALTER TABLE agents DROP COLUMN agent_class;
