-- Agents are user-created and cannot be rebuilt from anything else, so they
-- live in their own database file, separate from the disposable library index.
CREATE TABLE agents (
    id           INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    agent_class  TEXT    NOT NULL,
    portrait     TEXT    NOT NULL DEFAULT '',
    personality  TEXT    NOT NULL DEFAULT '',
    skills       TEXT    NOT NULL DEFAULT '[]',  -- JSON list of skill names
    creativity   INTEGER NOT NULL,
    diligence    INTEGER NOT NULL,
    xp           INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE journal_notes (
    id          INTEGER PRIMARY KEY,
    agent_id    INTEGER NOT NULL REFERENCES agents (id) ON DELETE CASCADE,
    note        TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);

CREATE TABLE quests (
    id          INTEGER PRIMARY KEY,
    task        TEXT    NOT NULL,
    agent_ids   TEXT    NOT NULL,             -- JSON list: [worker] or [worker, reviewer]
    state       TEXT    NOT NULL,
    steps       TEXT    NOT NULL DEFAULT '[]', -- JSON list of {kind, agent_name, text}
    result      TEXT,
    created_at  TEXT    NOT NULL
);
