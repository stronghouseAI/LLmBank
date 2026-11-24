-- Background jobs. The answers themselves live in Markdown files (output_path),
-- so this table can be deleted without losing any model output.
CREATE TABLE jobs (
    id                 INTEGER PRIMARY KEY,
    title              TEXT    NOT NULL,
    prompt             TEXT    NOT NULL,
    model_id           INTEGER,               -- NULL: use whatever model is loaded
    plan_settings      TEXT    NOT NULL,      -- JSON of PlanSettings
    think              INTEGER NOT NULL,
    max_tokens         INTEGER,
    state              TEXT    NOT NULL,
    created_at         TEXT    NOT NULL,
    started_at         TEXT,
    finished_at        TEXT,
    output_path        TEXT,
    generated_tokens   INTEGER NOT NULL DEFAULT 0,
    tokens_per_second  REAL,
    error              TEXT
);
CREATE INDEX jobs_by_state ON jobs (state, id);
