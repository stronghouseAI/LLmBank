-- Index of archived model files. The archive folder itself (with its
-- stronghouse.json sidecars) is the real record; this table can be rebuilt by
-- scanning the archive, so it is safe to delete.
CREATE TABLE models (
    id            INTEGER PRIMARY KEY,
    path          TEXT    NOT NULL UNIQUE,
    repo_id       TEXT    NOT NULL,
    filename      TEXT    NOT NULL,
    size_bytes    INTEGER NOT NULL,
    sha256        TEXT,
    license       TEXT,
    architecture  TEXT,
    is_moe        INTEGER NOT NULL DEFAULT 0,
    tags          TEXT    NOT NULL DEFAULT '',
    added_at      TEXT    NOT NULL
);

CREATE INDEX models_repo_id ON models (repo_id);
