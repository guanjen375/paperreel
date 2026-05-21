"""SQLite-backed pipeline state.

Tables:
    projects(name PK, root, created_at, source_pdf, config_overlay, meta_json)
    stages(name, status, input_hash, started_at, finished_at, error, PK(name))
    scenes(scene_id PK, chapter_id, status, input_hash, retry_count, last_error,
           updated_at, payload_json)
    artifacts(artifact_id PK, scene_id, stage, path, sha256, media_type,
              duration_sec, provenance_json, created_at)
    errors(error_id PK, stage, scene_id, message, traceback, created_at)

The DB is the single source of truth for "what has been done"; the JSON
artefacts on disk are the recoverable payloads. Resume logic combines both:
a stage is skipped iff (a) its row says completed, (b) input_hash matches,
(c) every declared output file exists.
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .hashing import sha256_file
from .io_utils import ensure_dir

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    name TEXT PRIMARY KEY,
    root TEXT NOT NULL,
    created_at TEXT NOT NULL,
    source_pdf TEXT,
    config_overlay TEXT,
    meta_json TEXT
);
CREATE TABLE IF NOT EXISTS stages (
    name TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    input_hash TEXT,
    output_paths_json TEXT,
    started_at TEXT,
    finished_at TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS scenes (
    scene_id TEXT PRIMARY KEY,
    chapter_id TEXT,
    status TEXT NOT NULL,
    input_hash TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TEXT NOT NULL,
    payload_json TEXT
);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    scene_id TEXT,
    stage TEXT NOT NULL,
    path TEXT NOT NULL UNIQUE,
    sha256 TEXT,
    media_type TEXT,
    duration_sec REAL,
    provenance_json TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS errors (
    error_id INTEGER PRIMARY KEY AUTOINCREMENT,
    stage TEXT NOT NULL,
    scene_id TEXT,
    message TEXT NOT NULL,
    traceback TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_scene ON artifacts(scene_id);
CREATE INDEX IF NOT EXISTS idx_artifacts_stage ON artifacts(stage);
CREATE INDEX IF NOT EXISTS idx_errors_scene ON errors(scene_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StateDB:
    """Tiny wrapper around sqlite3 with auto-reconnect and helpers."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        ensure_dir(self.db_path.parent)
        self._conn: sqlite3.Connection | None = None

    # ---------- connection ----------

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(
                self.db_path,
                isolation_level=None,        # autocommit; we use explicit txns
                detect_types=sqlite3.PARSE_DECLTYPES,
                timeout=30.0,
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.executescript(SCHEMA)
            self._conn = conn
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        conn.execute("BEGIN IMMEDIATE")
        try:
            yield conn
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    # ---------- projects ----------

    def upsert_project(
        self,
        name: str,
        root: str,
        *,
        source_pdf: str | None = None,
        config_overlay: str | None = None,
        meta: dict | None = None,
    ) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO projects(name, root, created_at, source_pdf, config_overlay, meta_json)
                   VALUES(?,?,?,?,?,?)
                   ON CONFLICT(name) DO UPDATE SET
                       root=excluded.root,
                       source_pdf=COALESCE(excluded.source_pdf, projects.source_pdf),
                       config_overlay=COALESCE(excluded.config_overlay, projects.config_overlay),
                       meta_json=COALESCE(excluded.meta_json, projects.meta_json)""",
                (name, root, _now(), source_pdf, config_overlay,
                 json.dumps(meta or {}, ensure_ascii=False)),
            )

    def get_project(self, name: str) -> dict | None:
        c = self._connect()
        row = c.execute("SELECT * FROM projects WHERE name=?", (name,)).fetchone()
        return dict(row) if row else None

    # ---------- stages ----------

    def get_stage(self, name: str) -> dict | None:
        c = self._connect()
        row = c.execute("SELECT * FROM stages WHERE name=?", (name,)).fetchone()
        return dict(row) if row else None

    def start_stage(self, name: str, input_hash: str) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO stages(name, status, input_hash, started_at)
                   VALUES(?,?,?,?)
                   ON CONFLICT(name) DO UPDATE SET
                       status='running',
                       input_hash=excluded.input_hash,
                       started_at=excluded.started_at,
                       error=NULL""",
                (name, "running", input_hash, _now()),
            )

    def finish_stage(self, name: str, output_paths: list[str]) -> None:
        with self.tx() as c:
            c.execute(
                """UPDATE stages
                       SET status='completed', finished_at=?, error=NULL,
                           output_paths_json=?
                       WHERE name=?""",
                (_now(), json.dumps(output_paths), name),
            )

    def fail_stage(self, name: str, error: str) -> None:
        with self.tx() as c:
            c.execute(
                "UPDATE stages SET status='failed', finished_at=?, error=? WHERE name=?",
                (_now(), error, name),
            )

    def stage_is_done(self, name: str, input_hash: str, output_paths: list[str]) -> bool:
        row = self.get_stage(name)
        if not row or row["status"] != "completed":
            return False
        if row["input_hash"] != input_hash:
            return False
        for p in output_paths:
            if not Path(p).exists():
                return False
        return True

    # ---------- scenes ----------

    def upsert_scene(
        self,
        scene_id: str,
        chapter_id: str | None,
        status: str,
        input_hash: str,
        payload: dict | None,
        *,
        last_error: str | None = None,
        bump_retry: bool = False,
    ) -> None:
        with self.tx() as c:
            existing = c.execute(
                "SELECT retry_count FROM scenes WHERE scene_id=?", (scene_id,)
            ).fetchone()
            retry = (existing["retry_count"] if existing else 0) + (1 if bump_retry else 0)
            c.execute(
                """INSERT INTO scenes(scene_id, chapter_id, status, input_hash,
                                       retry_count, last_error, updated_at, payload_json)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(scene_id) DO UPDATE SET
                       chapter_id=COALESCE(excluded.chapter_id, scenes.chapter_id),
                       status=excluded.status,
                       input_hash=excluded.input_hash,
                       retry_count=excluded.retry_count,
                       last_error=excluded.last_error,
                       updated_at=excluded.updated_at,
                       payload_json=COALESCE(excluded.payload_json, scenes.payload_json)""",
                (scene_id, chapter_id, status, input_hash, retry, last_error, _now(),
                 json.dumps(payload, ensure_ascii=False) if payload else None),
            )

    def get_scene(self, scene_id: str) -> dict | None:
        c = self._connect()
        row = c.execute("SELECT * FROM scenes WHERE scene_id=?", (scene_id,)).fetchone()
        return dict(row) if row else None

    def list_scenes(self, *, status: str | None = None) -> list[dict]:
        c = self._connect()
        q, args = "SELECT * FROM scenes", ()
        if status is not None:
            q += " WHERE status=?"
            args = (status,)
        q += " ORDER BY scene_id"
        return [dict(r) for r in c.execute(q, args).fetchall()]

    def reset_failed_scenes(self) -> int:
        with self.tx() as c:
            cur = c.execute(
                "UPDATE scenes SET status='pending', last_error=NULL WHERE status='failed'"
            )
            return cur.rowcount

    # ---------- artifacts ----------

    def register_artifact(
        self,
        path: str | Path,
        *,
        stage: str,
        scene_id: str | None = None,
        media_type: str | None = None,
        duration_sec: float | None = None,
        provenance: dict | None = None,
        compute_sha: bool = True,
    ) -> int:
        sha = sha256_file(path) if compute_sha and Path(path).exists() else None
        with self.tx() as c:
            cur = c.execute(
                """INSERT INTO artifacts(scene_id, stage, path, sha256, media_type,
                                          duration_sec, provenance_json, created_at)
                   VALUES(?,?,?,?,?,?,?,?)
                   ON CONFLICT(path) DO UPDATE SET
                       sha256=excluded.sha256,
                       media_type=excluded.media_type,
                       duration_sec=excluded.duration_sec,
                       provenance_json=excluded.provenance_json""",
                (scene_id, stage, str(path), sha, media_type, duration_sec,
                 json.dumps(provenance or {}, ensure_ascii=False), _now()),
            )
            return cur.lastrowid

    def list_artifacts(self, *, scene_id: str | None = None, stage: str | None = None) -> list[dict]:
        c = self._connect()
        clauses, args = [], []
        if scene_id is not None:
            clauses.append("scene_id=?"); args.append(scene_id)
        if stage is not None:
            clauses.append("stage=?"); args.append(stage)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        return [dict(r) for r in c.execute(
            "SELECT * FROM artifacts" + where + " ORDER BY created_at", args).fetchall()]

    # ---------- errors ----------

    def log_error(self, stage: str, message: str, *, scene_id: str | None = None,
                  traceback: str | None = None) -> None:
        with self.tx() as c:
            c.execute(
                """INSERT INTO errors(stage, scene_id, message, traceback, created_at)
                   VALUES(?,?,?,?,?)""",
                (stage, scene_id, message, traceback, _now()),
            )

    def list_errors(self) -> list[dict]:
        c = self._connect()
        return [dict(r) for r in c.execute(
            "SELECT * FROM errors ORDER BY error_id DESC").fetchall()]

    # ---------- summary ----------

    def status_summary(self) -> dict[str, Any]:
        c = self._connect()
        stages = [dict(r) for r in c.execute(
            "SELECT name, status, input_hash, finished_at FROM stages").fetchall()]
        scene_counts: dict[str, int] = {}
        for r in c.execute(
                "SELECT status, COUNT(*) as n FROM scenes GROUP BY status").fetchall():
            scene_counts[r["status"]] = r["n"]
        artifact_counts: dict[str, int] = {}
        for r in c.execute(
                "SELECT stage, COUNT(*) as n FROM artifacts GROUP BY stage").fetchall():
            artifact_counts[r["stage"]] = r["n"]
        errors = c.execute("SELECT COUNT(*) as n FROM errors").fetchone()["n"]
        return {
            "stages": stages,
            "scene_status_counts": scene_counts,
            "artifact_counts": artifact_counts,
            "error_count": errors,
            "queried_at": _now(),
        }
