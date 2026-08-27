import hashlib
import json
import logging
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

# Resolve packaged resources from the project directory, never from the caller's
# working directory, so the service behaves identically wherever it is launched.
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# Runtime SQLite log. Gitignored, and resolved at call time rather than bound
# into function defaults so tests can redirect it to a temporary file.
DB_PATH = PROJECT_ROOT / "data" / "inference_logs.db"

#: Column added after the original schema shipped. Every migration here must be
#: additive and idempotent: existing databases are opened in place, old rows keep
#: NULL, and nothing is ever dropped or rewritten.
ASSIGNMENT_COLUMN = "assignment_key_hash"


def hash_assignment_key(assignment_key: str) -> str:
    """One-way digest of the A/B assignment key.

    This identifies an *experiment assignment*, not a user. The public UI sends a
    random per-session value and the API defaults to "anonymous", so nothing here
    is an authenticated identity - but the raw key is still never stored, so a
    stored row cannot be matched back against a value a client still holds.

    SHA-256 is deterministic, which is the whole point: the same assignment key
    must always produce the same digest, or stability of an assignment could not
    be verified after the fact.
    """
    return hashlib.sha256(assignment_key.encode("utf-8")).hexdigest()


def backend_name() -> str:
    """Which store is in use, with nothing sensitive attached.

    Deliberately returns only the engine. Host, user, password, DSN and file path
    are all configuration and none of them belong on a dashboard; the purpose is
    to explain why a dashboard is empty, not to describe the deployment.
    """
    return "PostgreSQL" if _use_postgres() else "SQLite"


def _get_database_url() -> str:
    """Resolve DATABASE_URL from env var first, then Streamlit secrets."""
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url
    try:
        import streamlit as st
    except ImportError:
        # Streamlit is optional for the API and the training scripts.
        return url

    try:
        url = st.secrets.get("DATABASE_URL", "").strip()
    except Exception as exc:  # noqa: BLE001 - see below
        # Streamlit raises StreamlitSecretsFileNotFoundError when no secrets.toml
        # exists, and that class is not importable from any public, stable API.
        # This is a genuine external boundary: "no secrets configured" must fall
        # back to local SQLite rather than break the caller. Only the exception
        # type is logged - never the value, which would be a database URL.
        logger.debug("No Streamlit secrets available (%s); using local SQLite.", type(exc).__name__)
    return url


def _decode_payload(raw: Any, row_id: Any = None) -> dict[str, Any]:
    """Decode one stored payload, degrading to {} for an unusable row.

    The log listing powers an admin analytics view over historical rows, so a
    single malformed or NULL payload must not fail the whole request. The catch
    is narrow: json.loads raises JSONDecodeError (a ValueError) for bad JSON and
    TypeError for a non-string, and a decoded non-object is rejected explicitly.
    Only the row id and exception type are logged, never the payload contents.
    """
    try:
        decoded = json.loads(raw)
    except (ValueError, TypeError) as exc:
        logger.warning("Skipping unreadable payload for row %s (%s).", row_id, type(exc).__name__)
        return {}
    if not isinstance(decoded, dict):
        logger.warning("Skipping non-object payload for row %s.", row_id)
        return {}
    return decoded


def _use_postgres() -> bool:
    url = _get_database_url()
    return url.startswith(("postgresql://", "postgres://"))


def init_db(db_path: Path | None = None) -> None:
    db_path = DB_PATH if db_path is None else db_path

    if _use_postgres():
        import psycopg

        with psycopg.connect(_get_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS inference_logs (
                        id BIGSERIAL PRIMARY KEY,
                        request_id TEXT NOT NULL,
                        model_variant TEXT NOT NULL,
                        model_name TEXT NOT NULL,
                        probability DOUBLE PRECISION NOT NULL,
                        prediction INTEGER NOT NULL,
                        threshold DOUBLE PRECISION NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_inference_logs_created_at
                    ON inference_logs(created_at DESC)
                    """
                )
                # Additive migration. Existing deployments gain the column with
                # NULL for every historical row; nothing is rewritten.
                cur.execute(
                    f"ALTER TABLE inference_logs "
                    f"ADD COLUMN IF NOT EXISTS {ASSIGNMENT_COLUMN} TEXT"
                )
            conn.commit()
        return

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS inference_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                model_variant TEXT NOT NULL,
                model_name TEXT NOT NULL,
                probability REAL NOT NULL,
                prediction INTEGER NOT NULL,
                threshold REAL NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_inference_logs_created_at
            ON inference_logs(created_at DESC)
            """
        )
        # SQLite's ALTER TABLE has no IF NOT EXISTS, so the column list decides.
        # Re-running this is a no-op, which is what makes init_db safe to call on
        # every read and write the way the rest of this module does.
        existing = {row[1] for row in conn.execute("PRAGMA table_info(inference_logs)")}
        if ASSIGNMENT_COLUMN not in existing:
            conn.execute(f"ALTER TABLE inference_logs ADD COLUMN {ASSIGNMENT_COLUMN} TEXT")


def log_inference(
    request_id: str,
    model_variant: str,
    model_name: str,
    probability: float,
    prediction: int,
    threshold: float,
    payload: dict[str, Any],
    db_path: Path | None = None,
    assignment_key: str | None = None,
) -> None:
    """Record one served inference.

    ``assignment_key`` is the value the A/B router bucketed on. Only its digest
    is stored, and omitting it leaves the column NULL, which is exactly what
    every row written before this column existed looks like.
    """
    db_path = DB_PATH if db_path is None else db_path
    init_db(db_path)
    assignment_hash = hash_assignment_key(assignment_key) if assignment_key else None

    if _use_postgres():
        import psycopg

        with psycopg.connect(_get_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO inference_logs (
                        request_id,
                        model_variant,
                        model_name,
                        probability,
                        prediction,
                        threshold,
                        payload_json,
                        assignment_key_hash
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        request_id,
                        model_variant,
                        model_name,
                        probability,
                        prediction,
                        threshold,
                        json.dumps(payload),
                        assignment_hash,
                    ),
                )
            conn.commit()
        return

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO inference_logs (
                request_id,
                model_variant,
                model_name,
                probability,
                prediction,
                threshold,
                payload_json,
                assignment_key_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                model_variant,
                model_name,
                probability,
                prediction,
                threshold,
                json.dumps(payload),
                assignment_hash,
            ),
        )


def fetch_recent_logs(limit: int = 100, db_path: Path | None = None) -> list[dict[str, Any]]:
    db_path = DB_PATH if db_path is None else db_path
    init_db(db_path)

    if _use_postgres():
        import psycopg

        with psycopg.connect(_get_database_url()) as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, request_id, model_variant, model_name, probability,
                       prediction, threshold, payload_json, created_at
                FROM inference_logs
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

        result: list[dict[str, Any]] = []
        for row in rows:
            item = {
                "id": row[0],
                "request_id": row[1],
                "model_variant": row[2],
                "model_name": row[3],
                "probability": row[4],
                "prediction": row[5],
                "threshold": row[6],
                "payload_json": row[7],
                "created_at": str(row[8]),
            }
            item["payload"] = _decode_payload(item.pop("payload_json", None), item.get("id"))
            result.append(item)
        return result

    with sqlite3.connect(db_path) as sqlite_conn:
        sqlite_conn.row_factory = sqlite3.Row
        sqlite_rows = sqlite_conn.execute(
            """
            SELECT id, request_id, model_variant, model_name, probability,
                   prediction, threshold, payload_json, created_at
            FROM inference_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    decoded: list[dict[str, Any]] = []
    for row in sqlite_rows:
        item = dict(row)
        item["payload"] = _decode_payload(item.pop("payload_json", None), item.get("id"))
        decoded.append(item)
    return decoded


#: Columns every read returns, in a fixed order so both backends agree.
_SELECT_COLUMNS = (
    "id, request_id, model_variant, model_name, probability, "
    "prediction, threshold, payload_json, created_at, assignment_key_hash"
)


def _window_start(hours: int | None) -> str | None:
    """UTC cutoff for a rolling window, formatted the way rows are stored."""
    if not hours:
        return None
    return (datetime.now(UTC) - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S")


def _row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["created_at"] = str(item.get("created_at"))
    item["payload"] = _decode_payload(item.pop("payload_json", None), item.get("id"))
    return item


def fetch_logs(
    limit: int = 100,
    db_path: Path | None = None,
    *,
    within_hours: int | None = None,
    model_variant: str | None = None,
    prediction: int | None = None,
) -> list[dict[str, Any]]:
    """Recent inferences, filtered in SQL rather than in pandas.

    Every filter is applied by the database so a dashboard narrowing to one
    variant or one day does not first pull the whole table into memory. Filters
    compose, and omitting all of them is exactly ``fetch_recent_logs``.

    Values are always bound as parameters; only fixed column names are ever
    interpolated into the statement.
    """
    db_path = DB_PATH if db_path is None else db_path
    init_db(db_path)

    clauses: list[str] = []
    values: list[Any] = []
    cutoff = _window_start(within_hours)
    if cutoff is not None:
        clauses.append("created_at >= ?")
        values.append(cutoff)
    if model_variant:
        clauses.append("model_variant = ?")
        values.append(model_variant)
    if prediction is not None:
        clauses.append("prediction = ?")
        values.append(int(prediction))

    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    # Suppression rationale: the only interpolated pieces are _SELECT_COLUMNS and
    # the fixed clause strings assembled directly above - both are literals in
    # this module. Every value, including the limit, is bound as a parameter and
    # never formatted into the statement.
    statement = (
        f"SELECT {_SELECT_COLUMNS} FROM inference_logs{where} ORDER BY id DESC LIMIT ?"  # noqa: S608
    )
    values.append(limit)

    if _use_postgres():
        import psycopg

        with psycopg.connect(_get_database_url()) as conn, conn.cursor() as cur:
            cur.execute(statement.replace("?", "%s"), tuple(values))
            # cursor.description is None for a statement that returns no result
            # set. A SELECT always populates it, but reading it unguarded would
            # turn any future change here into a TypeError rather than an empty
            # listing.
            description = cur.description or []
            columns = [column[0] for column in description]
            fetched = [dict(zip(columns, record, strict=True)) for record in cur.fetchall()]
        return [_row_to_dict(row) for row in fetched]

    with sqlite3.connect(db_path) as sqlite_conn:
        sqlite_conn.row_factory = sqlite3.Row
        sqlite_rows = sqlite_conn.execute(statement, tuple(values)).fetchall()

    return [_row_to_dict(dict(row)) for row in sqlite_rows]
