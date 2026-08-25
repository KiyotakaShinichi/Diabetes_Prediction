import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

# Resolve packaged resources from the project directory, never from the caller's
# working directory, so the service behaves identically wherever it is launched.
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# Runtime SQLite log. Gitignored, and resolved at call time rather than bound
# into function defaults so tests can redirect it to a temporary file.
DB_PATH = PROJECT_ROOT / "data" / "inference_logs.db"


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


def log_inference(
    request_id: str,
    model_variant: str,
    model_name: str,
    probability: float,
    prediction: int,
    threshold: float,
    payload: dict[str, Any],
    db_path: Path | None = None,
) -> None:
    db_path = DB_PATH if db_path is None else db_path
    init_db(db_path)

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
                        payload_json
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        request_id,
                        model_variant,
                        model_name,
                        probability,
                        prediction,
                        threshold,
                        json.dumps(payload),
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
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_id,
                model_variant,
                model_name,
                probability,
                prediction,
                threshold,
                json.dumps(payload),
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

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, request_id, model_variant, model_name, probability,
                   prediction, threshold, payload_json, created_at
            FROM inference_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["payload"] = _decode_payload(item.pop("payload_json", None), item.get("id"))
        result.append(item)
    return result
