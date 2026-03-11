import json
import os
import sqlite3
from pathlib import Path
from typing import Any


DB_PATH = Path("data/inference_logs.db")


def _get_database_url() -> str:
    """Resolve DATABASE_URL from env var first, then Streamlit secrets."""
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url
    try:
        import streamlit as st
        url = st.secrets.get("DATABASE_URL", "").strip()
    except Exception:
        pass
    return url


def _use_postgres() -> bool:
    url = _get_database_url()
    return url.startswith("postgresql://") or url.startswith("postgres://")


def init_db(db_path: Path = DB_PATH) -> None:
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
    db_path: Path = DB_PATH,
) -> None:
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


def fetch_recent_logs(limit: int = 100, db_path: Path = DB_PATH) -> list[dict[str, Any]]:
    init_db(db_path)

    if _use_postgres():
        import psycopg

        with psycopg.connect(_get_database_url()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, request_id, model_variant, model_name, probability, prediction, threshold, payload_json, created_at
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
            try:
                item["payload"] = json.loads(item.pop("payload_json"))
            except Exception:
                item["payload"] = {}
                item.pop("payload_json", None)
            result.append(item)
        return result

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, request_id, model_variant, model_name, probability, prediction, threshold, payload_json, created_at
            FROM inference_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        try:
            item["payload"] = json.loads(item.pop("payload_json"))
        except Exception:
            item["payload"] = {}
            item.pop("payload_json", None)
        result.append(item)
    return result
