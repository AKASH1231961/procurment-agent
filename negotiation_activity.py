"""Persist negotiation session activity for live supervisor monitoring."""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


ACTIVITY_DB_PATH = Path(__file__).with_name("negotiation_activity.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(ACTIVITY_DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS negotiation_activity (
                session_id TEXT PRIMARY KEY,
                product TEXT NOT NULL,
                seller TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                opening_offer REAL NOT NULL,
                max_price REAL NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS negotiation_activity_messages (
                message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES negotiation_activity(session_id),
                turn_number INTEGER NOT NULL,
                speaker TEXT NOT NULL,
                message_type TEXT NOT NULL,
                logged_at TEXT NOT NULL,
                message_text TEXT NOT NULL,
                UNIQUE(session_id, turn_number)
            );

            CREATE INDEX IF NOT EXISTS idx_activity_status
                ON negotiation_activity(status, updated_at);
            """
        )
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def start_session(
    *,
    session_id: str,
    product: str,
    seller: str,
    quantity: int,
    opening_offer: float,
    max_price: float,
) -> None:
    started_at = _now()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO negotiation_activity
                (session_id, product, seller, quantity, opening_offer, max_price,
                 status, started_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 'in_progress', ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                product=excluded.product,
                seller=excluded.seller,
                quantity=excluded.quantity,
                opening_offer=excluded.opening_offer,
                max_price=excluded.max_price,
                status='in_progress',
                started_at=excluded.started_at,
                updated_at=excluded.updated_at
            """,
            (
                session_id,
                product,
                seller,
                quantity,
                opening_offer,
                max_price,
                started_at,
                started_at,
            ),
        )
        connection.execute(
            "DELETE FROM negotiation_activity_messages WHERE session_id=?",
            (session_id,),
        )


def record_message(
    *,
    session_id: str,
    turn_number: int,
    speaker: str,
    message_type: str,
    message_text: str,
) -> None:
    logged_at = _now()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO negotiation_activity_messages
                (session_id, turn_number, speaker, message_type, logged_at, message_text)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id, turn_number) DO UPDATE SET
                speaker=excluded.speaker,
                message_type=excluded.message_type,
                logged_at=excluded.logged_at,
                message_text=excluded.message_text
            """,
            (session_id, turn_number, speaker, message_type, logged_at, message_text),
        )
        connection.execute(
            """
            UPDATE negotiation_activity
            SET updated_at=?, status='in_progress'
            WHERE session_id=?
            """,
            (logged_at, session_id),
        )


def finish_session(session_id: str, status: str) -> None:
    if status not in {"agreed", "rejected"}:
        raise ValueError(f"Unsupported terminal negotiation status: {status}")
    with _connect() as connection:
        connection.execute(
            """
            UPDATE negotiation_activity
            SET status=?, updated_at=?
            WHERE session_id=?
            """,
            (status, _now(), session_id),
        )


def get_active_sessions() -> list[dict[str, Any]]:
    with _connect() as connection:
        sessions = [
            dict(row)
            for row in connection.execute(
                """
                SELECT session_id, product, seller, quantity, opening_offer,
                       max_price, status, started_at, updated_at
                FROM negotiation_activity
                WHERE status='in_progress'
                ORDER BY updated_at DESC
                """
            )
        ]
        for session in sessions:
            session["messages"] = [
                {
                    "turn": row["turn_number"],
                    "speaker": row["speaker"],
                    "message_type": row["message_type"],
                    "timestamp": row["logged_at"],
                    "text": row["message_text"],
                }
                for row in connection.execute(
                    """
                    SELECT turn_number, speaker, message_type, logged_at, message_text
                    FROM negotiation_activity_messages
                    WHERE session_id=?
                    ORDER BY turn_number
                    """,
                    (session["session_id"],),
                )
            ]
        return sessions
