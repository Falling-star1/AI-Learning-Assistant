import json
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any

from config import DATABASE_PATH
from modules.database.models import ChatRecord


def init_db(db_path: str | Path = DATABASE_PATH) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                mode TEXT NOT NULL,
                provider TEXT NOT NULL,
                metadata_json TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id, id)"
        )
        conn.commit()
    return path


def save_chat_record(
    session_id: str,
    role: str,
    content: str,
    mode: str,
    provider: str,
    metadata: dict[str, Any] | None = None,
    db_path: str | Path = DATABASE_PATH,
) -> int:
    """Save one chat message; callers save user and assistant separately."""
    init_db(db_path)
    created_at = datetime.now().isoformat(timespec="seconds")
    metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
    with closing(sqlite3.connect(db_path)) as conn:
        _ensure_metadata_column(conn)
        cursor = conn.execute(
            """
            INSERT INTO chat_history(session_id, role, content, mode, provider, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, role, content, mode, provider, metadata_json, created_at),
        )
        record_id = int(cursor.lastrowid)
        conn.commit()
        return record_id


def delete_chat_records(
    session_id: str,
    db_path: str | Path = DATABASE_PATH,
) -> int:
    """Delete a session transcript so removing a chat does not leave orphaned history rows."""
    init_db(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.commit()
        return int(cursor.rowcount)


def list_chat_records(
    session_id: str | None = None,
    db_path: str | Path = DATABASE_PATH,
    limit: int = 20,
) -> list[ChatRecord]:
    path = Path(db_path)
    if not path.exists():
        return []

    with closing(sqlite3.connect(db_path)) as conn:
        if not _has_table(conn, "chat_history"):
            return []
        has_metadata = _has_column(conn, "metadata_json")

    params: tuple[object, ...]
    metadata_expr = "metadata_json" if has_metadata else "NULL AS metadata_json"
    if session_id:
        sql = f"""
            SELECT id, session_id, role, content, mode, provider, {metadata_expr}, created_at
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """
        params = (session_id, limit)
    else:
        sql = f"""
            SELECT id, session_id, role, content, mode, provider, {metadata_expr}, created_at
            FROM chat_history
            ORDER BY id DESC
            LIMIT ?
        """
        params = (limit,)

    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()

    rows = list(reversed(rows))
    return [_row_to_record(row) for row in rows]


def _ensure_metadata_column(conn: sqlite3.Connection) -> None:
    if not _has_column(conn, "metadata_json"):
        conn.execute("ALTER TABLE chat_history ADD COLUMN metadata_json TEXT")


def _has_column(conn: sqlite3.Connection, column_name: str) -> bool:
    columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(chat_history)").fetchall()}
    return column_name in columns


def _has_table(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _row_to_record(row: tuple[object, ...]) -> ChatRecord:
    metadata = json.loads(str(row[6])) if row[6] else None
    return ChatRecord(
        id=int(row[0]),
        session_id=str(row[1]),
        role=str(row[2]),
        content=str(row[3]),
        mode=str(row[4]),
        provider=str(row[5]),
        metadata=metadata,
        created_at=datetime.fromisoformat(str(row[7])),
    )
