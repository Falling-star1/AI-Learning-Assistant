import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

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
    db_path: str | Path = DATABASE_PATH,
) -> int:
    """Save one chat message; callers save user and assistant separately."""
    init_db(db_path)
    created_at = datetime.now().isoformat(timespec="seconds")
    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.execute(
            """
            INSERT INTO chat_history(session_id, role, content, mode, provider, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, role, content, mode, provider, created_at),
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
    init_db(db_path)
    params: tuple[object, ...]
    if session_id:
        sql = """
            SELECT id, session_id, role, content, mode, provider, created_at
            FROM chat_history
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        """
        params = (session_id, limit)
    else:
        sql = """
            SELECT id, session_id, role, content, mode, provider, created_at
            FROM chat_history
            ORDER BY id DESC
            LIMIT ?
        """
        params = (limit,)

    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(sql, params).fetchall()

    rows = list(reversed(rows))
    return [_row_to_record(row) for row in rows]


def _row_to_record(row: tuple[object, ...]) -> ChatRecord:
    return ChatRecord(
        id=int(row[0]),
        session_id=str(row[1]),
        role=str(row[2]),
        content=str(row[3]),
        mode=str(row[4]),
        provider=str(row[5]),
        created_at=datetime.fromisoformat(str(row[6])),
    )
