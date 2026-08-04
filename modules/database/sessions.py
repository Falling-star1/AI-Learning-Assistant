import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from config import DATABASE_PATH
from modules.database.history import delete_chat_records
from modules.database.models import ChatSession

DEFAULT_SESSION_TITLE = "新会话"


def init_session_table(db_path: str | Path = DATABASE_PATH) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                active_kb_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at)")
        conn.commit()
    return path


def ensure_default_session(
    db_path: str | Path = DATABASE_PATH,
    active_kb_id: str = "",
) -> ChatSession:
    init_session_table(db_path)
    sessions = list_sessions(db_path=db_path, limit=1)
    if sessions:
        return sessions[0]
    return create_session(active_kb_id=active_kb_id, db_path=db_path)


def create_session(
    title: str = DEFAULT_SESSION_TITLE,
    active_kb_id: str = "",
    db_path: str | Path = DATABASE_PATH,
) -> ChatSession:
    init_session_table(db_path)
    session_id = f"session_{uuid4().hex[:12]}"
    now = datetime.now().isoformat(timespec="seconds")
    clean_title = title.strip() or DEFAULT_SESSION_TITLE
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            INSERT INTO sessions(session_id, title, active_kb_id, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, clean_title, active_kb_id, now, now),
        )
        conn.commit()
    return ChatSession(session_id, clean_title, active_kb_id, datetime.fromisoformat(now), datetime.fromisoformat(now))


def get_session(session_id: str, db_path: str | Path = DATABASE_PATH) -> ChatSession | None:
    init_session_table(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            """
            SELECT session_id, title, active_kb_id, created_at, updated_at
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
    return _session_from_row(row) if row else None


def list_sessions(
    db_path: str | Path = DATABASE_PATH,
    limit: int = 20,
) -> list[ChatSession]:
    init_session_table(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT session_id, title, active_kb_id, created_at, updated_at
            FROM sessions
            ORDER BY updated_at DESC, created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_session_from_row(row) for row in rows]


def delete_session(
    session_id: str,
    db_path: str | Path = DATABASE_PATH,
) -> bool:
    """Delete one persisted conversation and its transcript as one user-visible unit."""
    init_session_table(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()

    delete_chat_records(session_id, db_path=db_path)
    return True


def update_session_title(
    session_id: str,
    title: str,
    db_path: str | Path = DATABASE_PATH,
) -> None:
    _update_session(session_id, title=title.strip() or DEFAULT_SESSION_TITLE, db_path=db_path)


def update_session_active_kb(
    session_id: str,
    active_kb_id: str,
    db_path: str | Path = DATABASE_PATH,
) -> None:
    _update_session(session_id, active_kb_id=active_kb_id, db_path=db_path)


def touch_session(session_id: str, db_path: str | Path = DATABASE_PATH) -> None:
    _update_session(session_id, db_path=db_path)


def maybe_update_session_title(
    session_id: str,
    first_question: str,
    db_path: str | Path = DATABASE_PATH,
) -> None:
    session = get_session(session_id, db_path=db_path)
    if session is None or session.title != DEFAULT_SESSION_TITLE:
        return
    title = first_question.strip().replace("\n", " ")[:30] or DEFAULT_SESSION_TITLE
    update_session_title(session_id, title, db_path=db_path)


def _update_session(
    session_id: str,
    db_path: str | Path,
    title: str | None = None,
    active_kb_id: str | None = None,
) -> None:
    init_session_table(db_path)
    session = get_session(session_id, db_path=db_path)
    if session is None:
        return
    now = datetime.now().isoformat(timespec="seconds")
    next_title = session.title if title is None else title
    next_kb_id = session.active_kb_id if active_kb_id is None else active_kb_id
    with closing(sqlite3.connect(db_path)) as conn:
        conn.execute(
            """
            UPDATE sessions
            SET title = ?, active_kb_id = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (next_title, next_kb_id, now, session_id),
        )
        conn.commit()


def _session_from_row(row: tuple[object, ...]) -> ChatSession:
    return ChatSession(
        session_id=str(row[0]),
        title=str(row[1]),
        active_kb_id=str(row[2]),
        created_at=datetime.fromisoformat(str(row[3])),
        updated_at=datetime.fromisoformat(str(row[4])),
    )
