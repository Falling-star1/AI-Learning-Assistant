import sqlite3
import shutil
from contextlib import closing
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from config import DATABASE_PATH, VECTOR_DB_DIR
from modules.database.models import KnowledgeBase, KnowledgeBaseFile

DEFAULT_KB_NAME = "默认知识库"


def init_knowledge_tables(db_path: str | Path = DATABASE_PATH) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                kb_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                store_path TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_base_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kb_id TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                chunk_count INTEGER NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(kb_id) REFERENCES knowledge_bases(kb_id)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_kb_files_kb_id ON knowledge_base_files(kb_id, id)"
        )
        conn.commit()
    return path


def get_knowledge_base_store_path(kb_id: str) -> Path:
    return VECTOR_DB_DIR / kb_id / "rag_store.json"


def ensure_default_knowledge_base(db_path: str | Path = DATABASE_PATH) -> KnowledgeBase:
    init_knowledge_tables(db_path)
    active = get_active_knowledge_base(db_path=db_path)
    if active is not None:
        return active

    bases = list_knowledge_bases(db_path=db_path)
    if bases:
        set_active_knowledge_base(bases[0].kb_id, db_path=db_path)
        return get_active_knowledge_base(db_path=db_path)

    return create_knowledge_base(DEFAULT_KB_NAME, is_active=True, db_path=db_path)


def create_knowledge_base(
    name: str,
    db_path: str | Path = DATABASE_PATH,
    is_active: bool = False,
) -> KnowledgeBase:
    init_knowledge_tables(db_path)
    clean_name = name.strip() or DEFAULT_KB_NAME
    kb_id = f"kb_{uuid4().hex[:12]}"
    now = datetime.now().isoformat(timespec="seconds")
    store_path = str(get_knowledge_base_store_path(kb_id))

    with closing(sqlite3.connect(db_path)) as conn:
        if is_active:
            conn.execute("UPDATE knowledge_bases SET is_active = 0")
        conn.execute(
            """
            INSERT INTO knowledge_bases(kb_id, name, store_path, is_active, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (kb_id, clean_name, store_path, 1 if is_active else 0, now, now),
        )
        conn.commit()

    return KnowledgeBase(kb_id, clean_name, store_path, is_active, datetime.fromisoformat(now), datetime.fromisoformat(now))


def list_knowledge_bases(db_path: str | Path = DATABASE_PATH) -> list[KnowledgeBase]:
    init_knowledge_tables(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT kb_id, name, store_path, is_active, created_at, updated_at
            FROM knowledge_bases
            ORDER BY is_active DESC, updated_at DESC, created_at DESC
            """
        ).fetchall()
    return [_kb_from_row(row) for row in rows]


def get_active_knowledge_base(db_path: str | Path = DATABASE_PATH) -> KnowledgeBase | None:
    init_knowledge_tables(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            """
            SELECT kb_id, name, store_path, is_active, created_at, updated_at
            FROM knowledge_bases
            WHERE is_active = 1
            ORDER BY updated_at DESC
            LIMIT 1
            """
        ).fetchone()
    return _kb_from_row(row) if row else None


def set_active_knowledge_base(kb_id: str, db_path: str | Path = DATABASE_PATH) -> None:
    init_knowledge_tables(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    with closing(sqlite3.connect(db_path)) as conn:
        existing = conn.execute(
            "SELECT 1 FROM knowledge_bases WHERE kb_id = ?",
            (kb_id,),
        ).fetchone()
        if existing is None:
            raise ValueError(f"Knowledge base not found: {kb_id}")
        conn.execute("UPDATE knowledge_bases SET is_active = 0")
        conn.execute(
            "UPDATE knowledge_bases SET is_active = 1, updated_at = ? WHERE kb_id = ?",
            (now, kb_id),
        )
        conn.commit()



def delete_knowledge_base(
    kb_id: str,
    db_path: str | Path = DATABASE_PATH,
) -> bool:
    """Remove a knowledge base and activate a fallback so the sidebar never points at a deleted store."""
    init_knowledge_tables(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT is_active, store_path FROM knowledge_bases WHERE kb_id = ?",
            (kb_id,),
        ).fetchone()
        if row is None:
            return False

        was_active = bool(row[0])
        store_path = Path(str(row[1]))
        file_rows = conn.execute(
            "SELECT storage_path FROM knowledge_base_files WHERE kb_id = ?",
            (kb_id,),
        ).fetchall()
        storage_paths = [Path(str(file_row[0])) for file_row in file_rows]
        now = datetime.now().isoformat(timespec="seconds")
        conn.execute("DELETE FROM knowledge_base_files WHERE kb_id = ?", (kb_id,))
        conn.execute("DELETE FROM knowledge_bases WHERE kb_id = ?", (kb_id,))

        if was_active:
            fallback = conn.execute(
                """
                SELECT kb_id FROM knowledge_bases
                ORDER BY updated_at DESC, created_at DESC
                LIMIT 1
                """
            ).fetchone()
            if fallback is not None:
                conn.execute(
                    "UPDATE knowledge_bases SET is_active = 1, updated_at = ? WHERE kb_id = ?",
                    (now, str(fallback[0])),
                )
        conn.commit()
    _remove_knowledge_base_files(storage_paths)
    _remove_vector_store(store_path, kb_id)
    return True


def save_knowledge_base_file(
    kb_id: str,
    file_name: str,
    file_type: str,
    storage_path: str | Path,
    chunk_count: int,
    status: str,
    db_path: str | Path = DATABASE_PATH,
) -> int:
    init_knowledge_tables(db_path)
    created_at = datetime.now().isoformat(timespec="seconds")
    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.execute(
            """
            INSERT INTO knowledge_base_files(kb_id, file_name, file_type, storage_path, chunk_count, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (kb_id, file_name, file_type, str(storage_path), int(chunk_count), status, created_at),
        )
        conn.execute(
            "UPDATE knowledge_bases SET updated_at = ? WHERE kb_id = ?",
            (created_at, kb_id),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_knowledge_base_files(
    kb_id: str,
    db_path: str | Path = DATABASE_PATH,
) -> list[KnowledgeBaseFile]:
    init_knowledge_tables(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT id, kb_id, file_name, file_type, storage_path, chunk_count, status, created_at
            FROM knowledge_base_files
            WHERE kb_id = ?
            ORDER BY id
            """,
            (kb_id,),
        ).fetchall()
    return [_file_from_row(row) for row in rows]


def delete_knowledge_base_file(
    file_id: int,
    db_path: str | Path = DATABASE_PATH,
) -> bool:
    init_knowledge_tables(db_path)
    now = datetime.now().isoformat(timespec="seconds")
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute(
            "SELECT kb_id FROM knowledge_base_files WHERE id = ?",
            (int(file_id),),
        ).fetchone()
        if row is None:
            return False

        kb_id = str(row[0])
        conn.execute("DELETE FROM knowledge_base_files WHERE id = ?", (int(file_id),))
        conn.execute(
            "UPDATE knowledge_bases SET updated_at = ? WHERE kb_id = ?",
            (now, kb_id),
        )
        conn.commit()
    return True


def _remove_knowledge_base_files(paths: list[Path]) -> None:
    for path in paths:
        if path.exists() and path.is_file():
            path.unlink()
        _remove_empty_parent(path.parent)


def _remove_vector_store(store_path: Path, kb_id: str) -> None:
    if store_path.is_file():
        store_path.unlink()

    store_dir = store_path if store_path.is_dir() else store_path.parent
    if store_dir.exists() and store_dir.name == kb_id:
        shutil.rmtree(store_dir)


def _remove_empty_parent(path: Path) -> None:
    try:
        path.rmdir()
    except OSError:
        pass

def _kb_from_row(row: tuple[object, ...]) -> KnowledgeBase:
    return KnowledgeBase(
        kb_id=str(row[0]),
        name=str(row[1]),
        store_path=str(row[2]),
        is_active=bool(row[3]),
        created_at=datetime.fromisoformat(str(row[4])),
        updated_at=datetime.fromisoformat(str(row[5])),
    )


def _file_from_row(row: tuple[object, ...]) -> KnowledgeBaseFile:
    return KnowledgeBaseFile(
        id=int(row[0]),
        kb_id=str(row[1]),
        file_name=str(row[2]),
        file_type=str(row[3]),
        storage_path=str(row[4]),
        chunk_count=int(row[5]),
        status=str(row[6]),
        created_at=datetime.fromisoformat(str(row[7])),
    )
