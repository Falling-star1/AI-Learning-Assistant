import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

from config import DATABASE_PATH
from modules.database.models import FileRecord


def init_file_table(db_path: str | Path = DATABASE_PATH) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS file_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                storage_path TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_file_records_created_at ON file_records(created_at)"
        )
        conn.commit()
    return path


def save_file_record(
    file_name: str,
    file_type: str,
    storage_path: str | Path,
    status: str,
    db_path: str | Path = DATABASE_PATH,
) -> int:
    """Persist metadata for one uploaded file, not the file content itself."""
    init_file_table(db_path)
    created_at = datetime.now().isoformat(timespec="seconds")
    with closing(sqlite3.connect(db_path)) as conn:
        cursor = conn.execute(
            """
            INSERT INTO file_records(file_name, file_type, storage_path, status, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (file_name, file_type, str(storage_path), status, created_at),
        )
        record_id = int(cursor.lastrowid)
        conn.commit()
        return record_id


def list_file_records(
    db_path: str | Path = DATABASE_PATH,
    limit: int = 10,
) -> list[FileRecord]:
    init_file_table(db_path)
    with closing(sqlite3.connect(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT id, file_name, file_type, storage_path, status, created_at
            FROM file_records
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    rows = list(reversed(rows))
    return [_row_to_record(row) for row in rows]


def _row_to_record(row: tuple[object, ...]) -> FileRecord:
    return FileRecord(
        id=int(row[0]),
        file_name=str(row[1]),
        file_type=str(row[2]),
        storage_path=str(row[3]),
        status=str(row[4]),
        created_at=datetime.fromisoformat(str(row[5])),
    )
