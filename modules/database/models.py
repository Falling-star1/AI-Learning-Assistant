from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class ChatRecord:
    id: int
    session_id: str
    role: str
    content: str
    mode: str
    provider: str
    created_at: datetime
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class FileRecord:
    id: int
    file_name: str
    file_type: str
    storage_path: str
    status: str
    created_at: datetime

@dataclass(frozen=True)
class KnowledgeBase:
    kb_id: str
    name: str
    store_path: str
    is_active: bool
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class KnowledgeBaseFile:
    id: int
    kb_id: str
    file_name: str
    file_type: str
    storage_path: str
    chunk_count: int
    status: str
    created_at: datetime

@dataclass(frozen=True)
class ChatSession:
    session_id: str
    title: str
    active_kb_id: str
    created_at: datetime
    updated_at: datetime
