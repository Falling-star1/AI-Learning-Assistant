import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from config import VECTOR_DB_DIR


DEFAULT_STORE_PATH = VECTOR_DB_DIR / "rag_store.json"


@dataclass(frozen=True)
class SearchResult:
    text: str
    source_name: str
    score: float
    chunk_id: int


def build_vector_store(
    chunks: list[str],
    source_name: str,
    store_path: str | Path = DEFAULT_STORE_PATH,
    append: bool = False,
) -> Path:
    """Persist chunks in a small local store.

    V1 uses keyword scoring so the RAG flow works without downloading an
    embedding model. The file boundary stays the same, so this can later be
    replaced by Chroma/FAISS without changing the caller.
    """
    target_path = Path(store_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    existing_records = _load_records(target_path) if append else []
    next_chunk_id = _next_chunk_id(existing_records)
    records = [
        {
            "chunk_id": next_chunk_id + index - 1,
            "source_name": source_name,
            "text": chunk,
            "tokens": _tokenize(chunk),
        }
        for index, chunk in enumerate(chunks, start=1)
        if chunk.strip()
    ]
    target_path.write_text(
        json.dumps(existing_records + records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target_path


def _load_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _next_chunk_id(records: list[dict[str, object]]) -> int:
    if not records:
        return 1
    return max(int(record.get("chunk_id", 0)) for record in records) + 1

def list_vector_store_chunks(
    store_path: str | Path = DEFAULT_STORE_PATH,
    top_k: int = 3,
) -> list[SearchResult]:
    """Return representative chunks when a generation task needs course context but has no precise query hit."""
    path = Path(store_path)
    if not path.exists():
        return []

    records = _load_records(path)
    return [
        SearchResult(
            text=str(record["text"]),
            source_name=str(record["source_name"]),
            score=0.0,
            chunk_id=int(record["chunk_id"]),
        )
        for record in records[:top_k]
    ]

def search_vector_store(
    query: str,
    store_path: str | Path = DEFAULT_STORE_PATH,
    top_k: int = 3,
) -> list[SearchResult]:
    path = Path(store_path)
    if not path.exists():
        return []

    records = json.loads(path.read_text(encoding="utf-8"))
    query_tokens = Counter(_tokenize(query))
    if not query_tokens:
        return []

    results = []
    for record in records:
        chunk_tokens = Counter(record.get("tokens", []))
        score = _score(query_tokens, chunk_tokens)
        if score > 0:
            results.append(
                SearchResult(
                    text=record["text"],
                    source_name=record["source_name"],
                    score=score,
                    chunk_id=record["chunk_id"],
                )
            )

    return sorted(results, key=lambda item: item.score, reverse=True)[:top_k]


def _tokenize(text: str) -> list[str]:
    lower_text = text.lower()
    words = re.findall(r"[a-zA-Z0-9]+", lower_text)
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", lower_text)
    # Chinese text often has no spaces, so character bigrams give a simple
    # retrieval signal before we introduce a real embedding model.
    chinese_bigrams = [
        "".join(chinese_chars[index : index + 2])
        for index in range(max(len(chinese_chars) - 1, 0))
    ]
    return words + chinese_bigrams


def _score(query_tokens: Counter[str], chunk_tokens: Counter[str]) -> float:
    overlap = set(query_tokens) & set(chunk_tokens)
    if not overlap:
        return 0.0
    return sum(query_tokens[token] * chunk_tokens[token] for token in overlap)
