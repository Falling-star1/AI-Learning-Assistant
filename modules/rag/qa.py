from pathlib import Path

from modules.llm.provider import LLMProvider
from modules.llm.chat import chat_with_llm
from modules.rag.loader import load_document
from modules.rag.splitter import split_text
from modules.rag.vector_store import (
    DEFAULT_STORE_PATH,
    SearchResult,
    build_vector_store,
    list_vector_store_chunks,
    search_vector_store,
)


def build_knowledge_base(
    document_path: str | Path,
    store_path: str | Path = DEFAULT_STORE_PATH,
    chunk_size: int = 800,
    overlap: int = 100,
    append: bool = False,
) -> dict[str, int | str]:
    text = load_document(document_path)
    chunks = split_text(text, chunk_size=chunk_size, overlap=overlap)
    source_name = Path(document_path).name
    build_vector_store(
        chunks=chunks,
        source_name=source_name,
        store_path=store_path,
        append=append,
    )
    return {
        "source_name": source_name,
        "chunk_count": len(chunks),
        "store_path": str(store_path),
    }


def retrieve_relevant_chunks(
    query: str,
    store_path: str | Path = DEFAULT_STORE_PATH,
    top_k: int = 3,
) -> list[SearchResult]:
    """Expose structured retrieval for every RAG-backed workflow."""
    return search_vector_store(query, store_path=store_path, top_k=top_k)


def retrieve_context_or_preview(
    query: str,
    store_path: str | Path = DEFAULT_STORE_PATH,
    top_k: int = 3,
) -> list[SearchResult]:
    """Use precise retrieval first, then fall back to representative material for open-ended generation."""
    results = retrieve_relevant_chunks(query, store_path=store_path, top_k=top_k)
    if results:
        return results
    return list_vector_store_chunks(store_path=store_path, top_k=top_k)

def answer_with_rag(
    question: str,
    store_path: str | Path = DEFAULT_STORE_PATH,
    provider: LLMProvider | None = None,
    top_k: int = 3,
) -> str:
    """Answer a question using retrieved course material."""
    results = retrieve_relevant_chunks(
        question,
        store_path=store_path,
        top_k=top_k,
    )
    if not results:
        return "当前知识库中没有检索到相关片段，请先上传课程资料或换一种问法。"

    context_chunks = [
        f"来源：{result.source_name}，片段 {result.chunk_id}\n{result.text}"
        for result in results
    ]
    return chat_with_llm(question, context_chunks=context_chunks, provider=provider)
