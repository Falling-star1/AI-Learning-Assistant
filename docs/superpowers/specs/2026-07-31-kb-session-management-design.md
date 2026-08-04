# Knowledge Base And Session Management Design

## Goal

Fix the current V1 limitations where RAG uses one overwritten `rag_store.json`, uploads are tied to a mode, references are not consistently structured, and chat history cannot be restored after refresh.

## Scope

This V1 keeps Streamlit and the lightweight JSON keyword vector store. It adds SQLite metadata around the existing store instead of replacing the retrieval engine with Chroma/FAISS immediately.

Included:
- Multiple knowledge bases, each with its own store path.
- File records attached to a knowledge base.
- Upload to the active knowledge base from the sidebar, independent of chat mode.
- Rebuild active knowledge base from its file list.
- Structured RAG sources returned by the Agent for fallback and Qwen flows.
- Persistent sessions list, selectable in the sidebar, with full chat restoration.

Deferred:
- Per-file chunk deletion without full rebuild.
- User login and cloud sync.
- Full vector database migration.

## Data Model

`knowledge_bases` stores `kb_id`, name, store path, active flag and timestamps.
`knowledge_base_files` stores file metadata linked to `kb_id`.
`sessions` stores `session_id`, title, active `kb_id`, and timestamps.

The JSON vector store moves from one global file to one file per knowledge base:

```text
data/vector_db/kb_<id>/rag_store.json
```

## Flow

1. App starts and ensures a default knowledge base and session exist.
2. Sidebar lets the user select or create a knowledge base.
3. Upload goes to the selected knowledge base and appends chunks to that knowledge base store.
4. RAG asks and report generation use the selected knowledge base store path.
5. Agent always returns `sources` as structured data.
6. Sidebar session list loads complete messages by selected session.

## Error Handling

If upload parsing fails, the app shows a normal assistant-style error and does not mark the knowledge base as ready. If rebuilding fails for one file, the rebuild should report the failing file clearly.

## Testing

Use unittest. Add tests for knowledge base CRUD, append-style vector stores, route-level active store selection, structured sources, and session restoration helpers.
