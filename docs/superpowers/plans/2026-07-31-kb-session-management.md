# Knowledge Base And Session Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add V1 knowledge base management and persistent chat session recovery to the Streamlit RAG app.

**Architecture:** SQLite owns metadata for knowledge bases, files, and sessions. JSON vector stores remain the retrieval backend but are scoped by `kb_id`. Streamlit reads/writes selected `kb_id` and `session_id` from session state and sends the selected store path to the Agent.

**Tech Stack:** Python, Streamlit, SQLite, unittest, current keyword JSON vector store.

---

### Task 1: Knowledge Base Metadata

**Files:**
- Modify: `modules/database/models.py`
- Create: `modules/database/knowledge.py`
- Test: `tests/test_knowledge_base_management.py`

- [ ] Write failing tests for creating/listing/selecting a knowledge base and saving file metadata.
- [ ] Implement SQLite tables `knowledge_bases` and `knowledge_base_files`.
- [ ] Add helpers: `ensure_default_knowledge_base`, `create_knowledge_base`, `list_knowledge_bases`, `set_active_knowledge_base`, `get_active_knowledge_base`, `save_knowledge_base_file`, `list_knowledge_base_files`.
- [ ] Run focused tests and commit.

### Task 2: Per-KB Vector Store

**Files:**
- Modify: `modules/rag/vector_store.py`
- Modify: `modules/rag/qa.py`
- Test: `tests/test_knowledge_base_rag.py`

- [ ] Write failing tests proving appending a second file keeps the first file searchable.
- [ ] Add append mode to `build_vector_store`.
- [ ] Add `get_knowledge_base_store_path(kb_id)` helper.
- [ ] Run focused tests and commit.

### Task 3: Agent Structured Sources

**Files:**
- Modify: `modules/agent/router.py`
- Modify: `modules/rag/qa.py`
- Test: `tests/test_route_knowledge_base.py`

- [ ] Write failing tests that route uses a passed `store_path` and returns structured `sources` for course Q&A.
- [ ] Return `sources` from RAG route for fallback and remote provider flows.
- [ ] Keep `answer` as display text for compatibility.
- [ ] Run focused tests and commit.

### Task 4: Persistent Sessions

**Files:**
- Create: `modules/database/sessions.py`
- Modify: `modules/database/models.py`
- Test: `tests/test_sessions.py`

- [ ] Write failing tests for creating sessions, listing sessions, updating title, and loading chat records for a session.
- [ ] Implement `sessions` SQLite table and helper functions.
- [ ] Run focused tests and commit.

### Task 5: Streamlit Sidebar UI

**Files:**
- Modify: `app.py`
- Test: existing unit helpers plus manual browser verification

- [ ] Move course upload to sidebar knowledge-base area.
- [ ] Add selected knowledge base selectbox, create form, file list, rebuild button.
- [ ] Add session selectbox and new session button.
- [ ] Render structured RAG sources below answers.
- [ ] Run full tests, browser smoke test, update daily log and commit.
