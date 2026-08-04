# SQLite Chat History Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist chat history to SQLite and show recent conversation records in the Streamlit V1 sidebar.

**Architecture:** The database layer owns schema creation and record mapping. The Agent router saves user/assistant turns after each answer. Streamlit keeps its in-memory chat UI while also reading recent persisted records by session ID.

**Tech Stack:** Python, sqlite3, unittest, Streamlit session state.

---

### Task 1: Test Chat History

**Files:**
- Create: `tests/test_chat_history.py`

- [x] **Step 1: Test saving user and assistant records**
- [x] **Step 2: Test listing records by session ID**
- [x] **Step 3: Test route persistence after a RAG request**

### Task 2: Implement Database Layer

**Files:**
- Replace: `modules/database/history.py`
- Replace: `modules/database/models.py`

- [x] **Step 1: Create `chat_history` table on demand**
- [x] **Step 2: Save one chat message with mode and provider metadata**
- [x] **Step 3: List chat records as `ChatRecord` dataclasses**
- [x] **Step 4: Close SQLite connections explicitly on Windows**

### Task 3: Wire Router And UI

**Files:**
- Modify: `modules/agent/router.py`
- Modify: `app.py`

- [x] **Step 1: Add `session_id` and `db_path` route parameters**
- [x] **Step 2: Save user and assistant messages for supported modes**
- [x] **Step 3: Generate a Streamlit session ID**
- [x] **Step 4: Show recent assistant records in the sidebar**

### Task 4: Documentation And Logs

**Files:**
- Modify: `docs/03_数据库设计.md`
- Create: `../学习心得/05_SQLite对话历史.md`
- Create: `../日志/2026-07-28_开发日志.md`

- [x] **Step 1: Update database design document**
- [x] **Step 2: Write learning note**
- [x] **Step 3: Write standardized development log**
- [x] **Step 4: Run full verification**
