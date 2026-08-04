# Streamlit RAG Status Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Streamlit V1 page show visible RAG progress and answer metadata for classroom demonstration.

**Architecture:** Keep UI state in `st.session_state`; keep RAG execution in `modules.agent.router`; expose build metadata from the router so the UI can display it without parsing answer text.

**Tech Stack:** Streamlit, Python, unittest, local RAG fallback provider.

---

### Task 1: Add Router Metadata

**Files:**
- Modify: `tests/test_rag_pipeline.py`
- Modify: `modules/agent/router.py`

- [x] **Step 1: Test that course QA route returns source name, chunk count, and knowledge status**
- [x] **Step 2: Return metadata after uploaded document builds the knowledge base**

### Task 2: Enhance Streamlit Page

**Files:**
- Replace: `app.py`

- [x] **Step 1: Add `init_session_state()` for chat and knowledge status**
- [x] **Step 2: Add sidebar metrics for knowledge base state**
- [x] **Step 3: Render chat history from session state**
- [x] **Step 4: Show provider metadata in assistant messages**
- [x] **Step 5: Rerun after knowledge base creation so sidebar metrics update immediately**

### Task 3: Document And Verify

**Files:**
- Create: `../学习心得/03_Streamlit状态管理.md`

- [x] **Step 1: Explain Streamlit rerun and session state**
- [x] **Step 2: Run full unit tests**
- [x] **Step 3: Run Python syntax validation**
