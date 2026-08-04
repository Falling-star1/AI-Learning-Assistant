# Project Structure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure the course project into the user-approved `AI-Learning-Assistant` layout.

**Architecture:** Keep Streamlit as the V1 entry point and place reusable AI capabilities under `modules/`. Documentation is stored in `docs/`, runtime data in `data/`, generated outputs in `outputs/`, and static assets in `assets/`.

**Tech Stack:** Python, Streamlit, SQLite, LangChain/RAG-ready module boundaries, YOLO-ready module boundaries.

---

### Task 1: Create Project Skeleton

**Files:**
- Create: `AI-Learning-Assistant/`
- Move: `app.py`, `config.py`, `requirements.txt`, `README.md`
- Create: `data/`, `modules/`, `outputs/`, `assets/`, `docs/`

- [x] **Step 1: Create the top-level project folder**
- [x] **Step 2: Move existing root files into the project folder**
- [x] **Step 3: Create data, output, asset, module, and docs directories**

### Task 2: Add Placeholder Modules

**Files:**
- Create: `modules/agent/tools.py`
- Create: `modules/rag/embedding.py`
- Create: `modules/llm/prompt.py`
- Create: `modules/yolo/analyzer.py`
- Create: `modules/database/models.py`
- Create: `modules/utils/*.py`

- [x] **Step 1: Add Agent tool registry skeleton**
- [x] **Step 2: Add lazy embedding helper**
- [x] **Step 3: Add LLM prompt templates**
- [x] **Step 4: Add YOLO result summarizer**
- [x] **Step 5: Add database dataclass and utility helpers**

### Task 3: Complete Supporting Docs And Verification

**Files:**
- Create: `docs/03_数据库设计.md`
- Create: `docs/04_API接口设计.md`
- Verify: Python syntax and file layout

- [x] **Step 1: Add database design document**
- [x] **Step 2: Add API interface design document**
- [x] **Step 3: Verify required files exist and Python files parse**