# Pluggable LLM Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a pluggable LLM Provider layer so the project can use Qwen API when configured and fallback to local RAG evidence when no API key exists.

**Architecture:** `modules.llm.chat` exposes a stable `chat_with_llm()` function for upper layers. Provider-specific logic lives in `modules.llm.provider`, with `QwenLLMProvider`, `FallbackLLMProvider`, and a `create_llm_provider()` factory.

**Tech Stack:** Python, unittest, DashScope/Qwen API adapter, local fallback mode.

---

### Task 1: Test Provider Behavior

**Files:**
- Create: `tests/test_llm_provider.py`

- [x] **Step 1: Write tests for fallback behavior**
- [x] **Step 2: Write tests for missing Qwen API key provider selection**
- [x] **Step 3: Write tests for Qwen provider with injected fake client**
- [x] **Step 4: Write tests for `chat_with_llm()` using an injected provider**

### Task 2: Implement Provider Layer

**Files:**
- Create: `modules/llm/provider.py`
- Modify: `modules/llm/chat.py`
- Modify: `config.py`

- [x] **Step 1: Add `LLMResult` and `LLMProvider` interface**
- [x] **Step 2: Add fallback provider for no-key mode**
- [x] **Step 3: Add Qwen provider with injectable client**
- [x] **Step 4: Add provider factory**
- [x] **Step 5: Update `chat_with_llm()` to call the selected provider**
- [x] **Step 6: Add `QWEN_MODEL_NAME` config**

### Task 3: Document The Design

**Files:**
- Modify: `docs/01_项目需求文档.md`
- Create: `../学习心得/01_可插拔LLM架构.md`

- [x] **Step 1: Add Pluggable LLM Provider sentence to PRD**
- [x] **Step 2: Add learning note explaining Provider architecture and fallback mode**
- [x] **Step 3: Run tests and syntax validation**
