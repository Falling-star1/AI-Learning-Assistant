# PPT Outline And Network Fallback Implementation Plan

**Goal:** 补齐需求文档中列为必做但缺失的「PPT 提纲生成」功能，并为云端 LLM 调用增加网络异常降级，避免断网时直接抛 traceback 影响演示。

**Architecture:** PPT 提纲复用现有 report 生成管线（prompt 模板 + generator fallback + router 路由），仅新增模板与分支。网络降级在 `CloudLLMProvider.generate` 层捕获 `openai.APIConnectionError` / `APITimeoutError`，自动降级到 `FallbackLLMProvider` 并写入日志，上层无感知。

**Tech Stack:** Python, Streamlit, OpenAI SDK, 项目内 logger。

---

### Task 1: PPT 提纲 Prompt 模板

**Files:**
- Modify: `modules/llm/prompt.py`

- [x] 新增 `PPT_OUTLINE_TEMPLATE`，覆盖封面、目录、各页要点、演讲节奏、可视化建议。
- [x] 在 `LEARNING_TEMPLATES` 注册 `"PPT 提纲"` 键，使 `build_learning_prompt` 能命中。

### Task 2: PPT 提纲本地 Fallback 生成

**Files:**
- Modify: `modules/report/generator.py`

- [x] 在 `_build_local_report` 增加 `report_type == "PPT 提纲"` 分支。
- [x] 新增 `_build_local_ppt_outline`，输出 8 页结构化答辩提纲草稿，供无 API Key 时演示。
- [x] 确认 `build_report_prompt` 的非实验报告分支会自动走 `build_learning_prompt("PPT 提纲", ...)`。

### Task 3: 云端 LLM 网络异常降级

**Files:**
- Modify: `modules/llm/provider.py`

- [x] 顶部 `try` 导入 `openai.APIConnectionError` / `APITimeoutError`，未安装 openai 时降级为空 tuple，不影响其他 provider。
- [x] 接入 `modules.utils.logger.get_logger`。
- [x] `CloudLLMProvider.generate` 用 `try/except _NETWORK_ERRORS` 捕获网络异常，记录 warning 日志，调用 `FallbackLLMProvider` 降级，返回带 `⚠️` 提示的 `LLMResult(provider="fallback")`。
- [x] `DeepSeekLLMProvider` 继承自 `CloudLLMProvider`，自动获得降级能力。

### Task 4: Streamlit 生成类型选项

**Files:**
- Modify: `app.py`

- [x] 在 `MODE_REPORT` 的 `st.segmented_control` 选项加入 `"PPT 提纲"`。
- [x] 确认 `report_type` 与 `router.selected_report_type` 逻辑能正确把 `"PPT 提纲"` 透传到 `generate_report`。

### Task 5: 验证

- [x] 导入测试通过：`LEARNING_TEMPLATES` 含 `"PPT 提纲"`，`_build_local_ppt_outline` 正常输出。
- [x] `test_llm_provider_generic_cloud` 3 个测试全过，provider 改动未破坏现有逻辑。
- [ ] 手动验证：重启 Streamlit，在「学习辅助生成」模式选择「PPT 提纲」，确认生成与下载正常。
- [ ] 手动验证：断网或关停 Ollama 后用云端 provider 提问，确认降级提示出现而非 traceback。
