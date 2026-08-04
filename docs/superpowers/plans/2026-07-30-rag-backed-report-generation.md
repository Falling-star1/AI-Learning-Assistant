# RAG 驱动的报告生成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让学习报告和实验报告优先使用现有 RAG 知识库片段生成，并在页面与下载内容中显示真实引用来源。

**Architecture:** Agent 路由层负责检索和业务编排，RAG 模块提供共享的结构化检索接口，报告模块接收上下文并负责生成正文，Streamlit 负责展示知识库使用状态和来源。没有 API Key 或没有相关资料时继续提供可演示的 fallback 草稿，不自动写入报告文件。

**Tech Stack:** Python 3.11、unittest、Streamlit、本地 JSON RAG 检索库、可插拔 LLM Provider、SQLite。

---

## 文件职责

- `modules/rag/qa.py`：提供课程问答和报告生成共用的检索入口。
- `modules/report/templates.py`：根据是否存在课程上下文生成明确的报告 Prompt。
- `modules/report/generator.py`：接收检索上下文，调用 Provider 或生成本地报告草稿。
- `modules/agent/router.py`：检索知识库、组织来源、生成报告并返回页面数据。
- `app.py`：展示报告是否使用知识库、引用来源和下载按钮。
- `tests/test_rag_pipeline.py`：验证共享检索接口。
- `tests/test_report_generator.py`：验证上下文传递、fallback 报告和 Agent 编排。
- `../日志/2026-07-30_开发日志.md`：只记录 2026-07-30 实际完成内容和验证结果。

除特别说明外，命令均在 `AI-Learning-Assistant` 目录执行。

### Task 1: 提供共享的 RAG 检索接口

**Files:**
- Modify: `tests/test_rag_pipeline.py`
- Modify: `modules/rag/qa.py:1-46`

- [ ] **Step 1: 写共享检索接口的失败测试**

在 `RAGPipelineTests` 中新增：

```python
def test_retrieve_relevant_chunks_returns_structured_search_results(self):
    from modules.rag.qa import retrieve_relevant_chunks
    from modules.rag.vector_store import build_vector_store

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = Path(tmp_dir) / "rag_store.json"
        build_vector_store(
            chunks=[
                "YOLO 用于图片目标检测。",
                "RAG 的流程包括解析、切分、检索和生成。",
            ],
            source_name="course.md",
            store_path=store_path,
        )

        results = retrieve_relevant_chunks(
            "RAG 的流程是什么？",
            store_path=store_path,
            top_k=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source_name, "course.md")
        self.assertEqual(results[0].chunk_id, 2)
        self.assertIn("解析、切分、检索和生成", results[0].text)
```

- [ ] **Step 2: 运行测试并确认失败原因**

Run:

```powershell
python -m unittest discover -s tests -p test_rag_pipeline.py -v
```

Expected: 新测试因无法导入 `retrieve_relevant_chunks` 而失败，原有 RAG 测试继续通过。

- [ ] **Step 3: 实现共享检索入口**

在 `modules/rag/qa.py` 中导入 `SearchResult`，并新增：

```python
def retrieve_relevant_chunks(
    query: str,
    store_path: str | Path = DEFAULT_STORE_PATH,
    top_k: int = 3,
) -> list[SearchResult]:
    """Expose structured retrieval for every RAG-backed workflow."""
    return search_vector_store(query, store_path=store_path, top_k=top_k)
```

将 `answer_with_rag` 中的直接检索改为：

```python
results = retrieve_relevant_chunks(
    question,
    store_path=store_path,
    top_k=top_k,
)
```

这条注释解释共享接口的职责，不为显而易见的赋值添加语句注释。

- [ ] **Step 4: 运行 RAG 测试**

Run:

```powershell
python -m unittest discover -s tests -p test_rag_pipeline.py -v
```

Expected: 所有 RAG 测试通过。

- [ ] **Step 5: 提交共享检索接口**

```powershell
git add modules/rag/qa.py tests/test_rag_pipeline.py
git commit -m "refactor: expose shared rag retrieval"
```

### Task 2: 让报告生成器消费课程上下文

**Files:**
- Modify: `tests/test_report_generator.py`
- Modify: `modules/report/templates.py:1-49`
- Modify: `modules/report/generator.py:1-190`

- [ ] **Step 1: 让测试 Provider 记录收到的上下文**

将测试中的 `StubProvider` 改为：

```python
class StubProvider:
    provider_name = "stub"

    def __init__(self):
        self.context_chunks = []

    def generate(self, prompt, context_chunks=None):
        self.context_chunks = list(context_chunks or [])
        return LLMResult(
            text=f"生成内容\n\n{prompt}",
            provider=self.provider_name,
            used_remote_model=True,
        )
```

- [ ] **Step 2: 写上下文生成和 fallback 生成的失败测试**

在 `ReportGeneratorTests` 中新增：

```python
def test_remote_report_provider_receives_retrieved_context(self):
    from modules.report.generator import generate_report

    provider = StubProvider()
    result = generate_report(
        topic="RAG 的工作流程",
        report_type="学习报告",
        context_chunks=["来源：course.md，片段 2\nRAG 包括解析、切分、检索和生成。"],
        provider=provider,
    )

    self.assertEqual(len(provider.context_chunks), 1)
    self.assertIn("解析、切分、检索和生成", provider.context_chunks[0])
    self.assertEqual(result.knowledge_status, "ready")
    self.assertIn("仅依据提供的课程资料", result.content)

def test_fallback_report_includes_retrieved_course_evidence(self):
    from modules.llm.provider import FallbackLLMProvider
    from modules.report.generator import generate_report

    result = generate_report(
        topic="RAG 的工作流程",
        report_type="学习报告",
        context_chunks=["来源：course.md，片段 2\nRAG 包括解析、切分、检索和生成。"],
        provider=FallbackLLMProvider(),
    )

    self.assertEqual(result.knowledge_status, "ready")
    self.assertIn("课程资料要点", result.content)
    self.assertIn("解析、切分、检索和生成", result.content)
```

- [ ] **Step 3: 运行测试并确认接口尚不存在**

Run:

```powershell
python -m unittest discover -s tests -p test_report_generator.py -v
```

Expected: 新测试因 `generate_report` 不接受 `context_chunks` 或 `ReportResult` 没有 `knowledge_status` 而失败。

- [ ] **Step 4: 扩展报告 Prompt**

为三个 Prompt 构建函数增加 `has_context: bool = False` 参数，并加入统一资料约束：

```python
def _build_context_instruction(has_context: bool) -> str:
    if has_context:
        return (
            "资料使用要求：仅依据提供的课程资料陈述课程事实；"
            "资料未覆盖的内容请明确说明，不要编造引用。"
        )
    return "资料使用要求：当前没有相关课程资料，请将结果标记为通用草稿。"
```

学习报告和实验报告 Prompt 都将该字符串放在结构要求之前。`build_report_prompt` 将 `has_context` 原样传给对应函数。

- [ ] **Step 5: 扩展报告生成结果和调用接口**

在 `ReportResult` 增加：

```python
knowledge_status: str
```

将 `generate_report` 签名扩展为：

```python
def generate_report(
    topic: str,
    report_type: str = "学习报告",
    format_requirements: str = "",
    context_chunks: Sequence[str] | None = None,
    provider: LLMProvider | None = None,
) -> ReportResult:
```

在函数内部规范化上下文并传给 Provider：

```python
chunks = tuple(chunk.strip() for chunk in (context_chunks or []) if chunk.strip())
prompt = build_report_prompt(
    topic,
    report_type,
    format_requirements,
    has_context=bool(chunks),
)

if provider_name == "fallback":
    content = _build_local_report(
        topic,
        report_type,
        format_requirements,
        context_chunks=chunks,
    )
else:
    content = selected_provider.generate(
        prompt=prompt,
        context_chunks=chunks,
    ).text
```

返回结果时设置：

```python
knowledge_status="ready" if chunks else "empty"
```

- [ ] **Step 6: 将检索证据加入本地报告**

新增证据格式化函数：

```python
def _build_materials_section(context_chunks: Sequence[str]) -> str:
    if not context_chunks:
        return "## 课程资料要点\n\n本次未检索到相关课程资料，以下内容为通用草稿。"

    blocks = []
    for index, chunk in enumerate(context_chunks, start=1):
        safe_chunk = chunk.replace("```", "` ` `")
        blocks.append(f"### 检索片段 {index}\n\n```text\n{safe_chunk}\n```")
    return "## 课程资料要点\n\n" + "\n\n".join(blocks)
```

`_build_local_study_report` 和 `_build_local_experiment_report` 都接收 `context_chunks`，并在标题后插入 `_build_materials_section(context_chunks)`。代码围栏用于避免课程 PDF 中的 Markdown 标题破坏页面字号。

- [ ] **Step 7: 运行报告测试**

Run:

```powershell
python -m unittest discover -s tests -p test_report_generator.py -v
```

Expected: 所有报告测试通过，且现有“不自动保存”测试仍通过。

- [ ] **Step 8: 提交报告上下文支持**

```powershell
git add modules/report/templates.py modules/report/generator.py tests/test_report_generator.py
git commit -m "feat: generate reports from rag context"
```

### Task 3: 在 Agent 中串联检索、生成和引用

**Files:**
- Modify: `tests/test_report_generator.py`
- Modify: `modules/agent/router.py:1-122`

- [ ] **Step 1: 写有知识库和空知识库的路由失败测试**

新增两个测试：

```python
def test_report_route_retrieves_context_and_returns_sources(self):
    from modules.agent.router import route_user_request
    from modules.rag.vector_store import build_vector_store

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = Path(tmp_dir) / "rag_store.json"
        build_vector_store(
            chunks=["RAG 的核心流程是解析、切分、检索和生成。"],
            source_name="rag-course.md",
            store_path=store_path,
        )

        result = route_user_request(
            question="RAG 的核心流程",
            mode="学习辅助生成",
            report_type="学习报告",
            store_path=store_path,
            db_path=Path(tmp_dir) / "app.db",
            session_id="report-with-rag",
        )

        self.assertEqual(result["knowledge_status"], "ready")
        self.assertEqual(len(result["sources"]), 1)
        self.assertEqual(result["sources"][0]["source_name"], "rag-course.md")
        self.assertIn("解析、切分、检索和生成", result["answer"])
        self.assertIn("## 参考资料", result["answer"])

def test_report_route_without_context_returns_empty_sources(self):
    from modules.agent.router import route_user_request

    with tempfile.TemporaryDirectory() as tmp_dir:
        result = route_user_request(
            question="RAG 的核心流程",
            mode="学习辅助生成",
            report_type="学习报告",
            store_path=Path(tmp_dir) / "missing-store.json",
            db_path=Path(tmp_dir) / "app.db",
            session_id="report-without-rag",
        )

        self.assertEqual(result["knowledge_status"], "empty")
        self.assertEqual(result["sources"], [])
        self.assertNotIn("## 参考资料", result["answer"])
```

- [ ] **Step 2: 运行测试并确认 Agent 尚未返回来源**

Run:

```powershell
python -m unittest discover -s tests -p test_report_generator.py -v
```

Expected: 新测试因缺少 `knowledge_status`、`sources` 或参考资料章节而失败。

- [ ] **Step 3: 新增 Agent 内部的数据转换函数**

导入 `retrieve_relevant_chunks` 和 `SearchResult`，新增：

```python
def _format_report_context(results: list[SearchResult]) -> list[str]:
    return [
        f"来源：{result.source_name}，片段 {result.chunk_id}\n{result.text}"
        for result in results
    ]


def _build_report_sources(results: list[SearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "source_name": result.source_name,
            "chunk_id": result.chunk_id,
            "score": round(result.score, 3),
        }
        for result in results
    ]


def _append_reference_section(
    content: str,
    sources: list[dict[str, Any]],
) -> str:
    if not sources:
        return content
    references = "\n".join(
        f"- {source['source_name']}，片段 {source['chunk_id']}"
        for source in sources
    )
    return f"{content.rstrip()}\n\n## 参考资料\n\n{references}\n"
```

这些辅助函数只负责格式转换，检索算法仍留在 RAG 模块。

- [ ] **Step 4: 修改学习辅助生成分支**

在生成报告前执行：

```python
retrieval_results = retrieve_relevant_chunks(
    question,
    store_path=store_path,
    top_k=3,
)
context_chunks = _format_report_context(retrieval_results)
sources = _build_report_sources(retrieval_results)
```

调用 `generate_report` 时传入 `context_chunks=context_chunks`，再执行：

```python
answer = _append_reference_section(report.content, sources)
```

返回字典增加：

```python
"knowledge_status": report.knowledge_status,
"sources": sources,
```

- [ ] **Step 5: 运行报告和 RAG 测试**

Run:

```powershell
python -m unittest discover -s tests -p test_report_generator.py -v
python -m unittest discover -s tests -p test_rag_pipeline.py -v
```

Expected: 两组测试全部通过。

- [ ] **Step 6: 提交 Agent 编排**

```powershell
git add modules/agent/router.py tests/test_report_generator.py
git commit -m "feat: attach rag sources to generated reports"
```

### Task 4: 在 Streamlit 中展示资料使用状态

**Files:**
- Modify: `app.py:79-215`

- [ ] **Step 1: 增加报告来源展示函数**

在 `render_report_download` 前新增：

```python
def render_report_sources(
    knowledge_status: str | None,
    sources: list[dict[str, object]],
) -> None:
    if knowledge_status != "ready" or not sources:
        st.warning("本次报告未使用知识库资料，当前结果为通用草稿。")
        return

    st.success(f"本次报告参考了 {len(sources)} 个知识库片段。")
    with st.expander("查看引用来源"):
        for source in sources:
            st.write(f"{source['source_name']} · 片段 {source['chunk_id']}")
```

- [ ] **Step 2: 将来源元数据保存到会话消息**

构造 `assistant_message` 时增加：

```python
"knowledge_status": result.get("knowledge_status"),
"sources": result.get("sources", []),
```

在即时回复和 `render_chat_history` 中，仅当消息包含 `download_name` 时，先调用：

```python
render_report_sources(
    message.get("knowledge_status"),
    message.get("sources", []),
)
```

然后继续渲染下载按钮，保持“生成后由用户下载保存”的行为。

- [ ] **Step 3: 执行语法检查**

Run:

```powershell
python -m compileall -q app.py modules tests
```

Expected: 命令退出码为 0，无语法错误。

- [ ] **Step 4: 启动 Streamlit 并手动检查**

Run:

```powershell
streamlit run app.py --server.headless true
```

检查流程：

1. 在“课程资料问答”上传含有明确 RAG 内容的 TXT 文件并完成一次提问。
2. 切换“学习辅助生成”，输入同一主题并生成学习报告。
3. 确认页面显示“参考了 N 个知识库片段”和来源展开区。
4. 点击下载，确认 Markdown 中含“参考资料”章节。
5. 使用无关主题生成报告，确认页面提示通用草稿且没有虚假来源。

- [ ] **Step 5: 提交 Streamlit 展示**

```powershell
git add app.py
git commit -m "feat: show report knowledge sources"
```

### Task 5: 全量验证并记录 2026-07-30 日志

**Files:**
- Create: `../日志/2026-07-30_开发日志.md`

- [ ] **Step 1: 运行完整测试**

Run:

```powershell
python -m unittest discover -s tests -v
```

Expected: 全部测试通过。

- [ ] **Step 2: 运行工程完整性检查**

Run:

```powershell
python -m compileall -q app.py config.py modules tests
python -c "import sqlite3; from config import DATABASE_PATH; connection=sqlite3.connect(DATABASE_PATH); print(connection.execute('pragma integrity_check').fetchone()[0]); connection.close()"
git diff --check
git status --short
```

Expected:

- Python 编译检查退出码为 0。
- SQLite 输出 `ok`。
- `git diff --check` 无输出。
- Git 状态只包含尚未提交的 2026-07-30 日志和实施计划。

- [ ] **Step 3: 写当天开发日志**

日志固定包含以下章节：

```markdown
# 2026-07-30 开发日志

## 今日任务

- 打通 RAG 检索与报告生成链路。
- 在报告正文和页面中展示引用来源。
- 保持无 API Key、无相关资料和下载后保存三种兼容行为。

## 实现内容

- 记录共享检索接口、报告上下文、Agent 编排和 Streamlit 来源展示的实际改动。

## 遇到问题

- 记录测试驱动过程中实际出现的失败现象、原因和解决方式。

## 验证记录

- 填写本次真实执行的测试数量、编译检查、SQLite 检查和页面联调结果。

## 下一步计划

- 接入真实 Embedding 和 Chroma/FAISS，替换当前关键词检索实现。
```

日志只写 2026-07-30 实际完成的内容，不重复登记 2026-07-29 已完成的报告下载功能。

- [ ] **Step 4: 提交实施计划和开发日志**

```powershell
git add docs/superpowers/plans/2026-07-30-rag-backed-report-generation.md ../日志/2026-07-30_开发日志.md
git commit -m "docs: record rag-backed report work"
```

- [ ] **Step 5: 确认最终仓库状态**

Run:

```powershell
git status --short
git log -6 --oneline
```

Expected: 工作区干净，最近提交清楚展示设计、共享检索、报告上下文、Agent 来源、页面展示和开发日志。
