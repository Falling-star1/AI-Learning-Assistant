from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from modules.llm.provider import LLMProvider, create_llm_provider
from modules.report.templates import build_report_prompt


@dataclass(frozen=True)
class ReportResult:
    content: str
    download_name: str
    report_type: str
    provider: str
    knowledge_status: str


def generate_report(
    topic: str,
    report_type: str = "学习报告",
    format_requirements: str = "",
    context_chunks: Sequence[str] | None = None,
    provider: LLMProvider | None = None,
) -> ReportResult:
    selected_provider = provider or create_llm_provider()
    provider_name = _provider_name(selected_provider)
    chunks = tuple(chunk.strip() for chunk in (context_chunks or []) if chunk.strip())
    prompt = build_report_prompt(
        topic,
        report_type,
        format_requirements,
        has_context=bool(chunks),
    )

    # Learning helpers must be demonstrable without an API key, so fallback uses
    # deterministic structures instead of returning raw retrieval snippets only.
    if provider_name == "fallback":
        content = _build_local_report(topic, report_type, format_requirements, chunks)
    else:
        content = selected_provider.generate(prompt=prompt, context_chunks=chunks).text

    return ReportResult(
        content=content,
        download_name=get_report_download_name(topic, report_type),
        report_type=report_type,
        provider=provider_name,
        knowledge_status="ready" if chunks else "empty",
    )


def _build_local_report(
    topic: str,
    report_type: str,
    format_requirements: str,
    context_chunks: Sequence[str],
) -> str:
    if report_type == "实验报告":
        return _build_local_experiment_report(topic, format_requirements, context_chunks)
    if report_type == "复习提纲":
        return _build_local_outline(topic, context_chunks)
    if report_type == "报告大纲":
        return _build_local_report_outline(topic, context_chunks)
    if report_type == "PPT 提纲":
        return _build_local_ppt_outline(topic, context_chunks)
    return _build_local_summary(topic, context_chunks)


def _build_local_summary(topic: str, context_chunks: Sequence[str]) -> str:
    materials_section = _build_materials_section(context_chunks)
    return f"""# {topic} 课程总结

{materials_section}

## 核心概念

- {topic} 是本次学习的核心主题，需要结合课程资料进一步补充定义、流程和应用价值。

## 关键流程

1. 明确输入、任务目标和输出形式。
2. 梳理涉及的模型、工具或算法模块。
3. 结合案例验证理解是否正确。
4. 总结适用场景、优点和局限。

## 重要术语

- 关键词：待结合课程资料补充。
- 应用场景：适合整理为答辩说明和复习提纲。

## 易错点

- 只记结论，不理解技术流程。
- 混淆相近概念之间的边界。
- 忽略项目中的输入输出和异常情况。

## 练习题

1. 请用自己的话说明“{topic}”的核心作用。
2. 请列出“{topic}”在项目中的一个应用场景。
3. 如果该模块输出不理想，你会从哪些方面排查？

## 参考答案

1. 可围绕概念、流程、项目应用三个方面作答。
2. 结合本项目中的 RAG、Agent、YOLO 或报告生成流程说明。
3. 可从数据、参数、模型能力和使用场景四个方向排查。
"""


def _build_local_outline(topic: str, context_chunks: Sequence[str]) -> str:
    materials_section = _build_materials_section(context_chunks)
    return f"""# {topic} 复习提纲

{materials_section}

## 学习目标

- 理解 {topic} 的基本概念和适用场景。
- 能说明它在项目中的输入、处理流程和输出。

## 知识框架

1. 概念定义
2. 技术流程
3. 项目应用
4. 优势与局限
5. 常见问题

## 高频考点

- 该技术解决什么问题？
- 与相近方法相比有什么区别？
- 在项目中如何落地？

## 必会问题

1. {topic} 的核心流程是什么？
2. 如果没有 API Key，系统如何 fallback？
3. 这个模块如何体现课程知识点？

## 复习检查清单

- [ ] 能说清概念
- [ ] 能画出流程
- [ ] 能结合项目演示
- [ ] 能说明局限和改进方向
"""


def _build_local_report_outline(topic: str, context_chunks: Sequence[str]) -> str:
    materials_section = _build_materials_section(context_chunks)
    return f"""# {topic} 报告大纲

{materials_section}

## 报告标题

{topic} 项目报告

## 章节结构

1. 项目背景与需求分析
2. 系统总体架构
3. 核心功能设计
4. 关键模块实现
5. 测试与运行效果
6. 总结与展望

## 每章写作要点

- 背景：说明课程要求和用户痛点。
- 架构：展示前端、Agent、RAG、LLM、YOLO、数据库之间的关系。
- 实现：说明模块职责和关键流程。
- 测试：给出功能验证和边界情况。

## 可展示的项目亮点

- 可插拔 LLM Provider。
- RAG 知识库问答与结构化引用。
- Agent 自动意图识别。
- YOLO 检测与自然语言分析。

## 需要补充的材料

- 系统截图
- 运行步骤
- 关键代码片段
- 测试结果截图
"""


def _build_local_ppt_outline(topic: str, context_chunks: Sequence[str]) -> str:
    materials_section = _build_materials_section(context_chunks)
    return f"""# {topic} 答辩 PPT 提纲

{materials_section}

## 封面

- 标题：{topic}
- 副标题：AI 多模态学习助手课程大作业
- 成员与分工

## 目录页

1. 项目背景
2. 需求分析
3. 系统架构
4. 核心功能演示
5. 技术亮点
6. 测试与效果
7. 总结与展望

## 各页内容要点

### 第 1 页：项目背景

- 课程学习场景下的资料整理痛点
- 为什么需要 AI 多模态学习助手

### 第 2 页：需求分析

- 用户需求：资料问答、图片检测、内容生成
- 功能模块优先级划分

### 第 3 页：系统架构

- Streamlit + Python 模块化分层
- RAG / LLM / YOLO / Agent 协作关系图

### 第 4 页：核心功能演示

- 知识库问答 + 引用来源
- YOLO 目标检测与自然语言解释

### 第 5 页：学习辅助生成

- 课程总结、复习提纲、报告大纲、PPT 提纲
- 可插拔 LLM Provider（云端 / 本地 / fallback）

### 第 6 页：技术亮点

- 可插拔 LLM Provider 架构
- RAG 检索增强与结构化引用
- Agent 自动意图识别

### 第 7 页：测试与效果

- 功能验证截图
- 边界情况处理（断网降级、无 Key fallback）

### 第 8 页：总结与展望

- 完成情况
- 后续 Vue + FastAPI 升级方向

## 演讲节奏建议

- 封面与目录：约 1 分钟
- 背景与需求：约 2 分钟
- 架构与功能演示：约 4 分钟（重点）
- 亮点与测试：约 2 分钟
- 总结展望：约 1 分钟

## 可视化建议

- 架构图：用流程图展示模块协作
- 演示页：放真实运行截图
- 亮点页：用对比表展示云端/本地/fallback 差异
"""


def _build_local_experiment_report(
    topic: str,
    format_requirements: str,
    context_chunks: Sequence[str],
) -> str:
    materials_section = _build_materials_section(context_chunks)
    cleaned_format = format_requirements.strip()
    if cleaned_format:
        return f"""# {topic} 实验报告

{materials_section}

> 本报告已优先按照用户格式要求组织：{cleaned_format}

## 报告草稿

请按以下格式继续补充实验细节：

{cleaned_format}

## 内容提示

- 实验内容：{topic}
- 缺失信息可填写“待补充”
- 建议补充实验环境、关键步骤、结果截图和问题分析
"""

    return f"""# {topic} 实验报告

{materials_section}

## 实验名称

{topic}

## 实验目的

理解实验涉及的核心技术，并完成可演示的功能闭环。

## 实验环境

- Python
- Streamlit
- 本地项目环境

## 技术路线

先明确输入输出，再完成核心模块实现，最后通过页面进行演示。

## 实验步骤

1. 准备实验数据或输入内容。
2. 调用项目对应功能模块。
3. 观察输出结果。
4. 记录问题并进行调整。

## 核心代码说明

围绕模块职责、关键函数和数据流进行说明。

## 实验结果

待补充运行截图和结果描述。

## 问题与解决

待补充调试过程中遇到的问题和处理方式。

## 总结反思

本实验完成了从功能设计到运行验证的基本流程。
"""


def _build_materials_section(context_chunks: Sequence[str]) -> str:
    if not context_chunks:
        return "## 课程资料要点\n\n本次未检索到相关课程资料，以下内容为通用草稿。"

    blocks = []
    for index, chunk in enumerate(context_chunks, start=1):
        safe_chunk = chunk.replace("```", "` ` `")
        blocks.append(f"### 检索片段 {index}\n\n```text\n{safe_chunk}\n```")
    return "## 课程资料要点\n\n" + "\n\n".join(blocks)


def get_report_download_name(topic: str, report_type: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{_safe_filename(topic)}_{report_type}.md"


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r'[\\/:*?"<>|\s]+', "_", value.strip())
    return cleaned[:40].strip("_") or "report"


def _provider_name(provider: Any) -> str:
    return getattr(provider, "provider_name", provider.__class__.__name__)