import re
from pathlib import Path
from typing import Any

from config import (
    DATABASE_PATH,
    DEFAULT_LLM_PROVIDER,
    DETECTED_IMAGE_DIR,
    IMAGE_DIR,
    UPLOAD_DIR,
    YOLO_CONFIDENCE_THRESHOLD,
)
from modules.database.files import save_file_record
from modules.database.history import save_chat_record
from modules.llm.chat import chat_with_llm
from modules.llm.provider import create_llm_provider
from modules.rag.loader import DocumentLoadError
from modules.rag.qa import build_knowledge_base, retrieve_context_or_preview, retrieve_relevant_chunks
from modules.rag.vector_store import DEFAULT_STORE_PATH, SearchResult
from modules.report.generator import generate_report
from modules.utils.file_utils import save_uploaded_file
from modules.yolo.analyzer import analyze_with_llm, summarize_detections
from modules.yolo.detect import detect_image

MODE_AUTO = "自动识别"
MODE_COURSE_QA = "课程资料问答"
MODE_IMAGE = "图片目标检测"
MODE_REPORT = "学习辅助生成"
MODE_CHAT = "普通问答"
LEARNING_KEYWORDS = ("总结", "提纲", "大纲", "报告", "复习", "学习辅助")
RAG_KEYWORDS = ("根据资料", "课程资料", "知识库", "引用", "文档", "课件")
CASUAL_KEYWORDS = ("你好", "您好", "你是谁", "你是什么", "你能做什么", "早上好", "晚上好", "hello", "hi")
COURSE_TOPIC_KEYWORDS = (
    "cpu",
    "指令",
    "寄存器",
    "流水线",
    "cache",
    "缓存",
    "内存",
    "计算机",
    "操作系统",
    "算法",
    "网络",
    "数据库",
    "rag",
    "yolo",
    "工作流",
    "chatflow",
)
CONCEPT_QUERY_KEYWORDS = ("解释", "什么是", "概念", "原理", "流程", "机制", "区别", "为什么")
GENERAL_ANSWER_NOTE = "（未在课程资料中检索到相关内容，以下为通用回答）"


def detect_intent(
    question: str,
    uploaded_file: Any | None = None,
    uploaded_image: Any | None = None,
    has_knowledge_base: bool = False,
    provider: Any | None = None,
) -> str:
    text = question.strip().lower()
    if uploaded_image is not None:
        return MODE_IMAGE
    if uploaded_file is not None:
        return MODE_COURSE_QA

    if any(keyword in text for keyword in CASUAL_KEYWORDS):
        return MODE_CHAT
    if _looks_like_simple_math(text):
        return MODE_CHAT
    if any(keyword in text for keyword in LEARNING_KEYWORDS):
        return MODE_REPORT
    if any(keyword in text for keyword in RAG_KEYWORDS):
        return MODE_COURSE_QA

    llm_intent = _detect_intent_with_llm(question, has_knowledge_base=has_knowledge_base, provider=provider)
    if llm_intent:
        return llm_intent

    if has_knowledge_base and _looks_like_course_lookup(text):
        return MODE_COURSE_QA
    return MODE_CHAT


def route_user_request(
    question: str,
    mode: str,
    uploaded_file: Any | None = None,
    uploaded_image: Any | None = None,
    upload_dir: str | Path = UPLOAD_DIR,
    image_dir: str | Path = IMAGE_DIR,
    detected_dir: str | Path = DETECTED_IMAGE_DIR,
    confidence_threshold: float = YOLO_CONFIDENCE_THRESHOLD,
    report_type: str = "学习报告",
    learning_type: str = "学习报告",
    format_requirements: str = "",
    store_path: str | Path = DEFAULT_STORE_PATH,
    db_path: str | Path = DATABASE_PATH,
    session_id: str = "default",
    has_knowledge_base: bool = False,
    llm_provider_name: str = DEFAULT_LLM_PROVIDER,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Route user requests to the matching service module."""
    provider = create_llm_provider(llm_provider_name)
    provider_name = _provider_name(provider)
    resolved_mode = detect_intent(
        question,
        uploaded_file=uploaded_file,
        uploaded_image=uploaded_image,
        has_knowledge_base=has_knowledge_base,
        provider=provider,
    ) if mode == MODE_AUTO else mode

    if resolved_mode == MODE_COURSE_QA:
        metadata: dict[str, str] = {
            "source_name": "",
            "chunk_count": "0",
            "knowledge_status": "empty",
        }
        if uploaded_file is not None:
            saved_path = save_uploaded_file(uploaded_file, Path(upload_dir))
            try:
                build_info = build_knowledge_base(saved_path, store_path=store_path)
            except DocumentLoadError as exc:
                answer = f"资料解析失败：{exc}"
                workflow_steps = _build_workflow_steps(
                    mode,
                    resolved_mode,
                    "资料解析与 RAG 建库",
                    "资料解析失败，未调用问答生成。",
                )
                result = {
                    "answer": answer,
                    "provider": provider_name,
                    "resolved_mode": resolved_mode,
                    "source_name": saved_path.name,
                    "chunk_count": "0",
                    "knowledge_status": "error",
                    "workflow_steps": workflow_steps,
                }
                _save_turn(
                    session_id,
                    question,
                    answer,
                    resolved_mode,
                    provider_name,
                    db_path,
                    assistant_metadata=_assistant_metadata(result),
                )
                return result

            save_file_record(
                file_name=saved_path.name,
                file_type=_file_type(saved_path),
                storage_path=saved_path,
                status="indexed",
                db_path=db_path,
            )
            metadata = {
                "source_name": str(build_info["source_name"]),
                "chunk_count": str(build_info["chunk_count"]),
                "knowledge_status": "ready",
            }
        retrieval_results = retrieve_relevant_chunks(question, store_path=store_path, top_k=3)
        sources = _build_report_sources(retrieval_results)
        if not retrieval_results:
            fallback_answer = _append_general_answer_note(
                chat_with_llm(question, conversation_history=conversation_history, provider=provider)
            )
            workflow_steps = _build_workflow_steps(
                mode,
                resolved_mode,
                f"RAG 检索为空 → LLM 通用回答（Provider: {provider_name}）",
                "未检索到可引用来源，已回退到普通问答。",
            )
            result = {
                "answer": fallback_answer,
                "provider": provider_name,
                "resolved_mode": resolved_mode,
                "sources": [],
                "workflow_steps": workflow_steps,
                **metadata,
            }
            _save_turn(
                session_id,
                question,
                fallback_answer,
                resolved_mode,
                provider_name,
                db_path,
                assistant_metadata=_assistant_metadata(result),
            )
            return result

        answer = chat_with_llm(
            question,
            context_chunks=_format_report_context(retrieval_results),
            conversation_history=conversation_history,
            provider=provider,
        )
        workflow_steps = _build_workflow_steps(
            mode,
            resolved_mode,
            f"RAG 知识库问答（Provider: {provider_name}）",
            _result_detail("返回课程资料回答", sources),
        )
        result = {
            "answer": answer,
            "provider": provider_name,
            "resolved_mode": resolved_mode,
            "sources": sources,
            "workflow_steps": workflow_steps,
            **metadata,
        }
        _save_turn(
            session_id,
            question,
            answer,
            resolved_mode,
            provider_name,
            db_path,
            assistant_metadata=_assistant_metadata(result),
        )
        return result

    if resolved_mode == MODE_IMAGE:
        if uploaded_image is None:
            answer = "请先上传一张图片，再进行 YOLO 目标检测。"
            workflow_steps = _build_workflow_steps(
                mode,
                resolved_mode,
                "YOLO 图片目标检测",
                "缺少图片输入，返回上传提示。",
            )
            result = {
                "answer": answer,
                "provider": "yolo",
                "resolved_mode": resolved_mode,
                "workflow_steps": workflow_steps,
            }
            _save_turn(
                session_id,
                question,
                answer,
                resolved_mode,
                "yolo",
                db_path,
                assistant_metadata=_assistant_metadata(result),
            )
            return result

        saved_image = save_uploaded_file(uploaded_image, Path(image_dir))
        detection_result = detect_image(
            saved_image,
            output_dir=detected_dir,
            confidence_threshold=confidence_threshold,
        )
        if provider_name == "fallback":
            answer = summarize_detections(detection_result["detections"])
            answer_provider = "yolo"
        else:
            answer = analyze_with_llm(detection_result["detections"], question, provider=provider)
            answer_provider = f"yolo+{provider_name}"
        workflow_steps = _build_workflow_steps(
            mode,
            resolved_mode,
            f"YOLO 图片检测（Provider: {answer_provider}）",
            f"返回检测结果 {len(detection_result['detections'])} 个。",
        )
        result = {
            "answer": answer,
            "provider": answer_provider,
            "resolved_mode": resolved_mode,
            "workflow_steps": workflow_steps,
            **detection_result,
        }
        _save_turn(
            session_id,
            question,
            answer,
            resolved_mode,
            answer_provider,
            db_path,
            assistant_metadata=_assistant_metadata(result),
        )
        return result

    if resolved_mode == MODE_REPORT:
        retrieval_results = retrieve_context_or_preview(question, store_path=store_path, top_k=3)
        context_chunks = _format_report_context(retrieval_results)
        sources = _build_report_sources(retrieval_results)
        selected_report_type = report_type if report_type == "实验报告" else learning_type
        report = generate_report(
            topic=question,
            report_type=selected_report_type,
            format_requirements=format_requirements,
            context_chunks=context_chunks,
            provider=provider,
        )
        answer = _append_reference_section(report.content, sources)
        workflow_steps = _build_workflow_steps(
            mode,
            resolved_mode,
            f"RAG 检索 + {report.report_type}生成（Provider: {report.provider}）",
            _result_detail("返回可下载报告", sources),
        )
        result = {
            "answer": answer,
            "provider": report.provider,
            "resolved_mode": resolved_mode,
            "report_type": report.report_type,
            "download_name": report.download_name,
            "knowledge_status": report.knowledge_status,
            "sources": sources,
            "workflow_steps": workflow_steps,
        }
        _save_turn(
            session_id,
            question,
            answer,
            resolved_mode,
            provider_name,
            db_path,
            assistant_metadata=_assistant_metadata(result),
        )
        return result

    answer = chat_with_llm(question, conversation_history=conversation_history, provider=provider)
    workflow_steps = _build_workflow_steps(
        mode,
        resolved_mode,
        f"LLM 普通问答（Provider: {provider_name}）",
        "返回通用问答结果。",
    )
    result = {
        "answer": answer,
        "provider": provider_name,
        "resolved_mode": resolved_mode,
        "workflow_steps": workflow_steps,
    }
    _save_turn(
        session_id,
        question,
        answer,
        resolved_mode,
        provider_name,
        db_path,
        assistant_metadata=_assistant_metadata(result),
    )
    return result


def _format_report_context(results: list[SearchResult]) -> list[str]:
    return [
        f"来源：{result.source_name}，片段 {result.chunk_id}\n{result.text}"
        for result in results
    ]


def _detect_intent_with_llm(
    question: str,
    has_knowledge_base: bool,
    provider: Any | None,
) -> str | None:
    if provider is None or _provider_name(provider) == "fallback":
        return None

    prompt = (
        "你是学习助手的轻量意图分类器。请根据用户输入判断类别，只输出一个类别："
        "闲聊、概念解释、资料查询、复习生成、图片检测、普通问答。\n"
        "分类规则：\n"
        "- 闲聊：问候、你是谁、你是什么模型、能力介绍等。\n"
        "- 复习生成：要求总结、提纲、练习题、报告、PPT 大纲。\n"
        "- 图片检测：要求识别图片或检测物体。\n"
        "- 资料查询：明确要求根据资料、课程资料、知识库、引用来源回答。\n"
        "- 概念解释：解释课程或知识概念。\n"
        "- 普通问答：其他不需要课程资料的问题。\n"
        f"当前是否已有知识库：{'是' if has_knowledge_base else '否'}\n"
        f"用户输入：{question}\n"
        "只输出一个类别。"
    )
    try:
        result = provider.generate(prompt=prompt)
    except Exception:
        return None
    if not result.used_remote_model or result.provider == "fallback":
        return None
    return _map_intent_label(result.text, has_knowledge_base=has_knowledge_base)


def _map_intent_label(label: str, has_knowledge_base: bool) -> str | None:
    normalized = label.strip().lower()
    if "闲聊" in normalized:
        return MODE_CHAT
    if "图片检测" in normalized:
        return MODE_IMAGE
    if "复习生成" in normalized:
        return MODE_REPORT
    if "资料查询" in normalized:
        return MODE_COURSE_QA
    if "概念解释" in normalized:
        return MODE_COURSE_QA if has_knowledge_base else MODE_CHAT
    if "普通问答" in normalized:
        return MODE_CHAT
    return None


def _looks_like_course_lookup(text: str) -> bool:
    if any(keyword in text for keyword in COURSE_TOPIC_KEYWORDS):
        return True
    return any(keyword in text for keyword in CONCEPT_QUERY_KEYWORDS)


def _looks_like_simple_math(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if re.fullmatch(r"[0-9+\-*/×÷().=＝等于几多少]+", compact):
        return True
    return bool(re.search(r"\d+\s*([+\-*/×÷]|加|减|乘|除)\s*\d+", text))


def _append_general_answer_note(answer: str) -> str:
    return f"{answer.rstrip()}\n\n{GENERAL_ANSWER_NOTE}"


def _build_report_sources(results: list[SearchResult]) -> list[dict[str, Any]]:
    return [
        {
            "source_name": result.source_name,
            "chunk_id": result.chunk_id,
            "score": round(result.score, 3),
        }
        for result in results
    ]


def _append_reference_section(content: str, sources: list[dict[str, Any]]) -> str:
    if not sources:
        return content
    references = "\n".join(
        f"- {source['source_name']}，片段 {source['chunk_id']}"
        for source in sources
    )
    return f"{content.rstrip()}\n\n## 参考资料\n\n{references}\n"


def _build_workflow_steps(
    requested_mode: str,
    resolved_mode: str,
    tool_detail: str,
    result_detail: str,
) -> list[dict[str, str]]:
    return [
        {
            "title": "识别意图",
            "detail": f"{requested_mode} → {resolved_mode}",
        },
        {
            "title": "调用工具",
            "detail": tool_detail,
        },
        {
            "title": "返回结果",
            "detail": result_detail,
        },
    ]


def _result_detail(prefix: str, sources: list[dict[str, Any]]) -> str:
    if sources:
        return f"{prefix}，引用来源 {len(sources)} 条。"
    return f"{prefix}，未检索到可引用来源。"


def _provider_name(provider: Any) -> str:
    return getattr(provider, "provider_name", provider.__class__.__name__)


def _assistant_metadata(result: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key in ("workflow_steps", "sources", "download_name", "knowledge_status"):
        value = result.get(key)
        if value:
            metadata[key] = value
    if result.get("annotated_image_path"):
        metadata["image_path"] = result["annotated_image_path"]
    return metadata


def _file_type(path: Path) -> str:
    return path.suffix.lower().lstrip(".") or "unknown"


def _save_turn(
    session_id: str,
    question: str,
    answer: str,
    mode: str,
    provider: str,
    db_path: str | Path,
    assistant_metadata: dict[str, Any] | None = None,
    save_assistant: bool = True,
) -> None:
    save_chat_record(session_id, "user", question, mode, provider, db_path=db_path)
    if not save_assistant:
        return
    save_chat_record(
        session_id,
        "assistant",
        answer,
        mode,
        provider,
        metadata=assistant_metadata,
        db_path=db_path,
    )
