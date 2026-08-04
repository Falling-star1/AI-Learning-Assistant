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
from modules.rag.qa import answer_with_rag, build_knowledge_base, retrieve_context_or_preview, retrieve_relevant_chunks
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


def detect_intent(
    question: str,
    uploaded_file: Any | None = None,
    uploaded_image: Any | None = None,
    has_knowledge_base: bool = False,
) -> str:
    text = question.strip().lower()
    if uploaded_image is not None:
        return MODE_IMAGE
    if uploaded_file is not None:
        return MODE_COURSE_QA
    if any(keyword in text for keyword in LEARNING_KEYWORDS):
        return MODE_REPORT
    if has_knowledge_base or any(keyword in text for keyword in RAG_KEYWORDS):
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
) -> dict[str, Any]:
    """Route user requests to the matching service module."""
    provider = create_llm_provider(llm_provider_name)
    provider_name = _provider_name(provider)
    resolved_mode = detect_intent(
        question,
        uploaded_file=uploaded_file,
        uploaded_image=uploaded_image,
        has_knowledge_base=has_knowledge_base,
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
                _save_turn(session_id, question, answer, resolved_mode, provider_name, db_path)
                workflow_steps = _build_workflow_steps(
                    mode,
                    resolved_mode,
                    "资料解析与 RAG 建库",
                    "资料解析失败，未调用问答生成。",
                )
                return {
                    "answer": answer,
                    "provider": provider_name,
                    "resolved_mode": resolved_mode,
                    "source_name": saved_path.name,
                    "chunk_count": "0",
                    "knowledge_status": "error",
                    "workflow_steps": workflow_steps,
                }

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
        answer = answer_with_rag(question, store_path=store_path, provider=provider)
        retrieval_results = retrieve_relevant_chunks(question, store_path=store_path, top_k=3)
        sources = _build_report_sources(retrieval_results)
        _save_turn(session_id, question, answer, resolved_mode, provider_name, db_path)
        workflow_steps = _build_workflow_steps(
            mode,
            resolved_mode,
            f"RAG 知识库问答（Provider: {provider_name}）",
            _result_detail("返回课程资料回答", sources),
        )
        return {
            "answer": answer,
            "provider": provider_name,
            "resolved_mode": resolved_mode,
            "sources": sources,
            "workflow_steps": workflow_steps,
            **metadata,
        }

    if resolved_mode == MODE_IMAGE:
        if uploaded_image is None:
            answer = "请先上传一张图片，再进行 YOLO 目标检测。"
            _save_turn(session_id, question, answer, resolved_mode, "yolo", db_path)
            workflow_steps = _build_workflow_steps(
                mode,
                resolved_mode,
                "YOLO 图片目标检测",
                "缺少图片输入，返回上传提示。",
            )
            return {
                "answer": answer,
                "provider": "yolo",
                "resolved_mode": resolved_mode,
                "workflow_steps": workflow_steps,
            }

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
        _save_turn(session_id, question, answer, resolved_mode, answer_provider, db_path)
        workflow_steps = _build_workflow_steps(
            mode,
            resolved_mode,
            f"YOLO 图片检测（Provider: {answer_provider}）",
            f"返回检测结果 {len(detection_result['detections'])} 个。",
        )
        return {
            "answer": answer,
            "provider": answer_provider,
            "resolved_mode": resolved_mode,
            "workflow_steps": workflow_steps,
            **detection_result,
        }

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
        _save_turn(session_id, question, answer, resolved_mode, provider_name, db_path)
        workflow_steps = _build_workflow_steps(
            mode,
            resolved_mode,
            f"RAG 检索 + {report.report_type}生成（Provider: {report.provider}）",
            _result_detail("返回可下载报告", sources),
        )
        return {
            "answer": answer,
            "provider": report.provider,
            "resolved_mode": resolved_mode,
            "report_type": report.report_type,
            "download_name": report.download_name,
            "knowledge_status": report.knowledge_status,
            "sources": sources,
            "workflow_steps": workflow_steps,
        }

    answer = chat_with_llm(question, provider=provider)
    _save_turn(session_id, question, answer, resolved_mode, provider_name, db_path)
    workflow_steps = _build_workflow_steps(
        mode,
        resolved_mode,
        f"LLM 普通问答（Provider: {provider_name}）",
        "返回通用问答结果。",
    )
    return {
        "answer": answer,
        "provider": provider_name,
        "resolved_mode": resolved_mode,
        "workflow_steps": workflow_steps,
    }


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


def _file_type(path: Path) -> str:
    return path.suffix.lower().lstrip(".") or "unknown"


def _save_turn(
    session_id: str,
    question: str,
    answer: str,
    mode: str,
    provider: str,
    db_path: str | Path,
) -> None:
    save_chat_record(session_id, "user", question, mode, provider, db_path=db_path)
    save_chat_record(session_id, "assistant", answer, mode, provider, db_path=db_path)
