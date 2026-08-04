from html import escape
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 读取项目根目录下的 .env 文件，让通用 LLM 配置生效

import streamlit as st

from config import UPLOAD_DIR, YOLO_CONFIDENCE_THRESHOLD
from modules.agent.router import route_user_request
from modules.database.history import list_chat_records
from modules.database.knowledge import (
    create_knowledge_base,
    delete_knowledge_base,
    delete_knowledge_base_file,
    ensure_default_knowledge_base,
    list_knowledge_base_files,
    list_knowledge_bases,
    save_knowledge_base_file,
    set_active_knowledge_base,
)
from modules.database.models import ChatRecord, ChatSession, KnowledgeBase, KnowledgeBaseFile
from modules.database.sessions import (
    create_session,
    delete_session,
    ensure_default_session,
    list_sessions,
    maybe_update_session_title,
    update_session_active_kb,
)
from modules.rag.loader import DocumentLoadError
from modules.rag.qa import build_knowledge_base
from modules.utils.file_utils import save_uploaded_file

SUPPORTED_DOC_TYPES = ["pdf", "txt", "md", "pptx"]
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
MODE_AUTO = "自动识别"
MODE_COURSE_QA = "课程资料问答"
MODE_IMAGE = "图片目标检测"
MODE_REPORT = "学习辅助生成"
MODE_CHAT = "普通问答"
LLM_PROVIDER_OPTIONS = {
    "自动": "auto",
    "云端 API": "cloud",
    "本地 fallback": "fallback",
    "本地模型": "ollama",
}
DEFAULT_STATUS = {
    "status": "empty",
    "source_name": "未上传",
    "chunk_count": "0",
    "provider": "auto",
}


def chat_records_to_messages(records: list[ChatRecord]) -> list[dict[str, str]]:
    return [{"role": record.role, "content": record.content} for record in records]


def format_session_label(session: ChatSession) -> str:
    return f"{session.updated_at.strftime('%H:%M')} · {_shorten(session.title, 18)}"


def knowledge_status_from_files(
    kb: KnowledgeBase | None,
    files: list[KnowledgeBaseFile],
) -> dict[str, str]:
    provider = st.session_state.get("llm_provider_name", "auto") if hasattr(st, "session_state") else "auto"
    if kb is None:
        return {**DEFAULT_STATUS, "provider": provider}

    indexed_files = [file for file in files if file.status == "indexed"]
    if not indexed_files:
        return {**DEFAULT_STATUS, "source_name": kb.name, "provider": provider}

    # Sidebar only needs aggregate health; individual traceability is shown below each answer.
    total_chunks = sum(file.chunk_count for file in indexed_files)
    return {
        "status": "ready",
        "source_name": f"{kb.name}（{len(indexed_files)} 份资料）",
        "chunk_count": str(total_chunks),
        "provider": provider,
    }


def init_session_state() -> None:
    active_kb = ensure_default_knowledge_base()
    session = ensure_default_session(active_kb_id=active_kb.kb_id)

    st.session_state.setdefault("active_kb_id", session.active_kb_id or active_kb.kb_id)
    st.session_state.setdefault("session_id", session.session_id)
    st.session_state.setdefault("last_indexed_upload", "")
    st.session_state.setdefault("llm_provider_name", "auto")
    if "messages" not in st.session_state:
        st.session_state.messages = load_session_messages(st.session_state.session_id)
    refresh_knowledge_status()


def load_session_messages(session_id: str) -> list[dict[str, str]]:
    records = list_chat_records(session_id=session_id, limit=200)
    return chat_records_to_messages(records)


def get_current_kb() -> KnowledgeBase:
    bases = list_knowledge_bases()
    if not bases:
        return ensure_default_knowledge_base()

    active_id = st.session_state.get("active_kb_id")
    for kb in bases:
        if kb.kb_id == active_id:
            return kb

    set_active_knowledge_base(bases[0].kb_id)
    st.session_state.active_kb_id = bases[0].kb_id
    return bases[0]


def refresh_knowledge_status() -> None:
    kb = get_current_kb() if "active_kb_id" in st.session_state else ensure_default_knowledge_base()
    st.session_state.knowledge_status = knowledge_status_from_files(
        kb,
        list_knowledge_base_files(kb.kb_id),
    )


def apply_compact_style() -> None:
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] .status-card {
            border: 1px solid rgba(160, 166, 180, 0.24);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 10px;
            background: rgba(255, 255, 255, 0.04);
        }
        section[data-testid="stSidebar"] .status-label {
            color: rgba(250, 250, 250, 0.68);
            font-size: 0.78rem;
            line-height: 1.2;
            margin-bottom: 4px;
        }
        section[data-testid="stSidebar"] .status-value {
            color: rgb(250, 250, 250);
            font-size: 0.98rem;
            font-weight: 650;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .stChatMessage pre {
            white-space: pre-wrap;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(mode: str) -> None:
    render_session_manager()
    render_model_selector()
    render_knowledge_manager()

    st.sidebar.divider()
    st.sidebar.header("功能")
    st.sidebar.caption("V1 先跑通 RAG 主流程，其他模块快速逐步接入。")
    if mode != MODE_COURSE_QA:
        st.sidebar.info("知识库已独立管理，切换模式不会丢失已上传资料。")


def clear_sidebar_selector_state(*keys: str) -> None:
    """Reset Streamlit widget-owned selection so programmatic navigation is visible after rerun."""
    for key in keys:
        st.session_state.pop(key, None)


def render_session_manager() -> None:
    st.sidebar.header("会话")
    sessions = list_sessions(limit=20)
    if not sessions:
        sessions = [create_session(active_kb_id=st.session_state.active_kb_id)]

    session_ids = [session.session_id for session in sessions]
    current_session_id = st.session_state.get("session_id")
    index = session_ids.index(current_session_id) if current_session_id in session_ids else 0
    selected_session_id = st.sidebar.selectbox(
        "历史会话",
        session_ids,
        index=index,
        format_func=lambda session_id: format_session_label(
            next(session for session in sessions if session.session_id == session_id)
        ),
        key="session_selector",
    )

    if selected_session_id != current_session_id:
        st.session_state.session_id = selected_session_id
        st.session_state.messages = load_session_messages(selected_session_id)
        st.rerun()

    if st.sidebar.button("新建会话", icon=":material/add:", width="stretch"):
        session = create_session(active_kb_id=st.session_state.active_kb_id)
        st.session_state.session_id = session.session_id
        st.session_state.messages = []
        clear_sidebar_selector_state("session_selector")
        st.rerun()

    if st.sidebar.button("删除当前会话", icon=":material/delete:", width="stretch"):
        delete_session(st.session_state.session_id)
        remaining_sessions = list_sessions(limit=20)
        next_session = (
            remaining_sessions[0]
            if remaining_sessions
            else create_session(active_kb_id=st.session_state.active_kb_id)
        )
        st.session_state.session_id = next_session.session_id
        st.session_state.messages = load_session_messages(next_session.session_id)
        clear_sidebar_selector_state("session_selector")
        st.rerun()


def render_model_selector() -> None:
    st.sidebar.divider()
    st.sidebar.header("模型")
    provider_values = list(LLM_PROVIDER_OPTIONS.values())
    current = st.session_state.get("llm_provider_name", "auto")
    index = provider_values.index(current) if current in provider_values else 0
    selected_label = st.sidebar.selectbox(
        "LLM Provider",
        list(LLM_PROVIDER_OPTIONS.keys()),
        index=index,
        help="自动：有 LLM_API_KEY 时调用云端兼容 API，否则使用本地 fallback。本地模型需要先启动 Ollama。",
    )
    st.session_state.llm_provider_name = LLM_PROVIDER_OPTIONS[selected_label]
    if st.session_state.llm_provider_name == "ollama":
        st.sidebar.caption("本地模型模式默认调用 Ollama 中配置的模型。")
    refresh_knowledge_status()


def render_knowledge_manager() -> None:
    st.sidebar.divider()
    st.sidebar.header("知识库")
    bases = list_knowledge_bases()
    active_kb = get_current_kb()
    kb_ids = [kb.kb_id for kb in bases]
    index = kb_ids.index(active_kb.kb_id) if active_kb.kb_id in kb_ids else 0

    selected_kb_id = st.sidebar.selectbox(
        "当前知识库",
        kb_ids,
        index=index,
        format_func=lambda kb_id: _knowledge_base_label(
            next(kb for kb in bases if kb.kb_id == kb_id)
        ),
        key="kb_selector",
    )
    if selected_kb_id != st.session_state.active_kb_id:
        switch_knowledge_base(selected_kb_id)

    with st.sidebar.form("create_kb_form", border=False):
        kb_name = st.text_input("新知识库名称", placeholder="例如：Dify 课程资料")
        submitted = st.form_submit_button("新建知识库", icon=":material/create_new_folder:")
    if submitted:
        kb = create_knowledge_base(kb_name, is_active=True)
        switch_knowledge_base(kb.kb_id)

    if st.sidebar.button("删除当前知识库", icon=":material/delete:", width="stretch"):
        delete_knowledge_base(st.session_state.active_kb_id)
        next_kb = ensure_default_knowledge_base()
        st.session_state.active_kb_id = next_kb.kb_id
        update_session_active_kb(st.session_state.session_id, next_kb.kb_id)
        clear_sidebar_selector_state("kb_selector")
        refresh_knowledge_status()
        st.rerun()

    active_kb = get_current_kb()
    files = list_knowledge_base_files(active_kb.kb_id)
    st.session_state.knowledge_status = knowledge_status_from_files(active_kb, files)
    render_status_section(files)
    render_upload_panel(active_kb, files)
    render_file_management(active_kb, files)


def switch_knowledge_base(kb_id: str) -> None:
    set_active_knowledge_base(kb_id)
    st.session_state.active_kb_id = kb_id
    update_session_active_kb(st.session_state.session_id, kb_id)
    clear_sidebar_selector_state("kb_selector")
    refresh_knowledge_status()
    st.rerun()


def render_status_section(files: list[KnowledgeBaseFile]) -> None:
    status = st.session_state.knowledge_status
    status_label = "已建库" if status["status"] == "ready" else "未建库"

    st.sidebar.subheader("RAG 状态")
    render_status_card("知识库", status_label)
    render_status_card("资料", _shorten(status["source_name"], max_length=18))
    render_status_card("切分片段", status["chunk_count"])
    render_status_card("LLM Provider", status["provider"])

    if files:
        with st.sidebar.expander("查看资料列表", expanded=False):
            for file in files:
                st.caption(f"{file.created_at.strftime('%H:%M')} · {file.status} · {file.chunk_count} 片段")
                st.write(_shorten(file.file_name, 30))


def render_upload_panel(active_kb: KnowledgeBase, files: list[KnowledgeBaseFile]) -> None:
    uploaded_file = st.sidebar.file_uploader(
        "上传资料到当前知识库",
        type=SUPPORTED_DOC_TYPES,
        key=f"kb_upload_{active_kb.kb_id}",
    )
    if uploaded_file is None:
        return

    upload_fingerprint = f"{active_kb.kb_id}:{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.last_indexed_upload == upload_fingerprint:
        return

    try:
        build_info = index_uploaded_file(active_kb, uploaded_file, append=bool(files))
    except DocumentLoadError as exc:
        st.sidebar.error(f"资料解析失败：{exc}")
        return

    st.session_state.last_indexed_upload = upload_fingerprint
    st.sidebar.success(f"已索引 {build_info['source_name']}，新增 {build_info['chunk_count']} 个片段。")
    refresh_knowledge_status()
    st.rerun()


def index_uploaded_file(
    active_kb: KnowledgeBase,
    uploaded_file: object,
    append: bool,
) -> dict[str, object]:
    # Each knowledge base owns a physical upload folder so later rebuilds never depend on UI state.
    saved_path = save_uploaded_file(uploaded_file, UPLOAD_DIR / active_kb.kb_id)
    build_info = build_knowledge_base(saved_path, store_path=active_kb.store_path, append=append)
    save_knowledge_base_file(
        kb_id=active_kb.kb_id,
        file_name=saved_path.name,
        file_type=_file_type(saved_path),
        storage_path=saved_path,
        chunk_count=int(build_info["chunk_count"]),
        status="indexed",
    )
    return build_info


def render_file_management(active_kb: KnowledgeBase, files: list[KnowledgeBaseFile]) -> None:
    if not files:
        return

    with st.sidebar.expander("管理资料", expanded=False):
        if st.button("重新建库", icon=":material/refresh:", width="stretch"):
            try:
                chunk_count = rebuild_knowledge_base(active_kb, files)
            except DocumentLoadError as exc:
                st.error(f"重建失败：{exc}")
            else:
                st.success(f"已重建 {chunk_count} 个片段。")
                refresh_knowledge_status()
                st.rerun()

        file_ids = [file.id for file in files]
        selected_file_id = st.selectbox(
            "删除资料",
            file_ids,
            format_func=lambda file_id: next(file.file_name for file in files if file.id == file_id),
            key=f"delete_file_{active_kb.kb_id}",
        )
        if st.button("删除所选资料并重建", icon=":material/delete:", width="stretch"):
            delete_knowledge_base_file(selected_file_id)
            remaining_files = [file for file in files if file.id != selected_file_id]
            rebuild_knowledge_base(active_kb, remaining_files)
            refresh_knowledge_status()
            st.success("已删除所选资料并更新索引。")
            st.rerun()


def rebuild_knowledge_base(
    active_kb: KnowledgeBase,
    files: list[KnowledgeBaseFile],
) -> int:
    store_path = Path(active_kb.store_path)
    if store_path.exists():
        store_path.unlink()
    if not files:
        return 0

    total_chunks = 0
    for index, file in enumerate(files):
        build_info = build_knowledge_base(
            Path(file.storage_path),
            store_path=store_path,
            append=index > 0,
        )
        total_chunks += int(build_info["chunk_count"])
    return total_chunks


def render_status_card(label: str, value: str) -> None:
    st.sidebar.markdown(
        f"""
        <div class="status-card">
            <div class="status-label">{escape(label)}</div>
            <div class="status-value">{escape(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _knowledge_base_label(kb: KnowledgeBase) -> str:
    suffix = " · 当前" if kb.is_active else ""
    return f"{kb.name}{suffix}"


def _shorten(value: str, max_length: int) -> str:
    if len(value) <= max_length:
        return value
    return f"{value[:max_length - 1]}..."


def _file_type(path: Path) -> str:
    return path.suffix.lower().lstrip(".") or "unknown"


# Report generation can also return a ready knowledge status, so sidebar refreshes
# must require indexing-specific fields instead of trusting one shared flag.
def is_knowledge_base_build_result(result: dict[str, object]) -> bool:
    return (
        result.get("knowledge_status") == "ready"
        and bool(result.get("source_name"))
        and "chunk_count" in result
    )


def get_report_knowledge_notice(
    knowledge_status: str | None,
    sources: list[dict[str, object]],
) -> tuple[str, str]:
    if knowledge_status == "ready" and sources:
        return "success", f"本次报告参考了 {len(sources)} 个知识库片段。"
    return "warning", "本次报告未使用知识库资料，当前结果为通用草稿。"

def render_workflow_steps(steps: list[dict[str, object]], title: str = "Agent 工作流") -> None:
    if not steps:
        return
    with st.expander(title):
        for index, step in enumerate(steps, start=1):
            step_title = str(step.get("title", f"步骤 {index}"))
            detail = str(step.get("detail", ""))
            st.markdown(f"**{index}. {step_title}**")
            if detail:
                st.caption(detail)


def render_sources(sources: list[dict[str, object]], title: str = "查看引用来源") -> None:
    if not sources:
        return
    with st.expander(title):
        for source in sources:
            source_name = str(source.get("source_name", "未知资料"))
            chunk_id = str(source.get("chunk_id", "?"))
            score = source.get("score")
            suffix = f" · 相关度 {score}" if score is not None else ""
            st.write(f"来源：{source_name} · 片段 {chunk_id}{suffix}")


def render_report_sources(
    knowledge_status: str | None,
    sources: list[dict[str, object]],
) -> None:
    kind, message = get_report_knowledge_notice(knowledge_status, sources)
    if kind == "success":
        st.success(message)
    else:
        st.warning(message)
        return
    render_sources(sources)


def update_knowledge_status(result: dict[str, object]) -> None:
    if not is_knowledge_base_build_result(result):
        return
    st.session_state.knowledge_status = {
        "status": str(result["knowledge_status"]),
        "source_name": str(result.get("source_name", "未上传")),
        "chunk_count": str(result.get("chunk_count", "0")),
        "provider": str(result.get("provider", st.session_state.get("llm_provider_name", "auto"))),
    }


def render_chat_history() -> None:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("image_path"):
                st.image(message["image_path"], caption="检测结果")
            render_workflow_steps(message.get("workflow_steps", []))
            if message.get("download_name"):
                render_report_sources(
                    message.get("knowledge_status"),
                    message.get("sources", []),
                )
                render_report_download(message["content"], message["download_name"])
            elif message.get("sources"):
                render_sources(message.get("sources", []))


def render_report_download(content: str, download_name: str) -> None:
    st.download_button(
        "下载报告",
        data=content,
        file_name=download_name,
        mime="text/markdown",
        icon=":material/download:",
    )


def main() -> None:
    st.set_page_config(page_title="AI Multimodal Learning Assistant", layout="wide")
    init_session_state()
    apply_compact_style()

    st.title("AI 多模态学习助手")

    mode = st.sidebar.radio(
        "选择模式",
        [MODE_AUTO, MODE_COURSE_QA, MODE_IMAGE, MODE_REPORT, MODE_CHAT],
    )
    render_sidebar(mode)

    active_kb = get_current_kb()
    st.info("资料上传已移到侧边栏知识库区域；模型可在侧边栏切换，没有 API Key 时会自动 fallback。")

    uploaded_image = None
    confidence_threshold = YOLO_CONFIDENCE_THRESHOLD
    report_type = "学习报告"
    learning_type = "课程总结"
    format_requirements = ""

    if mode == MODE_COURSE_QA and st.session_state.knowledge_status["status"] != "ready":
        st.warning("请先在侧边栏上传课程资料，再在下方输入问题。")
    elif mode == MODE_IMAGE:
        confidence_threshold = st.slider(
            "置信度阈值",
            min_value=0.1,
            max_value=0.9,
            value=float(YOLO_CONFIDENCE_THRESHOLD),
            step=0.05,
            help="阈值越高，误检越少；阈值越低，能检测出的目标越多。",
        )
        uploaded_image = st.file_uploader("上传图片", type=SUPPORTED_IMAGE_TYPES)
        if uploaded_image is None:
            st.warning("请先上传图片，再在下方输入检测需求。")
        else:
            st.image(uploaded_image, caption="待检测图片", width=360)
    elif mode == MODE_REPORT:
        learning_type = st.segmented_control(
            "生成类型",
            ["课程总结", "复习提纲", "报告大纲", "PPT 提纲", "实验报告"],
            default="课程总结",
        )
        report_type = "实验报告" if learning_type == "实验报告" else "学习报告"
        if learning_type == "实验报告":
            format_requirements = st.text_area(
                "实验报告格式要求",
                placeholder="例如：请按实验目的、实验环境、实验步骤、结果分析、总结五部分生成。",
            )
            st.info("实验报告会优先按你填写的格式要求生成；未填写时使用默认实验报告结构。")
        else:
            st.info(f"{learning_type} 会使用专用 Prompt，并尽量结合当前知识库资料。")

    render_chat_history()

    question = st.chat_input("请输入你的问题", submit_mode="disable")
    if question:
        maybe_update_session_title(st.session_state.session_id, question)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.spinner("正在检索资料并生成回答..."):
            result = route_user_request(
                question=question,
                mode=mode,
                uploaded_file=None,
                uploaded_image=uploaded_image,
                confidence_threshold=confidence_threshold,
                report_type=report_type,
                learning_type=learning_type,
                format_requirements=format_requirements,
                store_path=active_kb.store_path,
                session_id=st.session_state.session_id,
                has_knowledge_base=st.session_state.knowledge_status["status"] == "ready",
                llm_provider_name=st.session_state.llm_provider_name,
            )

        update_knowledge_status(result)
        provider = result.get("provider", "fallback")
        resolved_mode = result.get("resolved_mode")
        answer = result["answer"]
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "image_path": result.get("annotated_image_path"),
            "download_name": result.get("download_name"),
            "knowledge_status": result.get("knowledge_status"),
            "sources": result.get("sources", []),
            "workflow_steps": result.get("workflow_steps", []),
        }
        st.session_state.messages.append(assistant_message)

        with st.chat_message("assistant"):
            caption = f"Provider: {provider}"
            if resolved_mode:
                caption = f"{caption} · 模式: {resolved_mode}"
            st.caption(caption)
            st.markdown(answer)
            if assistant_message.get("image_path"):
                st.image(assistant_message["image_path"], caption="检测结果")
            render_workflow_steps(assistant_message.get("workflow_steps", []))
            if assistant_message.get("download_name"):
                render_report_sources(
                    assistant_message.get("knowledge_status"),
                    assistant_message.get("sources", []),
                )
                render_report_download(answer, assistant_message["download_name"])
            elif assistant_message.get("sources"):
                render_sources(assistant_message.get("sources", []))

        if is_knowledge_base_build_result(result):
            st.rerun()


if __name__ == "__main__":
    main()
