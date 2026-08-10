import sqlite3
import sys
from html import escape
from pathlib import Path

from dotenv import load_dotenv

if "unittest" not in sys.modules:
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
    update_session_title,
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
MODE_DESCRIPTIONS = {
    MODE_AUTO: "自动判断问题类型并选择合适能力",
    MODE_COURSE_QA: "基于上传资料回答问题",
    MODE_IMAGE: "上传图片识别图中物体",
    MODE_REPORT: "生成复习提纲、练习题等",
    MODE_CHAT: "直接对话，不检索知识库",
}
CHAT_INPUT_LIMIT = 1200
CHAT_PLACEHOLDERS = {
    MODE_AUTO: "输入学习问题，AI 会自动选择合适模式…",
    MODE_COURSE_QA: "输入关于课程资料的问题…",
    MODE_IMAGE: "上传图片并描述检测需求…",
    MODE_REPORT: "例如：生成 CPU 章节的复习提纲…",
    MODE_CHAT: "直接输入你的问题…",
}
NO_RETRIEVAL_NOTICE = "当前知识库中没有检索到相关片段，请先上传课程资料或换一种问法。"
LLM_PROVIDER_OPTIONS = {
    "自动（推荐）": "auto",
    "云端模型": "cloud",
    "本地 Ollama": "ollama",
    "离线兜底（无模型）": "fallback",
}
DEFAULT_STATUS = {
    "status": "empty",
    "source_name": "未上传",
    "chunk_count": "0",
    "provider": "auto",
}


def chat_records_to_messages(records: list[ChatRecord]) -> list[dict[str, object]]:
    messages = []
    for record in records:
        message = {"role": record.role, "content": record.content}
        if record.metadata:
            message.update(record.metadata)
        messages.append(message)
    return messages


def build_conversation_history(
    messages: list[dict[str, object]],
    limit: int = 8,
) -> list[dict[str, str]]:
    history = []
    for message in messages:
        role = message.get("role")
        content = str(message.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        history.append({"role": str(role), "content": content})
    return history[-limit:]


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
    st.session_state.setdefault("system_notice", "")
    st.session_state.setdefault("pending_question", "")
    if "messages" not in st.session_state:
        st.session_state.messages = load_session_messages(st.session_state.session_id)
    refresh_knowledge_status()


def load_session_messages(session_id: str) -> list[dict[str, object]]:
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
            border: 1px solid rgba(148, 163, 184, 0.32);
            border-radius: 8px;
            padding: 10px 12px;
            margin-bottom: 10px;
            background: rgba(255, 255, 255, 0.72);
        }
        section[data-testid="stSidebar"] .status-label {
            color: rgb(100, 116, 139);
            font-size: 0.78rem;
            line-height: 1.2;
            margin-bottom: 4px;
        }
        section[data-testid="stSidebar"] .status-value {
            color: rgb(15, 23, 42);
            font-size: 0.98rem;
            font-weight: 650;
            line-height: 1.35;
            overflow-wrap: anywhere;
        }
        .stChatMessage pre {
            white-space: pre-wrap;
            word-break: break-word;
        }
        /* 让高级设置里的 LLM Provider 选项完整可见 */
        section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stVerticalBlock"] {
            flex: 1 1 100% !important;
            width: 100% !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_mode_selector() -> str:
    st.sidebar.header("模式")
    return st.sidebar.radio(
        "选择模式",
        [MODE_AUTO, MODE_COURSE_QA, MODE_IMAGE, MODE_REPORT, MODE_CHAT],
        captions=[MODE_DESCRIPTIONS[mode] for mode in [MODE_AUTO, MODE_COURSE_QA, MODE_IMAGE, MODE_REPORT, MODE_CHAT]],
        key="mode_selector",
        width="stretch",
    )


def render_sidebar() -> None:
    with st.sidebar.expander("高级设置", icon=":material/tune:", expanded=True):
        render_model_selector()
    render_session_manager()
    render_knowledge_manager()


def clear_sidebar_selector_state(*keys: str) -> None:
    """Reset Streamlit widget-owned selection so programmatic navigation is visible after rerun."""
    for key in keys:
        st.session_state.pop(key, None)


def delete_current_session() -> None:
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


@st.dialog("删除当前会话")
def confirm_delete_current_session_dialog() -> None:
    st.warning("此操作会删除当前会话及其聊天记录，无法撤销。", icon=":material/warning:")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("取消", key="cancel_delete_current_session"):
            st.rerun()
        if st.button(
            "确认删除",
            key="confirm_delete_current_session",
            icon=":material/delete:",
            type="primary",
        ):
            try:
                delete_current_session()
            except sqlite3.Error as exc:
                st.error(format_database_error("删除会话", exc))
            else:
                st.rerun()


@st.dialog("删除会话")
def confirm_delete_session_dialog(session: ChatSession) -> None:
    st.warning(f"将删除「{session.title}」及其中的聊天记录，无法撤销。", icon=":material/warning:")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("取消", key=f"cancel_delete_session_{session.session_id}"):
            st.rerun()
        if st.button(
            "确认删除",
            key=f"confirm_delete_session_{session.session_id}",
            icon=":material/delete:",
            type="primary",
        ):
            try:
                delete_session(session.session_id)
                if session.session_id == st.session_state.get("session_id"):
                    remaining_sessions = list_sessions(limit=20)
                    next_session = (
                        remaining_sessions[0]
                        if remaining_sessions
                        else create_session(active_kb_id=st.session_state.active_kb_id)
                    )
                    st.session_state.session_id = next_session.session_id
                    st.session_state.messages = load_session_messages(next_session.session_id)
            except sqlite3.Error as exc:
                st.error(format_database_error("删除会话", exc))
            else:
                clear_sidebar_selector_state("session_selector")
                st.rerun()


@st.dialog("重命名会话")
def rename_session_dialog(session: ChatSession) -> None:
    new_title = st.text_input("会话名称", value=session.title, key=f"rename_session_{session.session_id}")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("取消", key=f"cancel_rename_session_{session.session_id}"):
            st.rerun()
        if st.button("保存", key=f"save_rename_session_{session.session_id}", icon=":material/check:"):
            try:
                update_session_title(session.session_id, new_title)
            except sqlite3.Error as exc:
                st.error(format_database_error("重命名会话", exc))
            else:
                st.rerun()


def render_session_manager() -> None:
    st.sidebar.header("会话")
    sessions = list_sessions(limit=20)
    if not sessions:
        sessions = [create_session(active_kb_id=st.session_state.active_kb_id)]

    current_session_id = st.session_state.get("session_id")

    if st.sidebar.button("新建会话", icon=":material/add:", width="stretch"):
        session = create_session(active_kb_id=st.session_state.active_kb_id)
        st.session_state.session_id = session.session_id
        st.session_state.messages = []
        clear_sidebar_selector_state("session_selector")
        st.rerun()

    if st.sidebar.button(
        "删除当前会话",
        icon=":material/delete:",
        width="stretch",
    ):
        confirm_delete_current_session_dialog()

    st.sidebar.caption("历史会话")
    with st.sidebar.container(height=260, border=True, gap="xsmall"):
        for session in sessions:
            is_current = session.session_id == current_session_id
            cols = st.columns([7, 1, 1], vertical_alignment="center")
            if cols[0].button(
                format_session_label(session),
                key=f"open_session_{session.session_id}",
                width="stretch",
                type="primary" if is_current else "secondary",
            ):
                if not is_current:
                    st.session_state.session_id = session.session_id
                    st.session_state.messages = load_session_messages(session.session_id)
                    st.rerun()
            if cols[1].button(
                "",
                icon=":material/edit:",
                key=f"edit_session_{session.session_id}",
                help="重命名会话",
            ):
                rename_session_dialog(session)
            if cols[2].button(
                "",
                icon=":material/delete:",
                key=f"delete_session_{session.session_id}",
                help="删除会话",
            ):
                confirm_delete_session_dialog(session)


def render_model_selector() -> None:
    provider_values = list(LLM_PROVIDER_OPTIONS.values())
    current = st.session_state.get("llm_provider_name", "auto")
    index = provider_values.index(current) if current in provider_values else 0
    st.caption("选择回答问题所用的大模型来源。")
    selected_label = st.radio(
        "LLM Provider",
        list(LLM_PROVIDER_OPTIONS.keys()),
        index=index,
        help=(
            "自动：优先云端模型，无 API Key 时用本地 Ollama，都不可用才离线兜底。\n"
            "云端模型：需要 LLM_API_KEY。\n"
            "本地 Ollama：需要先启动 Ollama 并拉取模型。\n"
            "离线兜底：不调用任何大模型，仅返回检索片段，适合无网演示。"
        ),
    )
    st.session_state.llm_provider_name = LLM_PROVIDER_OPTIONS[selected_label]
    if st.session_state.llm_provider_name == "ollama":
        st.caption("本地 Ollama 模式：请确认已运行 `ollama serve` 并拉取模型。")
    refresh_knowledge_status()


def render_knowledge_manager() -> None:
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

    active_kb = get_current_kb()
    files = list_knowledge_base_files(active_kb.kb_id)
    st.session_state.knowledge_status = knowledge_status_from_files(active_kb, files)
    with st.sidebar.expander("知识库管理", icon=":material/database:", expanded=False):
        with st.form("create_kb_form", border=False):
            kb_name = st.text_input("新知识库名称", placeholder="例如：Dify 课程资料")
            submitted = st.form_submit_button("新建知识库", icon=":material/create_new_folder:")
        if submitted:
            kb = create_knowledge_base(kb_name, is_active=True)
            switch_knowledge_base(kb.kb_id)

        if st.button(
            "删除当前知识库",
            icon=":material/delete:",
            width="stretch",
        ):
            confirm_delete_knowledge_base_dialog(active_kb)

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


@st.dialog("删除当前知识库")
def confirm_delete_knowledge_base_dialog(kb: KnowledgeBase) -> None:
    st.warning(f"将删除知识库「{kb.name}」及其上传资料和索引文件，无法撤销。", icon=":material/warning:")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("取消", key=f"cancel_delete_kb_{kb.kb_id}"):
            st.rerun()
        if st.button(
            "确认删除",
            key=f"confirm_delete_kb_{kb.kb_id}",
            icon=":material/delete:",
            type="primary",
        ):
            try:
                delete_knowledge_base(kb.kb_id)
                next_kb = ensure_default_knowledge_base()
                update_session_active_kb(st.session_state.session_id, next_kb.kb_id)
            except sqlite3.Error as exc:
                st.error(format_database_error("删除知识库", exc))
            except OSError as exc:
                st.error(f"删除知识库失败：文件系统暂时不可写，请检查资料文件权限后重试。详情：{exc}")
            else:
                st.session_state.active_kb_id = next_kb.kb_id
                clear_sidebar_selector_state("kb_selector")
                refresh_knowledge_status()
                st.rerun()


@st.dialog("删除资料并重建")
def confirm_delete_file_dialog(active_kb: KnowledgeBase, files: list[KnowledgeBaseFile], selected_file_id: int) -> None:
    selected_file = next(file for file in files if file.id == selected_file_id)
    st.warning(f"将删除「{selected_file.file_name}」并重建当前知识库索引，无法撤销。", icon=":material/warning:")
    with st.container(horizontal=True, horizontal_alignment="right"):
        if st.button("取消", key=f"cancel_delete_file_{selected_file_id}"):
            st.rerun()
        if st.button(
            "确认删除",
            key=f"confirm_delete_file_{selected_file_id}",
            icon=":material/delete:",
            type="primary",
        ):
            try:
                delete_knowledge_base_file(selected_file_id)
                remaining_files = [file for file in files if file.id != selected_file_id]
                rebuild_knowledge_base(active_kb, remaining_files)
            except sqlite3.Error as exc:
                st.error(format_database_error("删除资料", exc))
            except OSError as exc:
                st.error(f"删除资料失败：文件系统暂时不可写，请检查资料文件权限后重试。详情：{exc}")
            else:
                refresh_knowledge_status()
                st.success("已删除所选资料并更新索引。")
                st.rerun()


def render_status_section(files: list[KnowledgeBaseFile]) -> None:
    status = st.session_state.knowledge_status
    status_label = "已建库" if status["status"] == "ready" else "未建库"

    st.subheader("RAG 状态")
    render_status_card("知识库", status_label)
    render_status_card("资料", _shorten(status["source_name"], max_length=18))
    render_status_card("切分片段", status["chunk_count"])
    render_status_card("LLM Provider", status["provider"])

    if files:
        with st.expander("查看资料列表", expanded=False):
            for file in files:
                st.caption(f"{file.created_at.strftime('%H:%M')} · {file.status} · {file.chunk_count} 片段")
                st.write(_shorten(file.file_name, 30))


def render_upload_panel(active_kb: KnowledgeBase, files: list[KnowledgeBaseFile]) -> None:
    uploaded_file = st.file_uploader(
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
        st.error(f"资料解析失败：{exc}")
        return

    st.session_state.last_indexed_upload = upload_fingerprint
    st.success(f"已索引 {build_info['source_name']}，新增 {build_info['chunk_count']} 个片段。")
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

    with st.expander("管理资料", expanded=False):
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
        if st.button(
            "删除所选资料并重建",
            icon=":material/delete:",
            width="stretch",
        ):
            confirm_delete_file_dialog(active_kb, files, selected_file_id)


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
    st.markdown(
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


def format_database_error(action: str, exc: sqlite3.Error) -> str:
    detail = str(exc)
    if "readonly" in detail.lower():
        return f"{action}失败：数据库暂时不可写，请关闭占用该数据库的程序或检查文件权限后重试。"
    return f"{action}失败：数据库操作未完成。详情：{detail}"


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

def render_workflow_steps(steps: list[dict[str, object]], title: str = "查看推理过程") -> None:
    if not steps:
        return
    with st.expander(title, icon=":material/account_tree:"):
        for index, step in enumerate(steps, start=1):
            step_title = str(step.get("title", f"步骤 {index}"))
            detail = str(step.get("detail", ""))
            st.markdown(f"**{index}. {step_title}**")
            if detail:
                st.caption(detail)


def render_sources(sources: list[dict[str, object]], title: str = "查看引用来源") -> None:
    if not sources:
        return
    with st.expander(f"{title} · {len(sources)} 篇引用", icon=":material/article:"):
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
    for index, message in enumerate(st.session_state.messages):
        if is_system_notice_message(message):
            st.warning(str(message["content"]), icon=":material/warning:")
            continue
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("image_path"):
                st.image(message["image_path"], caption="检测结果")
            if message.get("sources"):
                st.caption(f":material/article: {len(message.get('sources', []))} 篇引用")
            if should_show_workflow_steps(message):
                render_workflow_steps(message.get("workflow_steps", []))
            if message.get("download_name"):
                render_report_sources(
                    message.get("knowledge_status"),
                    message.get("sources", []),
                )
                render_report_download(message["content"], message["download_name"])
            elif message.get("sources"):
                render_sources(message.get("sources", []))
            if message["role"] == "assistant":
                render_regenerate_button(index)


def is_system_notice_message(message: dict[str, object]) -> bool:
    return message.get("role") == "assistant" and str(message.get("content", "")) == NO_RETRIEVAL_NOTICE


def should_show_workflow_steps(message: dict[str, object]) -> bool:
    steps = message.get("workflow_steps", [])
    if not steps:
        return False
    first_detail = str(steps[0].get("detail", "")) if isinstance(steps, list) and steps else ""
    return "自动识别" in first_detail


def render_regenerate_button(message_index: int) -> None:
    previous_question = ""
    for message in reversed(st.session_state.messages[:message_index]):
        if message.get("role") == "user":
            previous_question = str(message.get("content", ""))
            break
    if not previous_question:
        return
    if st.button(
        "",
        icon=":material/refresh:",
        key=f"regenerate_{message_index}",
        help="重新生成",
    ):
        st.session_state.pending_question = previous_question
        st.rerun()


def select_mode(mode: str) -> None:
    st.session_state.mode_selector = mode


def render_empty_state() -> None:
    if st.session_state.messages:
        return
    st.subheader("从一个任务开始")
    cards = st.columns(3)
    card_specs = [
        (MODE_COURSE_QA, "课程问答", "上传课程资料后，提问并查看引用来源。", ":material/article:"),
        (MODE_IMAGE, "图片检测", "上传图片，识别图中的关键目标。", ":material/image_search:"),
        (MODE_REPORT, "复习提纲生成", "把章节内容整理成提纲、练习题或报告。", ":material/edit_note:"),
    ]
    for column, (mode, title, body, icon) in zip(cards, card_specs, strict=False):
        with column.container(border=True, height=160):
            st.markdown(f"{icon} **{title}**")
            st.caption(body)
            st.button(
                "进入",
                key=f"empty_state_{mode}",
                icon=":material/arrow_forward:",
                on_click=select_mode,
                args=(mode,),
            )


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
    st.caption("上传课程资料 → 提问 → 获取带引用的答案。试试问我：生成 CPU 章节的复习提纲。")

    mode = render_mode_selector()
    render_sidebar()

    active_kb = get_current_kb()

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

    render_empty_state()
    render_chat_history()

    notice_slot = st.empty()
    if st.session_state.get("system_notice"):
        notice_slot.warning(st.session_state.system_notice, icon=":material/warning:")

    st.caption(f"最多 {CHAT_INPUT_LIMIT} 字，Enter 发送，Shift+Enter 换行。")
    question = st.chat_input(
        CHAT_PLACEHOLDERS.get(mode, "请输入你的问题…"),
        key="chat_input",
        max_chars=CHAT_INPUT_LIMIT,
        submit_mode="stop",
    )
    question = question or st.session_state.pop("pending_question", "")
    if question:
        maybe_update_session_title(st.session_state.session_id, question)
        conversation_history = build_conversation_history(st.session_state.messages)
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
                conversation_history=conversation_history,
            )

        update_knowledge_status(result)
        if result.get("system_notice"):
            st.session_state.system_notice = str(result["system_notice"])
            notice_slot.warning(st.session_state.system_notice, icon=":material/warning:")
            return
        st.session_state.system_notice = ""
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
            if assistant_message.get("sources"):
                st.caption(f":material/article: {len(assistant_message.get('sources', []))} 篇引用")
            if should_show_workflow_steps(assistant_message):
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
