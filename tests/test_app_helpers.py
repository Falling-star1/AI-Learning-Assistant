import unittest
from datetime import datetime

from modules.database.models import ChatRecord, ChatSession, KnowledgeBase, KnowledgeBaseFile


class AppHelperTests(unittest.TestCase):
    def test_chat_records_to_messages_restores_chronological_messages(self):
        from app import chat_records_to_messages

        now = datetime.now()
        records = [
            ChatRecord(1, "s1", "user", "什么是 RAG？", "课程资料问答", "user", now),
            ChatRecord(2, "s1", "assistant", "RAG 是检索增强生成。", "课程资料问答", "fallback", now),
        ]

        messages = chat_records_to_messages(records)

        self.assertEqual(
            messages,
            [
                {"role": "user", "content": "什么是 RAG？"},
                {"role": "assistant", "content": "RAG 是检索增强生成。"},
            ],
        )

    def test_build_conversation_history_keeps_recent_text_turns_only(self):
        from app import build_conversation_history

        messages = [
            {"role": "assistant", "content": "旧回答"},
            {"role": "tool", "content": "不应进入上下文"},
            {"role": "user", "content": "CPU 是什么？"},
            {"role": "assistant", "content": "CPU 是中央处理器。", "sources": [{"source_name": "cpu.md"}]},
            {"role": "assistant", "content": ""},
        ]

        history = build_conversation_history(messages, limit=2)

        self.assertEqual(
            history,
            [
                {"role": "user", "content": "CPU 是什么？"},
                {"role": "assistant", "content": "CPU 是中央处理器。"},
            ],
        )

    def test_knowledge_status_from_files_counts_indexed_chunks(self):
        from app import knowledge_status_from_files

        now = datetime.now()
        kb = KnowledgeBase("kb_1", "Dify 课程", "store.json", True, now, now)
        files = [
            KnowledgeBaseFile(1, "kb_1", "a.pdf", "pdf", "a.pdf", 3, "indexed", now),
            KnowledgeBaseFile(2, "kb_1", "b.pdf", "pdf", "b.pdf", 4, "indexed", now),
        ]

        status = knowledge_status_from_files(kb, files)

        self.assertEqual(status["status"], "ready")
        self.assertEqual(status["source_name"], "Dify 课程（2 份资料）")
        self.assertEqual(status["chunk_count"], "7")

    def test_format_session_label_uses_title_and_update_time(self):
        from app import format_session_label

        updated_at = datetime(2026, 7, 31, 9, 30, 0)
        session = ChatSession("session_1", "解释 YOLO", "kb_1", updated_at, updated_at)

        self.assertEqual(format_session_label(session), "09:30 · 解释 YOLO")

    def test_format_database_error_for_readonly_database(self):
        import sqlite3

        from app import format_database_error

        message = format_database_error("删除资料", sqlite3.OperationalError("attempt to write a readonly database"))

        self.assertIn("删除资料失败", message)
        self.assertIn("数据库暂时不可写", message)


if __name__ == "__main__":
    unittest.main()
