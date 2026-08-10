import tempfile
import unittest
import sqlite3
from contextlib import closing
from pathlib import Path


class ChatHistoryTests(unittest.TestCase):
    def test_save_and_list_chat_records(self):
        from modules.database.history import list_chat_records, save_chat_record

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "history.db"

            save_chat_record(
                session_id="session-1",
                role="user",
                content="什么是 RAG？",
                mode="课程资料问答",
                provider="fallback",
                db_path=db_path,
            )
            save_chat_record(
                session_id="session-1",
                role="assistant",
                content="RAG 是检索增强生成。",
                mode="课程资料问答",
                provider="fallback",
                db_path=db_path,
            )

            records = list_chat_records(session_id="session-1", db_path=db_path)

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0].role, "user")
            self.assertEqual(records[1].content, "RAG 是检索增强生成。")
            self.assertEqual(records[1].provider, "fallback")

    def test_route_user_request_persists_user_and_assistant_records(self):
        from modules.agent.router import route_user_request
        from modules.database.history import list_chat_records

        class UploadedFile:
            name = "rag.txt"

            def getbuffer(self):
                return "RAG 是检索增强生成。".encode("utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            db_path = tmp_path / "history.db"

            route_user_request(
                question="什么是 RAG？",
                mode="课程资料问答",
                uploaded_file=UploadedFile(),
                upload_dir=tmp_path / "uploads",
                store_path=tmp_path / "rag_store.json",
                db_path=db_path,
                session_id="session-2",
                llm_provider_name="fallback",
            )

            records = list_chat_records(session_id="session-2", db_path=db_path)

            self.assertEqual([record.role for record in records], ["user", "assistant"])
            self.assertEqual(records[0].content, "什么是 RAG？")
            self.assertIn("RAG 是检索增强生成", records[1].content)

            self.assertEqual(len(records[1].metadata["workflow_steps"]), 3)
            self.assertEqual(records[1].metadata["sources"][0]["source_name"], "rag.txt")

    def test_list_chat_records_returns_recent_records_for_session(self):
        from modules.database.history import list_chat_records, save_chat_record

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "history.db"
            for index in range(5):
                save_chat_record(
                    session_id="session-3",
                    role="user",
                    content=f"问题 {index}",
                    mode="普通问答",
                    provider="fallback",
                    db_path=db_path,
                )

            records = list_chat_records(session_id="session-3", db_path=db_path, limit=2)

            self.assertEqual([record.content for record in records], ["问题 3", "问题 4"])
    def test_chat_record_metadata_round_trips_to_streamlit_messages(self):
        from app import chat_records_to_messages
        from modules.database.history import list_chat_records, save_chat_record

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "history.db"
            save_chat_record(
                session_id="session-meta",
                role="assistant",
                content="answer",
                mode="report",
                provider="fallback",
                metadata={
                    "workflow_steps": [{"title": "Retrieve", "detail": "Found 1 source"}],
                    "sources": [{"source_name": "course.md", "chunk_id": 1, "score": 0.7}],
                    "download_name": "report.md",
                    "knowledge_status": "ready",
                },
                db_path=db_path,
            )

            records = list_chat_records(session_id="session-meta", db_path=db_path)
            messages = chat_records_to_messages(records)

            self.assertEqual(messages[0]["content"], "answer")
            self.assertEqual(messages[0]["workflow_steps"][0]["title"], "Retrieve")
            self.assertEqual(messages[0]["sources"][0]["source_name"], "course.md")
            self.assertEqual(messages[0]["download_name"], "report.md")
            self.assertEqual(messages[0]["knowledge_status"], "ready")

    def test_list_chat_records_does_not_migrate_legacy_database_on_read(self):
        from modules.database.history import list_chat_records

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "legacy.db"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE chat_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        provider TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO chat_history(session_id, role, content, mode, provider, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    ("legacy-session", "assistant", "old answer", "chat", "fallback", "2026-08-10T10:00:00"),
                )
                conn.commit()

            records = list_chat_records(session_id="legacy-session", db_path=db_path)

            self.assertEqual(records[0].content, "old answer")
            self.assertIsNone(records[0].metadata)
            with closing(sqlite3.connect(db_path)) as conn:
                columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(chat_history)").fetchall()}
            self.assertNotIn("metadata_json", columns)

    def test_list_chat_records_returns_empty_without_creating_database(self):
        from modules.database.history import list_chat_records

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "missing.db"

            records = list_chat_records(session_id="missing-session", db_path=db_path)

            self.assertEqual(records, [])
            self.assertFalse(db_path.exists())


if __name__ == "__main__":
    unittest.main()
