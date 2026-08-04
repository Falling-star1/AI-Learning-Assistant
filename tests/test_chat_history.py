import tempfile
import unittest
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
            )

            records = list_chat_records(session_id="session-2", db_path=db_path)

            self.assertEqual([record.role for record in records], ["user", "assistant"])
            self.assertEqual(records[0].content, "什么是 RAG？")
            self.assertIn("RAG 是检索增强生成", records[1].content)

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


if __name__ == "__main__":
    unittest.main()
