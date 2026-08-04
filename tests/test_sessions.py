import tempfile
import unittest
from pathlib import Path


class SessionManagementTests(unittest.TestCase):
    def test_ensure_default_session_creates_recoverable_session(self):
        from modules.database.sessions import ensure_default_session, list_sessions

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "app.db"
            session = ensure_default_session(db_path=db_path)
            sessions = list_sessions(db_path=db_path)

            self.assertTrue(session.session_id.startswith("session_"))
            self.assertEqual(session.title, "新会话")
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0].session_id, session.session_id)

    def test_create_session_and_update_title(self):
        from modules.database.sessions import create_session, get_session, update_session_title

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "app.db"
            session = create_session(title="RAG 问答", active_kb_id="kb_1", db_path=db_path)
            update_session_title(session.session_id, "Dify 复习", db_path=db_path)

            loaded = get_session(session.session_id, db_path=db_path)

            self.assertEqual(loaded.title, "Dify 复习")
            self.assertEqual(loaded.active_kb_id, "kb_1")

    def test_session_title_can_be_inferred_from_first_question(self):
        from modules.database.sessions import create_session, maybe_update_session_title

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "app.db"
            session = create_session(db_path=db_path)

            maybe_update_session_title(session.session_id, "请解释 RAG 的流程是什么", db_path=db_path)
            maybe_update_session_title(session.session_id, "第二个问题不应该覆盖标题", db_path=db_path)

            sessions = list(__import__("modules.database.sessions", fromlist=["list_sessions"]).list_sessions(db_path=db_path))

            self.assertEqual(sessions[0].title, "请解释 RAG 的流程是什么")

    def test_delete_session_removes_chat_history(self):
        from modules.database.history import list_chat_records, save_chat_record
        from modules.database.sessions import create_session, delete_session, get_session

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "app.db"
            session = create_session(db_path=db_path)
            save_chat_record(session.session_id, "user", "解释 RAG", "课程资料问答", "fallback", db_path=db_path)
            save_chat_record(session.session_id, "assistant", "RAG 是检索增强生成", "课程资料问答", "fallback", db_path=db_path)

            deleted = delete_session(session.session_id, db_path=db_path)

            self.assertTrue(deleted)
            self.assertIsNone(get_session(session.session_id, db_path=db_path))
            self.assertEqual(list_chat_records(session_id=session.session_id, db_path=db_path), [])


if __name__ == "__main__":
    unittest.main()
