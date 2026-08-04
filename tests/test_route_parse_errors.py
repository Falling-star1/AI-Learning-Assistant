import tempfile
import unittest
from pathlib import Path


class RouteParseErrorTests(unittest.TestCase):
    def test_route_user_request_returns_readable_error_for_empty_upload(self):
        from modules.agent.router import route_user_request

        class UploadedFile:
            name = "empty.txt"

            def getbuffer(self):
                return b"   \n\n"

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = route_user_request(
                question="What is RAG?",
                mode="课程资料问答",
                uploaded_file=UploadedFile(),
                upload_dir=Path(tmp_dir) / "uploads",
                store_path=Path(tmp_dir) / "rag_store.json",
                db_path=Path(tmp_dir) / "app.db",
            )

            self.assertEqual(result["knowledge_status"], "error")
            self.assertIn("资料解析失败", result["answer"])
            self.assertIn("没有提取到可用文字", result["answer"])


if __name__ == "__main__":
    unittest.main()
