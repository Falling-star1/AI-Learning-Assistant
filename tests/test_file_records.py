import tempfile
import unittest
from pathlib import Path


class FileRecordTests(unittest.TestCase):
    def test_save_and_list_file_records(self):
        from modules.database.files import list_file_records, save_file_record

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "files.db"
            storage_path = Path(tmp_dir) / "uploads" / "lesson.pdf"

            save_file_record(
                file_name="lesson.pdf",
                file_type="pdf",
                storage_path=storage_path,
                status="indexed",
                db_path=db_path,
            )

            records = list_file_records(db_path=db_path)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].file_name, "lesson.pdf")
            self.assertEqual(records[0].file_type, "pdf")
            self.assertEqual(records[0].status, "indexed")
            self.assertEqual(records[0].storage_path, str(storage_path))

    def test_route_user_request_saves_uploaded_file_record(self):
        from modules.agent.router import route_user_request
        from modules.database.files import list_file_records

        class UploadedFile:
            name = "rag-note.txt"

            def getbuffer(self):
                return "RAG 是检索增强生成。".encode("utf-8")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            db_path = tmp_path / "app.db"

            route_user_request(
                question="什么是 RAG？",
                mode="课程资料问答",
                uploaded_file=UploadedFile(),
                upload_dir=tmp_path / "uploads",
                store_path=tmp_path / "rag_store.json",
                db_path=db_path,
                session_id="session-file",
            )

            records = list_file_records(db_path=db_path)

            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].file_name, "rag-note.txt")
            self.assertEqual(records[0].file_type, "txt")
            self.assertEqual(records[0].status, "indexed")


if __name__ == "__main__":
    unittest.main()
