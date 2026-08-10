import tempfile
import unittest
import sqlite3
from contextlib import closing
from pathlib import Path


class KnowledgeBaseManagementTests(unittest.TestCase):
    def test_ensure_default_knowledge_base_creates_active_kb(self):
        from modules.database.knowledge import ensure_default_knowledge_base, list_knowledge_bases

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "app.db"
            kb = ensure_default_knowledge_base(db_path=db_path)
            bases = list_knowledge_bases(db_path=db_path)

            self.assertEqual(kb.name, "默认知识库")
            self.assertTrue(kb.is_active)
            self.assertEqual(len(bases), 1)
            self.assertEqual(bases[0].kb_id, kb.kb_id)

    def test_create_and_switch_active_knowledge_base(self):
        from modules.database.knowledge import (
            create_knowledge_base,
            ensure_default_knowledge_base,
            get_active_knowledge_base,
            set_active_knowledge_base,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "app.db"
            ensure_default_knowledge_base(db_path=db_path)
            created = create_knowledge_base("Dify 课程", db_path=db_path)

            set_active_knowledge_base(created.kb_id, db_path=db_path)
            active = get_active_knowledge_base(db_path=db_path)

            self.assertEqual(active.kb_id, created.kb_id)
            self.assertEqual(active.name, "Dify 课程")
            self.assertTrue(active.is_active)

    def test_save_and_list_files_by_knowledge_base(self):
        from modules.database.knowledge import (
            ensure_default_knowledge_base,
            list_knowledge_base_files,
            save_knowledge_base_file,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "app.db"
            kb = ensure_default_knowledge_base(db_path=db_path)
            file_id = save_knowledge_base_file(
                kb_id=kb.kb_id,
                file_name="rag.pdf",
                file_type="pdf",
                storage_path=Path(tmp_dir) / "rag.pdf",
                chunk_count=3,
                status="indexed",
                db_path=db_path,
            )

            files = list_knowledge_base_files(kb.kb_id, db_path=db_path)

            self.assertGreater(file_id, 0)
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].file_name, "rag.pdf")
            self.assertEqual(files[0].chunk_count, 3)
            self.assertEqual(files[0].kb_id, kb.kb_id)

    def test_delete_file_from_knowledge_base(self):
        from modules.database.knowledge import (
            delete_knowledge_base_file,
            ensure_default_knowledge_base,
            list_knowledge_base_files,
            save_knowledge_base_file,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "app.db"
            kb = ensure_default_knowledge_base(db_path=db_path)
            file_id = save_knowledge_base_file(
                kb_id=kb.kb_id,
                file_name="rag.pdf",
                file_type="pdf",
                storage_path=Path(tmp_dir) / "rag.pdf",
                chunk_count=3,
                status="indexed",
                db_path=db_path,
            )

            deleted = delete_knowledge_base_file(file_id, db_path=db_path)
            files = list_knowledge_base_files(kb.kb_id, db_path=db_path)

            self.assertTrue(deleted)
            self.assertEqual(files, [])

    def test_delete_knowledge_base_removes_files_and_activates_fallback(self):
        from modules.database.knowledge import (
            create_knowledge_base,
            delete_knowledge_base,
            ensure_default_knowledge_base,
            get_active_knowledge_base,
            list_knowledge_base_files,
            list_knowledge_bases,
            save_knowledge_base_file,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "app.db"
            original = ensure_default_knowledge_base(db_path=db_path)
            target = create_knowledge_base("要删除的知识库", db_path=db_path, is_active=True)
            save_knowledge_base_file(
                kb_id=target.kb_id,
                file_name="rag.pdf",
                file_type="pdf",
                storage_path=Path(tmp_dir) / "rag.pdf",
                chunk_count=3,
                status="indexed",
                db_path=db_path,
            )

            deleted = delete_knowledge_base(target.kb_id, db_path=db_path)
            bases = list_knowledge_bases(db_path=db_path)
            active = get_active_knowledge_base(db_path=db_path)

            self.assertTrue(deleted)
            self.assertEqual([kb.kb_id for kb in bases], [original.kb_id])
            self.assertEqual(active.kb_id, original.kb_id)
            self.assertEqual(list_knowledge_base_files(target.kb_id, db_path=db_path), [])

    def test_delete_knowledge_base_removes_uploaded_files_and_vector_store(self):
        from modules.database.knowledge import (
            create_knowledge_base,
            delete_knowledge_base,
            ensure_default_knowledge_base,
            save_knowledge_base_file,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            db_path = tmp_path / "app.db"
            ensure_default_knowledge_base(db_path=db_path)
            target = create_knowledge_base("kb with files", db_path=db_path, is_active=True)
            upload_dir = tmp_path / "uploads" / target.kb_id
            upload_dir.mkdir(parents=True)
            uploaded_file = upload_dir / "rag.md"
            uploaded_file.write_text("course material", encoding="utf-8")
            store_path = tmp_path / "vector_db" / target.kb_id / "rag_store.json"
            with closing(sqlite3.connect(db_path)) as conn:
                conn.execute(
                    "UPDATE knowledge_bases SET store_path = ? WHERE kb_id = ?",
                    (str(store_path), target.kb_id),
                )
                conn.commit()
            store_path.parent.mkdir(parents=True, exist_ok=True)
            store_path.write_text("[]", encoding="utf-8")
            save_knowledge_base_file(
                kb_id=target.kb_id,
                file_name=uploaded_file.name,
                file_type="md",
                storage_path=uploaded_file,
                chunk_count=1,
                status="indexed",
                db_path=db_path,
            )

            deleted = delete_knowledge_base(target.kb_id, db_path=db_path)

            self.assertTrue(deleted)
            self.assertFalse(uploaded_file.exists())
            self.assertFalse(upload_dir.exists())
            self.assertFalse(store_path.exists())
            self.assertFalse(store_path.parent.exists())


if __name__ == "__main__":
    unittest.main()
