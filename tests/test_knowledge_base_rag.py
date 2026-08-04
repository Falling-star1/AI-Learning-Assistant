import tempfile
import unittest
from pathlib import Path


class KnowledgeBaseRAGTests(unittest.TestCase):
    def test_vector_store_append_keeps_existing_documents_searchable(self):
        from modules.rag.vector_store import build_vector_store, search_vector_store

        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "kb" / "rag_store.json"
            build_vector_store(
                chunks=["RAG 用于检索增强生成。"],
                source_name="rag.md",
                store_path=store_path,
            )
            build_vector_store(
                chunks=["YOLO 用于目标检测。"],
                source_name="yolo.md",
                store_path=store_path,
                append=True,
            )

            rag_results = search_vector_store("检索增强", store_path=store_path, top_k=2)
            yolo_results = search_vector_store("目标检测", store_path=store_path, top_k=2)

            self.assertEqual(rag_results[0].source_name, "rag.md")
            self.assertEqual(yolo_results[0].source_name, "yolo.md")
            self.assertEqual(yolo_results[0].chunk_id, 2)

    def test_build_knowledge_base_can_append_to_existing_store(self):
        from modules.rag.qa import build_knowledge_base, retrieve_relevant_chunks

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store_path = tmp_path / "kb" / "rag_store.json"
            rag_path = tmp_path / "rag.txt"
            yolo_path = tmp_path / "yolo.txt"
            rag_path.write_text("RAG 用于检索增强生成。", encoding="utf-8")
            yolo_path.write_text("YOLO 用于目标检测。", encoding="utf-8")

            build_knowledge_base(rag_path, store_path=store_path)
            build_knowledge_base(yolo_path, store_path=store_path, append=True)

            results = retrieve_relevant_chunks("目标检测", store_path=store_path, top_k=2)

            self.assertEqual(results[0].source_name, "yolo.txt")
            self.assertEqual(results[0].chunk_id, 2)


if __name__ == "__main__":
    unittest.main()
