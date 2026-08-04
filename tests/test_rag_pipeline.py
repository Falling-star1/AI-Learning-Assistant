import tempfile
import unittest
from pathlib import Path


class RAGPipelineTests(unittest.TestCase):
    def test_load_document_reads_text_and_markdown_files(self):
        from modules.rag.loader import load_document

        with tempfile.TemporaryDirectory() as tmp_dir:
            txt_path = Path(tmp_dir) / "course.txt"
            md_path = Path(tmp_dir) / "notes.md"
            txt_path.write_text("RAG 是检索增强生成。", encoding="utf-8")
            md_path.write_text("# YOLO\nYOLO 用于目标检测。", encoding="utf-8")

            self.assertIn("检索增强生成", load_document(txt_path))
            self.assertIn("目标检测", load_document(md_path))

    def test_split_text_keeps_overlap_between_chunks(self):
        from modules.rag.splitter import split_text

        text = "abcdefghij" * 4
        chunks = split_text(text, chunk_size=15, overlap=5)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0][-5:], chunks[1][:5])

    def test_vector_store_builds_and_searches_relevant_chunks(self):
        from modules.rag.vector_store import build_vector_store, search_vector_store

        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "rag_store.json"
            build_vector_store(
                chunks=["YOLO 是目标检测算法。", "RAG 会检索课程资料再生成回答。"],
                source_name="lesson.txt",
                store_path=store_path,
            )

            results = search_vector_store("什么是 RAG？", store_path=store_path, top_k=1)

            self.assertEqual(len(results), 1)
            self.assertIn("检索课程资料", results[0].text)
            self.assertEqual(results[0].source_name, "lesson.txt")

    def test_retrieve_relevant_chunks_returns_structured_search_results(self):
        from modules.rag.qa import retrieve_relevant_chunks
        from modules.rag.vector_store import build_vector_store

        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "rag_store.json"
            build_vector_store(
                chunks=[
                    "YOLO 用于图片目标检测。",
                    "RAG 的流程包括解析、切分、检索和生成。",
                ],
                source_name="course.md",
                store_path=store_path,
            )

            results = retrieve_relevant_chunks(
                "RAG 的流程是什么？",
                store_path=store_path,
                top_k=1,
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].source_name, "course.md")
            self.assertEqual(results[0].chunk_id, 2)
            self.assertIn("解析、切分、检索和生成", results[0].text)

    def test_answer_with_rag_uses_retrieved_context_and_provider(self):
        from modules.llm.provider import LLMResult
        from modules.rag.qa import answer_with_rag
        from modules.rag.vector_store import build_vector_store

        class StubProvider:
            def generate(self, prompt, context_chunks=None):
                context = " / ".join(context_chunks or [])
                return LLMResult(text=f"{prompt} -> {context}", provider="stub")

        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "rag_store.json"
            build_vector_store(
                chunks=["RAG 的核心流程是解析、切分、检索和生成。"],
                source_name="rag.md",
                store_path=store_path,
            )

            answer = answer_with_rag(
                "RAG 的流程是什么？",
                store_path=store_path,
                provider=StubProvider(),
            )

            self.assertIn("解析、切分、检索和生成", answer)

    def test_route_user_request_builds_knowledge_base_from_uploaded_file(self):
        from modules.agent.router import route_user_request

        class UploadedFile:
            name = "rag.txt"

            def getbuffer(self):
                return b"RAG is retrieval augmented generation."

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = route_user_request(
                question="What is RAG?",
                mode="课程资料问答",
                uploaded_file=UploadedFile(),
                upload_dir=Path(tmp_dir) / "uploads",
                store_path=Path(tmp_dir) / "rag_store.json",
                db_path=Path(tmp_dir) / "app.db",
            )

            self.assertIn("fallback", result["provider"])
            self.assertIn("retrieval augmented generation", result["answer"])
            self.assertEqual(result["source_name"], "rag.txt")
            self.assertGreater(int(result["chunk_count"]), 0)
            self.assertEqual(result["knowledge_status"], "ready")


if __name__ == "__main__":
    unittest.main()
