import tempfile
import unittest
from pathlib import Path


class RouteKnowledgeBaseTests(unittest.TestCase):
    def test_course_qa_returns_structured_sources_from_selected_store(self):
        from modules.agent.router import route_user_request
        from modules.rag.vector_store import build_vector_store

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            selected_store = tmp_path / "kb_selected" / "rag_store.json"
            other_store = tmp_path / "kb_other" / "rag_store.json"
            build_vector_store(
                chunks=["当前知识库包含 RAG 检索增强生成。"],
                source_name="selected.md",
                store_path=selected_store,
            )
            build_vector_store(
                chunks=["另一个知识库包含 YOLO 目标检测。"],
                source_name="other.md",
                store_path=other_store,
            )

            result = route_user_request(
                question="RAG 是什么？",
                mode="课程资料问答",
                store_path=selected_store,
                db_path=tmp_path / "app.db",
                session_id="session-kb",
            )

            self.assertEqual(result["sources"][0]["source_name"], "selected.md")
            self.assertEqual(result["sources"][0]["chunk_id"], 1)
            self.assertIn("score", result["sources"][0])
            self.assertNotIn("other.md", result["answer"])

    def test_course_qa_returns_agent_workflow_steps(self):
        from modules.agent.router import route_user_request
        from modules.rag.vector_store import build_vector_store

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            store_path = tmp_path / "rag_store.json"
            build_vector_store(
                chunks=["Agent 会识别意图，然后调用 RAG 工具检索课程资料。"],
                source_name="agent.md",
                store_path=store_path,
            )

            result = route_user_request(
                question="Agent 如何调用 RAG？",
                mode="自动识别",
                store_path=store_path,
                db_path=tmp_path / "app.db",
                session_id="session-workflow",
                has_knowledge_base=True,
            )

            self.assertEqual(
                [step["title"] for step in result["workflow_steps"]],
                ["识别意图", "调用工具", "返回结果"],
            )
            self.assertEqual(result["workflow_steps"][0]["detail"], "自动识别 → 课程资料问答")
            self.assertIn("RAG 知识库问答", result["workflow_steps"][1]["detail"])
            self.assertIn("引用来源 1 条", result["workflow_steps"][2]["detail"])

if __name__ == "__main__":
    unittest.main()

