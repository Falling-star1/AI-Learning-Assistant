import tempfile
import unittest
from pathlib import Path

from modules.llm.provider import LLMResult


class StubProvider:
    provider_name = "stub"

    def __init__(self):
        self.context_chunks = []

    def generate(self, prompt, context_chunks=None):
        self.context_chunks = list(context_chunks or [])
        return LLMResult(
            text=f"生成内容\n\n{prompt}",
            provider=self.provider_name,
            used_remote_model=True,
        )


class ReportGeneratorTests(unittest.TestCase):
    def test_study_report_includes_practice_questions_without_auto_saving_file(self):
        from modules.report.generator import generate_report

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = generate_report(
                topic="RAG 的工作流程",
                report_type="学习报告",
                provider=StubProvider(),
            )

            self.assertIn("练习题", result.content)
            self.assertIn("选择题", result.content)
            self.assertIn("简答题", result.content)
            self.assertIn("实践题", result.content)
            self.assertTrue(result.download_name.endswith(".md"))
            self.assertFalse(hasattr(result, "report_path"))
            self.assertEqual(list(Path(tmp_dir).glob("*.md")), [])

    def test_remote_report_provider_receives_retrieved_context(self):
        from modules.report.generator import generate_report

        provider = StubProvider()
        result = generate_report(
            topic="RAG 的工作流程",
            report_type="学习报告",
            context_chunks=["来源：course.md，片段 2\nRAG 包括解析、切分、检索和生成。"],
            provider=provider,
        )

        self.assertEqual(len(provider.context_chunks), 1)
        self.assertIn("解析、切分、检索和生成", provider.context_chunks[0])
        self.assertEqual(result.knowledge_status, "ready")
        self.assertIn("仅依据提供的课程资料", result.content)

    def test_fallback_report_includes_retrieved_course_evidence(self):
        from modules.llm.provider import FallbackLLMProvider
        from modules.report.generator import generate_report

        result = generate_report(
            topic="RAG 的工作流程",
            report_type="学习报告",
            context_chunks=["来源：course.md，片段 2\nRAG 包括解析、切分、检索和生成。"],
            provider=FallbackLLMProvider(),
        )

        self.assertEqual(result.knowledge_status, "ready")
        self.assertIn("课程资料要点", result.content)
        self.assertIn("解析、切分、检索和生成", result.content)

    def test_experiment_report_prompt_prioritizes_user_format_requirements(self):
        from modules.report.templates import build_experiment_report_prompt

        prompt = build_experiment_report_prompt(
            topic="YOLO 图片检测实验",
            format_requirements="请按：目的、环境、步骤、结果、总结 五部分生成",
        )

        self.assertIn("优先遵循用户提供的格式要求", prompt)
        self.assertIn("目的、环境、步骤、结果、总结", prompt)
        self.assertNotIn("实验名称\n- 实验目的\n- 实验环境", prompt)

    def test_route_user_request_generates_downloadable_report_content(self):
        from modules.agent.router import route_user_request

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = route_user_request(
                question="生成 RAG 学习报告",
                mode="学习辅助生成",
                report_type="学习报告",
                format_requirements="",
                store_path=Path(tmp_dir) / "missing-store.json",
                db_path=Path(tmp_dir) / "app.db",
                session_id="session-report",
            )

            self.assertEqual(result["provider"], "fallback")
            self.assertEqual(result["report_type"], "学习报告")
            self.assertNotIn("report_path", result)
            self.assertTrue(result["download_name"].endswith(".md"))
            self.assertIn("练习题", result["answer"])

    def test_report_route_retrieves_context_and_returns_sources(self):
        from modules.agent.router import route_user_request
        from modules.rag.vector_store import build_vector_store

        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "rag_store.json"
            build_vector_store(
                chunks=["RAG 的核心流程是解析、切分、检索和生成。"],
                source_name="rag-course.md",
                store_path=store_path,
            )

            result = route_user_request(
                question="RAG 的核心流程",
                mode="学习辅助生成",
                report_type="学习报告",
                store_path=store_path,
                db_path=Path(tmp_dir) / "app.db",
                session_id="report-with-rag",
            )

            self.assertEqual(result["knowledge_status"], "ready")
            self.assertEqual(len(result["sources"]), 1)
            self.assertEqual(result["sources"][0]["source_name"], "rag-course.md")
            self.assertIn("解析、切分、检索和生成", result["answer"])
            self.assertIn("## 参考资料", result["answer"])

    def test_report_route_uses_material_preview_when_prompt_is_too_generic(self):
        from modules.agent.router import route_user_request
        from modules.rag.vector_store import build_vector_store

        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "rag_store.json"
            build_vector_store(
                chunks=["CPU 的基本组成包括运算器、控制器和寄存器，指令周期包含取指、分析和执行。"],
                source_name="cpu-course.pptx",
                store_path=store_path,
            )

            result = route_user_request(
                question="请生成一份实践题",
                mode="学习辅助生成",
                report_type="学习报告",
                store_path=store_path,
                has_knowledge_base=True,
                db_path=Path(tmp_dir) / "app.db",
                session_id="report-generic-with-kb",
            )

            self.assertEqual(result["knowledge_status"], "ready")
            self.assertEqual(result["sources"][0]["source_name"], "cpu-course.pptx")
            self.assertIn("CPU 的基本组成", result["answer"])

    def test_report_route_without_context_returns_empty_sources(self):
        from modules.agent.router import route_user_request

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = route_user_request(
                question="RAG 的核心流程",
                mode="学习辅助生成",
                report_type="学习报告",
                store_path=Path(tmp_dir) / "missing-store.json",
                db_path=Path(tmp_dir) / "app.db",
                session_id="report-without-rag",
            )

            self.assertEqual(result["knowledge_status"], "empty")
            self.assertEqual(result["sources"], [])
            self.assertNotIn("## 参考资料", result["answer"])

    def test_report_knowledge_notice_summarizes_used_sources(self):
        from app import get_report_knowledge_notice

        kind, message = get_report_knowledge_notice(
            "ready",
            [{"source_name": "rag.md", "chunk_id": 2}],
        )

        self.assertEqual(kind, "success")
        self.assertEqual(message, "本次报告参考了 1 个知识库片段。")

    def test_report_knowledge_notice_warns_when_sources_are_empty(self):
        from app import get_report_knowledge_notice

        kind, message = get_report_knowledge_notice("empty", [])

        self.assertEqual(kind, "warning")
        self.assertIn("通用草稿", message)

    def test_only_indexing_results_refresh_sidebar_knowledge_status(self):
        from app import is_knowledge_base_build_result

        report_result = {"knowledge_status": "ready", "sources": [{"chunk_id": 1}]}
        indexing_result = {
            "knowledge_status": "ready",
            "source_name": "course.md",
            "chunk_count": "2",
        }

        self.assertFalse(is_knowledge_base_build_result(report_result))
        self.assertTrue(is_knowledge_base_build_result(indexing_result))

    def test_download_file_name_uses_topic_and_report_type(self):
        from modules.report.generator import get_report_download_name

        name = get_report_download_name("RAG 的工作流程", "学习报告")

        self.assertTrue(name.endswith("_RAG_的工作流程_学习报告.md"))


if __name__ == "__main__":
    unittest.main()
