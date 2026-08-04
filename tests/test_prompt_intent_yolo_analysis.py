import unittest

from modules.llm.provider import LLMResult


class StubProvider:
    provider_name = "stub"

    def __init__(self):
        self.calls = []

    def generate(self, prompt, context_chunks=None):
        self.calls.append({"prompt": prompt, "context_chunks": list(context_chunks or [])})
        return LLMResult(text="自然语言分析结果", provider=self.provider_name, used_remote_model=True)


class PromptIntentYoloAnalysisTests(unittest.TestCase):
    def test_learning_prompt_templates_are_specialized_by_type(self):
        from modules.llm.prompt import build_learning_prompt

        summary = build_learning_prompt("课程总结", "RAG 工作流", has_context=True)
        outline = build_learning_prompt("复习提纲", "YOLO 检测", has_context=False)
        report_outline = build_learning_prompt("报告大纲", "多模态助手", has_context=True)

        self.assertIn("课程总结", summary)
        self.assertIn("核心概念", summary)
        self.assertIn("复习提纲", outline)
        self.assertIn("考点", outline)
        self.assertIn("报告大纲", report_outline)
        self.assertIn("章节结构", report_outline)
        self.assertNotEqual(summary, outline)

    def test_detect_intent_prefers_uploaded_image_then_learning_keywords(self):
        from modules.agent.router import detect_intent

        self.assertEqual(detect_intent("帮我总结 RAG", uploaded_image=object()), "图片目标检测")
        self.assertEqual(detect_intent("帮我生成复习提纲"), "学习辅助生成")
        self.assertEqual(detect_intent("根据资料解释 RAG", has_knowledge_base=True), "课程资料问答")
        self.assertEqual(detect_intent("你好"), "普通问答")

    def test_yolo_analysis_uses_llm_provider_when_available(self):
        from modules.yolo.analyzer import analyze_with_llm

        provider = StubProvider()
        answer = analyze_with_llm(
            detections=[
                {"class_name": "person", "confidence": 0.93},
                {"class_name": "book", "confidence": 0.81},
            ],
            question="描述这张图",
            provider=provider,
        )

        self.assertEqual(answer, "自然语言分析结果")
        self.assertEqual(len(provider.calls), 1)
        self.assertIn("person", provider.calls[0]["prompt"])
        self.assertIn("book", provider.calls[0]["prompt"])


if __name__ == "__main__":
    unittest.main()