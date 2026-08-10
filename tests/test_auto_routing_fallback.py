import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.llm.provider import LLMResult


class StubProvider:
    provider_name = "stub"

    def __init__(self, intent: str | None = None) -> None:
        self.intent = intent
        self.calls: list[dict[str, object]] = []

    def generate(self, prompt: str, context_chunks=None, conversation_history=None) -> LLMResult:
        self.calls.append(
            {
                "prompt": prompt,
                "context_chunks": context_chunks,
                "conversation_history": conversation_history,
            }
        )
        if "只输出一个类别" in prompt and self.intent:
            return LLMResult(text=self.intent, provider=self.provider_name, used_remote_model=True)
        return LLMResult(text=f"LLM 回复：{prompt}", provider=self.provider_name, used_remote_model=True)


class FallbackLikeClassifierProvider(StubProvider):
    provider_name = "deepseek"

    def generate(self, prompt: str, context_chunks=None, conversation_history=None) -> LLMResult:
        self.calls.append(
            {
                "prompt": prompt,
                "context_chunks": context_chunks,
                "conversation_history": conversation_history,
            }
        )
        if "只输出一个类别" in prompt:
            return LLMResult(
                text=f"云端失败，fallback 回显分类提示：{prompt}",
                provider="fallback",
                used_remote_model=False,
            )
        return LLMResult(text=f"LLM 回复：{prompt}", provider=self.provider_name, used_remote_model=True)


class AutoRoutingFallbackTest(unittest.TestCase):
    def test_auto_mode_chat_does_not_route_to_rag_just_because_kb_exists(self):
        from modules.agent import router
        from modules.agent.router import MODE_AUTO, route_user_request

        with tempfile.TemporaryDirectory() as tmp_dir:
            provider = StubProvider(intent="闲聊")
            with patch.object(router, "create_llm_provider", return_value=provider):
                result = route_user_request(
                    question="你好",
                    mode=MODE_AUTO,
                    store_path=Path(tmp_dir) / "missing_store.json",
                    db_path=Path(tmp_dir) / "chat.db",
                    session_id="auto-chat",
                    has_knowledge_base=True,
                    llm_provider_name="stub",
                )

        self.assertEqual(result["resolved_mode"], "普通问答")
        self.assertIn("LLM 回复：你好", result["answer"])
        self.assertNotIn("当前知识库中没有检索到相关片段", result["answer"])
        self.assertNotIn("system_notice", result)

    def test_course_qa_falls_back_to_llm_when_retrieval_is_empty(self):
        from modules.agent import router
        from modules.agent.router import MODE_COURSE_QA, route_user_request

        with tempfile.TemporaryDirectory() as tmp_dir:
            provider = StubProvider()
            with patch.object(router, "create_llm_provider", return_value=provider):
                result = route_user_request(
                    question="解释一个资料没有覆盖的概念",
                    mode=MODE_COURSE_QA,
                    store_path=Path(tmp_dir) / "missing_store.json",
                    db_path=Path(tmp_dir) / "chat.db",
                    session_id="rag-empty",
                    has_knowledge_base=True,
                    llm_provider_name="stub",
                )

        self.assertEqual(result["resolved_mode"], "课程资料问答")
        self.assertIn("LLM 回复：解释一个资料没有覆盖的概念", result["answer"])
        self.assertIn("通用回答", result["answer"])
        self.assertEqual(result["sources"], [])
        self.assertNotIn("system_notice", result)

    def test_auto_mode_simple_math_uses_plain_chat(self):
        from modules.agent import router
        from modules.agent.router import MODE_AUTO, route_user_request

        with tempfile.TemporaryDirectory() as tmp_dir:
            provider = StubProvider(intent="资料查询")
            with patch.object(router, "create_llm_provider", return_value=provider):
                result = route_user_request(
                    question="1+1等于几",
                    mode=MODE_AUTO,
                    store_path=Path(tmp_dir) / "missing_store.json",
                    db_path=Path(tmp_dir) / "chat.db",
                    session_id="auto-math",
                    has_knowledge_base=True,
                    llm_provider_name="stub",
                )

        self.assertEqual(result["resolved_mode"], "普通问答")
        self.assertIn("LLM 回复：1+1等于几", result["answer"])
        self.assertEqual(len(provider.calls), 1)

    def test_plain_chat_passes_previous_turns_to_llm(self):
        from modules.agent import router
        from modules.agent.router import MODE_CHAT, route_user_request

        with tempfile.TemporaryDirectory() as tmp_dir:
            provider = StubProvider()
            with patch.object(router, "create_llm_provider", return_value=provider):
                result = route_user_request(
                    question="那它的作用是什么？",
                    mode=MODE_CHAT,
                    store_path=Path(tmp_dir) / "missing_store.json",
                    db_path=Path(tmp_dir) / "chat.db",
                    session_id="follow-up",
                    has_knowledge_base=True,
                    llm_provider_name="stub",
                    conversation_history=[
                        {"role": "user", "content": "CPU 是什么？"},
                        {"role": "assistant", "content": "CPU 是中央处理器。"},
                    ],
                )

        self.assertEqual(result["resolved_mode"], MODE_CHAT)
        self.assertEqual(
            provider.calls[-1]["conversation_history"],
            [
                {"role": "user", "content": "CPU 是什么？"},
                {"role": "assistant", "content": "CPU 是中央处理器。"},
            ],
        )

    def test_auto_mode_course_query_uses_rag_when_retrieval_matches(self):
        from modules.agent import router
        from modules.agent.router import MODE_AUTO, route_user_request
        from modules.rag.vector_store import build_vector_store

        with tempfile.TemporaryDirectory() as tmp_dir:
            store_path = Path(tmp_dir) / "rag_store.json"
            build_vector_store(
                ["CPU 指令周期包括取指、译码、执行和写回等阶段。"],
                source_name="cpu.md",
                store_path=store_path,
            )
            provider = StubProvider(intent="资料查询")
            with patch.object(router, "create_llm_provider", return_value=provider):
                result = route_user_request(
                    question="解释 CPU 指令周期",
                    mode=MODE_AUTO,
                    store_path=store_path,
                    db_path=Path(tmp_dir) / "chat.db",
                    session_id="auto-rag",
                    has_knowledge_base=True,
                    llm_provider_name="stub",
                )

        self.assertEqual(result["resolved_mode"], "课程资料问答")
        self.assertEqual(result["sources"][0]["source_name"], "cpu.md")
        self.assertEqual(len(provider.calls), 2)
        self.assertIn("CPU 指令周期", str(provider.calls[-1]["context_chunks"][0]))

    def test_failed_llm_intent_classification_falls_back_to_heuristics(self):
        from modules.agent.router import detect_intent

        provider = FallbackLikeClassifierProvider()

        self.assertEqual(
            detect_intent("解释 CPU 指令周期", has_knowledge_base=True, provider=provider),
            "课程资料问答",
        )


if __name__ == "__main__":
    unittest.main()
