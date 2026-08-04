import unittest


class LLMProviderTests(unittest.TestCase):
    def test_fallback_provider_returns_reference_chunks_without_api_key(self):
        from modules.llm.provider import FallbackLLMProvider

        provider = FallbackLLMProvider()
        result = provider.generate(
            prompt="什么是 RAG？",
            context_chunks=["RAG 是检索增强生成。", "它会先检索资料再生成回答。"],
        )

        self.assertEqual(result.provider, "fallback")
        self.assertFalse(result.used_remote_model)
        self.assertIn("未检测到可用的大模型 API Key", result.text)
        self.assertIn("RAG 是检索增强生成。", result.text)

    def test_fallback_provider_wraps_markdown_reference_as_plain_text(self):
        from modules.llm.provider import FallbackLLMProvider

        provider = FallbackLLMProvider()
        result = provider.generate(
            prompt="安装地址是什么？",
            context_chunks=["# 本地环境设置地址\nhttp://localhost/install"],
        )

        self.assertIn("```text", result.text)
        self.assertIn("# 本地环境设置地址", result.text)
        self.assertNotIn("[引用片段 1]\n#", result.text)

    def test_factory_uses_fallback_when_legacy_qwen_api_key_missing(self):
        from modules.llm.provider import FallbackLLMProvider, create_llm_provider

        provider = create_llm_provider("qwen", api_key="")

        self.assertIsInstance(provider, FallbackLLMProvider)

    def test_qwen_provider_accepts_injected_client_for_legacy_config(self):
        from modules.llm.provider import QwenLLMProvider

        calls = []

        def fake_client(**kwargs):
            calls.append(kwargs)
            return {"output": {"choices": [{"message": {"content": "这是云端模型生成的回答。"}}]}}

        provider = QwenLLMProvider(api_key="test-key", client=fake_client)
        result = provider.generate("解释 YOLO", context_chunks=["YOLO 是目标检测模型。"])

        self.assertEqual(result.provider, "qwen")
        self.assertTrue(result.used_remote_model)
        self.assertEqual(result.text, "这是云端模型生成的回答。")
        self.assertEqual(calls[0]["api_key"], "test-key")
        self.assertIn("YOLO 是目标检测模型。", calls[0]["messages"][1]["content"])

    def test_chat_with_llm_returns_text_from_selected_provider(self):
        from modules.llm.chat import chat_with_llm
        from modules.llm.provider import LLMResult

        class StubProvider:
            def generate(self, prompt, context_chunks=None):
                return LLMResult(text=f"answer: {prompt}", provider="stub")

        self.assertEqual(chat_with_llm("你好", provider=StubProvider()), "answer: 你好")

    def test_deepseek_provider_remains_as_compatibility_alias(self):
        from modules.llm.provider import DeepSeekLLMProvider

        calls = []

        class FakeMessage:
            def __init__(self, content):
                self.content = content

        class FakeChoice:
            def __init__(self, content):
                self.message = FakeMessage(content)

        class FakeResponse:
            def __init__(self, content):
                self.choices = [FakeChoice(content)]

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse("这是云端兼容 API 生成的回答。")

        class FakeChat:
            def __init__(self):
                self.completions = FakeCompletions()

        class FakeOpenAIClient:
            def __init__(self):
                self.chat = FakeChat()

        provider = DeepSeekLLMProvider(api_key="test-key", client=FakeOpenAIClient())
        result = provider.generate("解释 YOLO", context_chunks=["YOLO 是目标检测模型。"])

        self.assertEqual(result.provider, "deepseek")
        self.assertTrue(result.used_remote_model)
        self.assertEqual(result.text, "这是云端兼容 API 生成的回答。")
        self.assertEqual(calls[0]["model"], "deepseek-chat")
        self.assertEqual(calls[0]["messages"][0]["role"], "system")
        self.assertEqual(calls[0]["messages"][1]["role"], "user")
        self.assertIn("YOLO 是目标检测模型。", calls[0]["messages"][1]["content"])

    def test_factory_uses_fallback_when_legacy_deepseek_api_key_missing(self):
        from modules.llm.provider import FallbackLLMProvider, create_llm_provider

        provider = create_llm_provider("deepseek")

        self.assertIsInstance(provider, FallbackLLMProvider)


if __name__ == "__main__":
    unittest.main()