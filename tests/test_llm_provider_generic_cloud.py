import os
import unittest
from unittest.mock import patch


class GenericCloudProviderTests(unittest.TestCase):
    def test_cloud_provider_uses_openai_compatible_client(self):
        from modules.llm.provider import CloudLLMProvider

        calls = []

        class FakeMessage:
            def __init__(self, content):
                self.content = content

        class FakeChoice:
            def __init__(self, content):
                self.message = FakeMessage(content)

        class FakeResponse:
            choices = [FakeChoice("云端模型回答")]

        class FakeCompletions:
            def create(self, **kwargs):
                calls.append(kwargs)
                return FakeResponse()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        provider = CloudLLMProvider(
            api_key="test-key",
            model_name="compatible-chat",
            base_url="https://example.com/v1",
            client=FakeClient(),
        )
        result = provider.generate("解释 RAG", context_chunks=["RAG 是检索增强生成。"])

        self.assertEqual(result.provider, "cloud")
        self.assertTrue(result.used_remote_model)
        self.assertEqual(result.text, "云端模型回答")
        self.assertEqual(calls[0]["model"], "compatible-chat")
        self.assertIn("RAG 是检索增强生成。", calls[0]["messages"][1]["content"])

    def test_factory_cloud_reads_generic_environment(self):
        from modules.llm.provider import CloudLLMProvider, create_llm_provider

        with patch.dict(
            os.environ,
            {
                "LLM_API_KEY": "generic-key",
                "LLM_MODEL_NAME": "generic-model",
                "LLM_BASE_URL": "https://example.com/v1",
            },
            clear=True,
        ):
            provider = create_llm_provider("cloud")

        self.assertIsInstance(provider, CloudLLMProvider)
        self.assertEqual(provider.model_name, "generic-model")
        self.assertEqual(provider.base_url, "https://example.com/v1")

    def test_auto_uses_generic_cloud_before_fallback(self):
        from modules.llm.provider import CloudLLMProvider, create_llm_provider

        with patch.dict(os.environ, {"LLM_API_KEY": "generic-key"}, clear=True):
            provider = create_llm_provider("auto")

        self.assertIsInstance(provider, CloudLLMProvider)


if __name__ == "__main__":
    unittest.main()