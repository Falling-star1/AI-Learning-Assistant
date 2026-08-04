import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class LLMProviderSwitchingTests(unittest.TestCase):
    def test_auto_provider_prefers_generic_cloud_when_api_key_exists(self):
        from modules.llm.provider import CloudLLMProvider, create_llm_provider

        provider = create_llm_provider("auto", api_key="test-key")

        self.assertIsInstance(provider, CloudLLMProvider)

    def test_auto_provider_falls_back_without_api_key(self):
        from modules.llm.provider import FallbackLLMProvider, create_llm_provider

        provider = create_llm_provider("auto", api_key="")

        self.assertIsInstance(provider, FallbackLLMProvider)

    def test_ollama_provider_can_be_selected(self):
        from modules.llm.provider import OllamaLLMProvider, create_llm_provider

        provider = create_llm_provider("ollama")

        self.assertIsInstance(provider, OllamaLLMProvider)
        self.assertEqual(provider.provider_name, "ollama")

    def test_router_uses_selected_llm_provider_name(self):
        from modules.agent import router
        from modules.llm.provider import FallbackLLMProvider

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(router, "create_llm_provider", return_value=FallbackLLMProvider()) as factory:
                router.route_user_request(
                    question="你好",
                    mode="普通问答",
                    llm_provider_name="fallback",
                    db_path=Path(tmp_dir) / "app.db",
                    session_id="session-provider",
                )

        factory.assert_called_once_with("fallback")


if __name__ == "__main__":
    unittest.main()