from collections.abc import Sequence

from modules.llm.provider import LLMProvider, create_llm_provider


def chat_with_llm(
    prompt: str,
    context_chunks: Sequence[str] | None = None,
    provider: LLMProvider | None = None,
) -> str:
    """Generate an answer with the configured LLM provider."""
    selected_provider = provider or create_llm_provider()
    return selected_provider.generate(
        prompt=prompt,
        context_chunks=context_chunks,
    ).text
