from collections.abc import Sequence

from modules.llm.provider import LLMProvider, create_llm_provider


def chat_with_llm(
    prompt: str,
    context_chunks: Sequence[str] | None = None,
    conversation_history: Sequence[dict[str, str]] | None = None,
    provider: LLMProvider | None = None,
) -> str:
    """Generate an answer with the configured LLM provider."""
    selected_provider = provider or create_llm_provider()
    if conversation_history:
        return selected_provider.generate(
            prompt=prompt,
            context_chunks=context_chunks,
            conversation_history=conversation_history,
        ).text
    return selected_provider.generate(prompt=prompt, context_chunks=context_chunks).text
