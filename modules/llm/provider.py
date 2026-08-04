import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from config import DEFAULT_LLM_PROVIDER, LLM_BASE_URL, LLM_MODEL_NAME
from modules.utils.logger import get_logger

# 网络类异常类型；openai 未安装时为空 tuple，不影响其他 provider
try:
    from openai import APIConnectionError, APITimeoutError

    _NETWORK_ERRORS: tuple[type[BaseException], ...] = (APIConnectionError, APITimeoutError)
except ImportError:
    _NETWORK_ERRORS = ()

LEGACY_QWEN_MODEL_NAME = "qwen-turbo"
LEGACY_DEEPSEEK_MODEL_NAME = "deepseek-chat"
LEGACY_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "qwen2.5:7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


@dataclass(frozen=True)
class LLMResult:
    text: str
    provider: str
    used_remote_model: bool = False


class LLMProvider(Protocol):
    def generate(
        self,
        prompt: str,
        context_chunks: Sequence[str] | None = None,
    ) -> LLMResult:
        """Generate an answer from the selected provider."""


class FallbackLLMProvider:
    """Return local RAG evidence when no remote model is configured."""

    provider_name = "fallback"

    def generate(
        self,
        prompt: str,
        context_chunks: Sequence[str] | None = None,
    ) -> LLMResult:
        chunks = [chunk.strip() for chunk in (context_chunks or []) if chunk.strip()]
        if not chunks:
            text = (
                "未检测到可用的大模型 API Key，当前处于本地 fallback 模式。\n\n"
                f"用户问题：{prompt}\n\n"
                "暂未检索到可引用的课程资料片段。"
            )
        else:
            references = "\n\n".join(
                f"**引用片段 {index}**\n\n```text\n{_escape_code_fence(chunk)}\n```"
                for index, chunk in enumerate(chunks, start=1)
            )
            text = (
                "未检测到可用的大模型 API Key，当前处于本地 fallback 模式。\n"
                "以下为 RAG 检索到的参考片段，可先用于人工判断和答辩演示：\n\n"
                f"{references}"
            )
        return LLMResult(text=text, provider=self.provider_name, used_remote_model=False)


class QwenLLMProvider:
    """Legacy Qwen adapter kept for compatibility with earlier tests and configs."""

    provider_name = "qwen"

    def __init__(
        self,
        api_key: str,
        model_name: str = LEGACY_QWEN_MODEL_NAME,
        client: Callable[..., Any] | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.client = client or self._default_client

    def generate(
        self,
        prompt: str,
        context_chunks: Sequence[str] | None = None,
    ) -> LLMResult:
        messages = _build_chat_messages(prompt, context_chunks)
        response = self.client(api_key=self.api_key, model=self.model_name, messages=messages)
        return LLMResult(text=self._extract_text(response), provider=self.provider_name, used_remote_model=True)

    def _default_client(self, **kwargs: Any) -> Any:
        try:
            import dashscope
        except ImportError as exc:
            raise RuntimeError("未安装 dashscope，无法调用云端模型 API。请先安装 requirements.txt 中的依赖。") from exc
        return dashscope.Generation.call(**kwargs)

    def _extract_text(self, response: Any) -> str:
        if isinstance(response, dict):
            try:
                return response["output"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"云端 API 返回格式无法解析：{response}") from exc
        output = getattr(response, "output", None)
        if output and isinstance(output, dict):
            try:
                return output["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise RuntimeError(f"云端 API 返回格式无法解析：{response}") from exc
        raise RuntimeError(f"云端 API 返回格式无法解析：{response}")


class CloudLLMProvider:
    """OpenAI-compatible cloud provider configured by generic LLM_* variables."""

    provider_name = "cloud"

    def __init__(
        self,
        api_key: str,
        model_name: str = LLM_MODEL_NAME,
        base_url: str = LLM_BASE_URL,
        client: Any | None = None,
    ) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url.rstrip("/") if base_url else ""
        self.client = client

    def generate(
        self,
        prompt: str,
        context_chunks: Sequence[str] | None = None,
    ) -> LLMResult:
        client = self.client or self._default_client()
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=_build_chat_messages(prompt, context_chunks, system_prompt=self._system_prompt()),
                stream=False,
            )
        except _NETWORK_ERRORS as exc:
            get_logger().warning(
                "Cloud LLM 网络异常，降级到 fallback：provider=%s model=%s error=%s",
                self.provider_name,
                self.model_name,
                exc,
            )
            fallback = FallbackLLMProvider()
            result = fallback.generate(prompt=prompt, context_chunks=context_chunks)
            return LLMResult(
                text=f"⚠️ 云端模型调用失败（{type(exc).__name__}），已降级到本地 fallback 模式。\n\n{result.text}",
                provider="fallback",
                used_remote_model=False,
            )
        return LLMResult(text=self._extract_text(response), provider=self.provider_name, used_remote_model=True)

    def _default_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("未安装 openai 库，无法调用云端兼容 API。请先安装 requirements.txt 中的依赖。") from exc
        kwargs: dict[str, str] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def _system_prompt(self) -> str:
        return "你是 AI 多模态学习助手，基于云端大模型，请优先依据用户提供的课程资料回答。"

    def _extract_text(self, response: Any) -> str:
        try:
            choices = response.choices
            if not choices:
                raise RuntimeError(f"云端 API 返回中没有 choices：{response}")
            content = choices[0].message.content
        except (AttributeError, IndexError) as exc:
            raise RuntimeError(f"云端 API 返回格式无法解析：{response}") from exc
        if not content:
            raise RuntimeError(f"云端 API 返回内容为空：{response}")
        return content


class DeepSeekLLMProvider(CloudLLMProvider):
    """Compatibility alias for old explicit DeepSeek configuration."""

    provider_name = "deepseek"

    def __init__(self, api_key: str, model_name: str = LEGACY_DEEPSEEK_MODEL_NAME, base_url: str = LEGACY_DEEPSEEK_BASE_URL, client: Any | None = None) -> None:
        super().__init__(api_key=api_key, model_name=model_name, base_url=base_url, client=client)

    def _system_prompt(self) -> str:
        return "你是 DeepSeek 驱动的 AI 多模态学习助手，请优先依据用户提供的课程资料回答。"


class OllamaLLMProvider:
    """Local Ollama adapter for switching to a locally deployed model."""

    provider_name = "ollama"

    def __init__(
        self,
        model_name: str = OLLAMA_MODEL_NAME,
        base_url: str = OLLAMA_BASE_URL,
        client: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.client = client or self._default_client

    def generate(
        self,
        prompt: str,
        context_chunks: Sequence[str] | None = None,
    ) -> LLMResult:
        payload = {"model": self.model_name, "prompt": _build_user_message(prompt, context_chunks), "stream": False}
        response = self.client(f"{self.base_url}/api/generate", payload)
        text = str(response.get("response", "")).strip()
        if not text:
            raise RuntimeError(f"本地模型返回格式无法解析：{response}")
        return LLMResult(text=text, provider=self.provider_name, used_remote_model=False)

    def _default_client(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("无法连接本地模型服务，请确认已运行 Ollama 并已拉取本地模型。") from exc


def create_llm_provider(provider_name: str = DEFAULT_LLM_PROVIDER, api_key: str | None = None) -> LLMProvider:
    normalized = provider_name.strip().lower()
    generic_key = api_key if api_key is not None else os.getenv("LLM_API_KEY", "")
    generic_model = os.getenv("LLM_MODEL_NAME", LLM_MODEL_NAME)
    generic_base_url = os.getenv("LLM_BASE_URL", LLM_BASE_URL)
    qwen_key = api_key if api_key is not None else os.getenv("DASHSCOPE_API_KEY", "")
    deepseek_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY", "")

    if normalized == "auto":
        if generic_key:
            return CloudLLMProvider(api_key=generic_key, model_name=generic_model, base_url=generic_base_url)
        if qwen_key:
            return QwenLLMProvider(api_key=qwen_key)
        if deepseek_key:
            return DeepSeekLLMProvider(api_key=deepseek_key)
        return FallbackLLMProvider()

    if normalized in {"cloud", "api", "openai-compatible"}:
        if generic_key:
            return CloudLLMProvider(api_key=generic_key, model_name=generic_model, base_url=generic_base_url)
        return FallbackLLMProvider()

    if normalized == "qwen":
        if qwen_key:
            return QwenLLMProvider(api_key=qwen_key)
        return FallbackLLMProvider()

    if normalized == "deepseek":
        if deepseek_key:
            return DeepSeekLLMProvider(api_key=deepseek_key)
        return FallbackLLMProvider()

    if normalized in {"fallback", "local"}:
        return FallbackLLMProvider()

    if normalized in {"ollama", "local_model", "local-model"}:
        return OllamaLLMProvider()

    raise ValueError(f"Unsupported LLM provider: {provider_name}")


def _build_chat_messages(
    prompt: str,
    context_chunks: Sequence[str] | None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt or "你是 AI 多模态学习助手，请优先依据用户提供的课程资料回答。"},
        {"role": "user", "content": _build_user_message(prompt, context_chunks)},
    ]


def _build_user_message(prompt: str, context_chunks: Sequence[str] | None) -> str:
    chunks = [chunk.strip() for chunk in (context_chunks or []) if chunk.strip()]
    if not chunks:
        return prompt
    context = "\n\n".join(f"[资料片段 {index}]\n{chunk}" for index, chunk in enumerate(chunks, start=1))
    return f"课程资料：\n{context}\n\n用户问题：\n{prompt}"


def _escape_code_fence(text: str) -> str:
    # Retrieved course notes may contain Markdown headings; code fences keep
    # fallback evidence readable instead of turning references into giant titles.
    return text.replace("```", "` ` `")