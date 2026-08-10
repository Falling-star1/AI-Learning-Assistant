import os


# Keep unit tests deterministic even when a local .env selects cloud, DeepSeek,
# or Ollama providers for manual Streamlit runs.
os.environ["LLM_PROVIDER"] = "fallback"
os.environ["LLM_API_KEY"] = ""
os.environ["DASHSCOPE_API_KEY"] = ""
os.environ["DEEPSEEK_API_KEY"] = ""
