"""Application configuration.

Values are read from environment variables (and an optional .env file).
No LLM/embedding calls happen here — this is just the settings surface
described in PLAN.md #45.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM — Groq's OpenAI-compatible chat-completions API, serving the
    # open-weight gpt-oss-120b model. External API call, not self-hosted,
    # so it fits CLAUDE.md's "an LLM API may be used" allowance the same
    # way Claude/GPT would.
    llm_provider: str = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    llm_api_key: str | None = None

    # Embeddings — HuggingFace's hosted Inference API (external call,
    # not a locally-run model), using a small well-supported
    # sentence-transformers model. Needs its own key, separate from
    # llm_api_key (a different provider).
    embedding_provider: str = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_api_key: str | None = None

    # Persistence
    database_path: str = "./data/weavex.db"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Retrieval
    max_graph_depth: int = 2
    top_k: int = 10


settings = Settings()
