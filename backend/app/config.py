"""Application configuration.

Values are read from environment variables (and an optional .env file).
No LLM/embedding calls happen here — this is just the settings surface
described in PLAN.md #45.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-5"
    llm_api_key: str | None = None

    # Embeddings
    embedding_provider: str = "anthropic"
    embedding_model: str = "voyage-3"

    # Persistence
    database_path: str = "./data/weavex.db"

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Retrieval
    max_graph_depth: int = 2
    top_k: int = 10


settings = Settings()
