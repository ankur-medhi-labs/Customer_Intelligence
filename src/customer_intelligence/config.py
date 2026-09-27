"""Application settings, loaded from environment variables / .env."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM / embedding endpoint (OpenAI-compatible; see README for the AWS Bedrock substitution note)
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    embedding_model: str
    embedding_dimensions: int = 1024

    # Generation
    gen_temperature: float = 0.2
    max_tokens: int = 1024

    # RAG
    retrieval_top_k: int = 4
    retrieval_min_score: float = 0.35
    chunk_size: int = 400
    chunk_overlap: int = 60

    # Paths
    artifacts_dir: str = "artifacts"
    kb_dir: str = "data/kb"
    raw_dataset_path: str = "docs/Chat Dataset 1.xlsx"
    processed_dir: str = "data/processed"
    synthetic_dir: str = "data/synthetic"


settings = Settings()
