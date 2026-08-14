"""Single source of truth for configuration.

Everything is an environment variable with a working default, so `git clone &&
docker compose up && python -m sift.ingest` works with no setup at all. The only
value you *must* supply is an LLM API key, and only when you want synthesised
answers -- ingestion and retrieval run entirely offline.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SIFT_",
        env_file=".env",
        extra="ignore",
    )

    # --- storage -----------------------------------------------------------
    # Host port is 5433 (see docker-compose.yml) to avoid colliding with a
    # Postgres you may already run locally.
    db_url: str = "postgresql://sift:sift@localhost:5433/sift"

    data_dir: Path = REPO_ROOT / "data"

    # --- embeddings --------------------------------------------------------
    # bge-small is 384-dim, ~130MB, and clearly better than MiniLM on retrieval
    # benchmarks. If you change this, update vector(384) in sql/001_schema.sql.
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384
    embedding_batch_size: int = 64
    # bge models were trained with an asymmetric prefix on the *query* side only.
    # Omitting this is a silent ~5-10% retrieval quality loss.
    query_instruction: str = "Represent this sentence for searching relevant passages: "

    # --- chunking ----------------------------------------------------------
    chunk_max_chars: int = 1800
    chunk_overlap: int = 150
    chunk_combine_under: int = 500  # merge tiny sections into their neighbour

    # --- retrieval ---------------------------------------------------------
    vector_top_k: int = 20  # candidates from dense search
    keyword_top_k: int = 20  # candidates from BM25 / full-text
    final_top_k: int = 6  # what actually reaches the LLM
    rrf_k: int = 60  # reciprocal-rank-fusion constant

    # --- LLM ---------------------------------------------------------------
    llm_provider: str = "anthropic"  # anthropic | openai
    anthropic_model: str = "claude-sonnet-5"
    openai_model: str = "gpt-4o-mini"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.0  # determinism matters for an eval gate

    # --- ingest ------------------------------------------------------------
    ingest_workers: int = 4
    parse_timeout_seconds: int = 300

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def failed_log(self) -> Path:
        return self.data_dir / "failed_documents.log"

    def api_key(self) -> str | None:
        """API keys are read WITHOUT the SIFT_ prefix -- they are the vendors'
        conventional names, and reusing them means no extra setup for the user."""
        if self.llm_provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        return os.getenv("OPENAI_API_KEY")


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
