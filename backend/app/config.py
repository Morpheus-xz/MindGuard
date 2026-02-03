"""Configuration settings for MindGuard backend."""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # App
    app_name: str = "MindGuard API"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "change-me-in-production"

    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_key: Optional[str] = None

    # Pinecone
    pinecone_api_key: Optional[str] = None
    pinecone_environment: str = "us-east-1"
    pinecone_index_name: str = "mindguard"

    # ML
    bert_model_path: str = "models/bert"
    lstm_model_path: str = "models/lstm"
    max_text_length: int = 512

    # CORS
    frontend_url: str = "http://localhost:5173"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()