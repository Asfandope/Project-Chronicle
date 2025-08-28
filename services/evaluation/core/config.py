from functools import lru_cache
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    debug: bool = True
    log_level: str = "INFO"
    allowed_origins: List[str] = ["http://localhost:3000"]

    # Database
    database_url: str = (
        "postgresql://postgres:postgres@localhost:5432/magazine_extractor"
    )

    # External Services
    model_service_url: str = "http://localhost:8001"
    orchestrator_url: str = "http://localhost:8000"

    # Evaluation Settings
    evaluation_batch_size: int = 10
    evaluation_timeout: int = 300
    min_confidence_threshold: float = 0.7

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
