from functools import lru_cache
from typing import List

import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    debug: bool = True
    log_level: str = "INFO"
    allowed_origins: List[str] = ["http://localhost:3000"]

    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Model paths and cache
    model_cache_dir: str = "models"
    layout_model_name: str = "microsoft/layoutlm-v3-base"
    ner_model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"

    # Processing parameters
    max_image_size: int = 2048
    ocr_confidence_threshold: float = 0.7
    layout_confidence_threshold: float = 0.8
    batch_size: int = 8 if torch.cuda.is_available() else 4

    # OCR settings
    tesseract_config: str = "--oem 3 --psm 6"

    # Timeouts
    model_loading_timeout: int = 300  # 5 minutes
    inference_timeout: int = 120  # 2 minutes per request

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
