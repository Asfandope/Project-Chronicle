from functools import lru_cache
from typing import List, Dict
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/magazine_extractor"
    
    # Application
    debug: bool = True
    log_level: str = "INFO"
    allowed_origins: List[str] = ["http://localhost:3000"]
    
    # Accuracy thresholds
    accuracy_threshold: float = 0.999
    brand_pass_rate_threshold: float = 0.95
    quarantine_threshold: float = 0.95
    
    # Field-level weights for accuracy calculation
    field_weights: Dict[str, float] = {
        "title": 0.3,
        "body": 0.4, 
        "contributors": 0.2,
        "media": 0.1
    }
    
    # Drift detection parameters
    drift_window_size: int = 10  # Number of recent issues to analyze
    drift_threshold: float = 0.02  # Accuracy drop that triggers drift detection
    consecutive_failures_threshold: int = 2
    
    # Gold set parameters
    min_gold_issues_per_brand: int = 10
    synthetic_augmentation_factor: int = 10  # Generate 10x synthetic variants
    holdout_percentage: float = 0.2  # 20% holdout for validation
    
    # Auto-tuning parameters
    max_tuning_frequency_hours: int = 24  # Max once per day per brand
    tuning_improvement_threshold: float = 0.01  # Minimum improvement to deploy
    max_tuning_attempts: int = 3
    
    # File paths
    gold_sets_directory: str = "data/gold_sets"
    evaluation_results_directory: str = "data/evaluation_results"
    tuning_logs_directory: str = "data/tuning_logs"
    
    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()