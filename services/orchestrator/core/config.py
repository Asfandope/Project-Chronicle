from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:postgres@localhost:5432/magazine_extractor"
    
    # Redis/Celery
    redis_url: str = "redis://localhost:6379"
    celery_result_backend: Optional[str] = None  # Uses redis_url if not set
    celery_broker_url: Optional[str] = None  # Uses redis_url if not set
    
    # External Services
    model_service_url: str = "http://localhost:8001"
    evaluation_service_url: str = "http://localhost:8002"
    
    # Application
    debug: bool = True
    log_level: str = "INFO"
    log_format: str = "json"  # json or console
    root_path: str = ""
    allowed_origins: List[str] = ["http://localhost:3000"]
    allowed_hosts: List[str] = ["*"]
    
    # File Processing
    input_directory: str = "data/input"
    output_directory: str = "data/output"
    quarantine_directory: str = "data/quarantine"
    temp_directory: str = "temp"
    
    # File Watcher
    enable_file_watcher: bool = True
    file_watcher_poll_interval: float = 1.0  # seconds
    supported_file_extensions: List[str] = [".pdf"]
    
    # Processing Limits
    max_file_size_mb: int = 100
    max_pages_per_issue: int = 500
    processing_timeout_minutes: int = 30
    max_concurrent_jobs: int = 10
    
    # Quality Thresholds
    accuracy_threshold: float = 0.999
    quarantine_threshold: float = 0.95
    confidence_threshold: float = 0.85
    
    # Retry Configuration
    max_retries: int = 3
    retry_delay_seconds: int = 60
    exponential_backoff: bool = True
    
    # Health Check Configuration
    health_check_timeout: int = 5
    health_check_interval: int = 30
    
    # Security
    enable_cors: bool = True
    enable_rate_limiting: bool = False
    rate_limit_requests_per_minute: int = 100
    
    # Monitoring
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_tracing: bool = False
    jaeger_endpoint: Optional[str] = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Set celery URLs from redis_url if not explicitly set
        if not self.celery_result_backend:
            self.celery_result_backend = self.redis_url
        if not self.celery_broker_url:
            self.celery_broker_url = self.redis_url
        
        # Ensure directories exist
        for directory in [
            self.input_directory,
            self.output_directory, 
            self.quarantine_directory,
            self.temp_directory
        ]:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    @property
    def async_database_url(self) -> str:
        """Get async database URL for SQLAlchemy."""
        return self.database_url.replace("postgresql://", "postgresql+asyncpg://")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()