"""
Configuration management for the Neural Network API
"""
import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Settings
    APP_NAME: str = "Neural Network API"
    APP_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    RELOAD: bool = DEBUG
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]
    
    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/neuralnet")
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Model Settings
    MODEL_NAME: str = "neural-chat-model"
    MODEL_PATH: str = "./models"
    MAX_SEQUENCE_LENGTH: int = 2048
    VOCAB_SIZE: int = 50000
    EMBEDDING_DIM: int = 768
    NUM_ATTENTION_HEADS: int = 12
    NUM_LAYERS: int = 12
    HIDDEN_DIM: int = 3072
    DROPOUT_RATE: float = 0.1
    
    # Training Settings
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 5e-5
    WARMUP_STEPS: int = 1000
    MAX_GRAD_NORM: float = 1.0
    WEIGHT_DECAY: float = 0.01
    
    # Inference Settings
    MAX_RESPONSE_LENGTH: int = 512
    TEMPERATURE: float = 0.7
    TOP_K: int = 50
    TOP_P: float = 0.9
    REPETITION_PENALTY: float = 1.1
    
    # Performance Settings
    DEVICE: str = "cuda" if os.getenv("CUDA_AVAILABLE", "true").lower() == "true" else "cpu"
    MIXED_PRECISION: bool = True
    GRADIENT_CHECKPOINTING: bool = True
    DATALOADER_NUM_WORKERS: int = 4
    
    # Monitoring Settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    LOG_LEVEL: str = "INFO"
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds
    
    # File Upload Settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [".txt", ".md", ".json", ".csv"]
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()