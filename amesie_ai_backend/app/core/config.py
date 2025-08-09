"""
Configuration settings for Amesie AI Backend
"""

import os
from typing import List, Optional
from pydantic import BaseSettings, validator
from pydantic.types import SecretStr


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Amesie AI Backend"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Amesie AI API"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]
    
    # Security
    SECRET_KEY: SecretStr = SecretStr("your-secret-key-change-in-production")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/amesie_ai"
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379"
    
    # Model Configuration
    MODEL_NAME: str = "mistralai/Mistral-7B-Instruct-v0.2"
    MODEL_QUANTIZATION: str = "4bit"  # 4bit, 8bit, or none
    DEVICE: str = "cuda"  # cuda, cpu, or auto
    MAX_LENGTH: int = 2048
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    TOP_K: int = 50
    
    # Performance
    BATCH_SIZE: int = 1
    MAX_CONCURRENT_REQUESTS: int = 10
    
    # Monitoring
    SENTRY_DSN: Optional[str] = None
    PROMETHEUS_PORT: int = 9090
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    
    # Cache
    CACHE_TTL: int = 3600  # 1 hour
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = ["txt", "pdf", "doc", "docx", "csv", "json"]
    
    # WebSocket
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MAX_CONNECTIONS: int = 100
    
    @validator("CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    @validator("DEVICE")
    def validate_device(cls, v):
        valid_devices = ["cuda", "cpu", "auto"]
        if v not in valid_devices:
            raise ValueError(f"Device must be one of {valid_devices}")
        return v
    
    @validator("MODEL_QUANTIZATION")
    def validate_quantization(cls, v):
        valid_quantizations = ["4bit", "8bit", "none"]
        if v not in valid_quantizations:
            raise ValueError(f"Quantization must be one of {valid_quantizations}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()