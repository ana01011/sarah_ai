"""
Security utilities for Amesie AI Backend
"""

import time
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from fastapi import HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import redis
from app.core.config import settings

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT token security
security = HTTPBearer()

# Redis for rate limiting
redis_client = redis.from_url(settings.REDIS_URL)


class Token(BaseModel):
    access_token: str
    token_type: str
    expires_at: datetime


class TokenData(BaseModel):
    username: Optional[str] = None
    user_id: Optional[str] = None


class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def is_allowed(self, key: str, limit: int, window: int) -> bool:
        """Check if request is allowed within rate limit"""
        current = int(time.time())
        window_start = current - window
        
        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        count = self.redis.zcard(key)
        
        if count >= limit:
            return False
        
        # Add current request
        self.redis.zadd(key, {str(current): current})
        self.redis.expire(key, window)
        
        return True
    
    def get_remaining(self, key: str, limit: int, window: int) -> int:
        """Get remaining requests allowed"""
        current = int(time.time())
        window_start = current - window
        
        # Remove old entries
        self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        count = self.redis.zcard(key)
        
        return max(0, limit - count)


rate_limiter = RateLimiter(redis_client)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY.get_secret_value(), algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY.get_secret_value(), algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


async def get_current_user(credentials: HTTPAuthorizationCredentials = security) -> Optional[Dict[str, Any]]:
    """Get current user from JWT token"""
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload


def generate_api_key() -> str:
    """Generate a secure API key"""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Hash API key for storage"""
    return hashlib.sha256(api_key.encode()).hexdigest()


def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    # Remove potentially dangerous characters
    dangerous_chars = ['<', '>', '"', "'", '&', ';', '(', ')', '{', '}', '[', ']']
    sanitized = text
    for char in dangerous_chars:
        sanitized = sanitized.replace(char, '')
    
    # Limit length
    if len(sanitized) > 10000:  # 10KB limit
        sanitized = sanitized[:10000]
    
    return sanitized.strip()


def validate_file_upload(filename: str, file_size: int) -> bool:
    """Validate file upload"""
    # Check file size
    if file_size > settings.MAX_FILE_SIZE:
        return False
    
    # Check file extension
    file_ext = filename.split('.')[-1].lower()
    if file_ext not in settings.ALLOWED_FILE_TYPES:
        return False
    
    return True


async def check_rate_limit(request: Request, user_id: Optional[str] = None) -> bool:
    """Check rate limit for request"""
    # Use IP address or user ID as key
    key = f"rate_limit:{user_id or request.client.host}"
    
    # Check per-minute limit
    if not rate_limiter.is_allowed(f"{key}:minute", settings.RATE_LIMIT_PER_MINUTE, 60):
        return False
    
    # Check per-hour limit
    if not rate_limiter.is_allowed(f"{key}:hour", settings.RATE_LIMIT_PER_HOUR, 3600):
        return False
    
    return True


def get_rate_limit_headers(request: Request, user_id: Optional[str] = None) -> Dict[str, str]:
    """Get rate limit headers for response"""
    key = f"rate_limit:{user_id or request.client.host}"
    
    minute_remaining = rate_limiter.get_remaining(f"{key}:minute", settings.RATE_LIMIT_PER_MINUTE, 60)
    hour_remaining = rate_limiter.get_remaining(f"{key}:hour", settings.RATE_LIMIT_PER_HOUR, 3600)
    
    return {
        "X-RateLimit-Limit-Minute": str(settings.RATE_LIMIT_PER_MINUTE),
        "X-RateLimit-Remaining-Minute": str(minute_remaining),
        "X-RateLimit-Limit-Hour": str(settings.RATE_LIMIT_PER_HOUR),
        "X-RateLimit-Remaining-Hour": str(hour_remaining),
    }