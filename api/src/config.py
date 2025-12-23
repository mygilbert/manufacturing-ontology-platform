"""
API 서버 설정
"""
import os
from typing import Optional, List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """API 서버 설정"""

    # 서버
    app_name: str = "Manufacturing Ontology Platform API"
    app_version: str = "1.0.0"
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"

    # PostgreSQL (AGE)
    postgres_host: str = os.getenv("POSTGRES_HOST", "postgres")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_user: str = os.getenv("POSTGRES_USER", "ontology")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "ontology123")
    postgres_db: str = os.getenv("POSTGRES_DB", "manufacturing")

    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def postgres_async_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # TimescaleDB
    timescale_host: str = os.getenv("TIMESCALE_HOST", "timescaledb")
    timescale_port: int = int(os.getenv("TIMESCALE_PORT", "5432"))
    timescale_user: str = os.getenv("TIMESCALE_USER", "timescale")
    timescale_password: str = os.getenv("TIMESCALE_PASSWORD", "timescale123")
    timescale_db: str = os.getenv("TIMESCALE_DB", "measurements")

    @property
    def timescale_url(self) -> str:
        return f"postgresql://{self.timescale_user}:{self.timescale_password}@{self.timescale_host}:{self.timescale_port}/{self.timescale_db}"

    @property
    def timescale_async_url(self) -> str:
        return f"postgresql+asyncpg://{self.timescale_user}:{self.timescale_password}@{self.timescale_host}:{self.timescale_port}/{self.timescale_db}"

    # Redis
    redis_host: str = os.getenv("REDIS_HOST", "redis")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD", "redis123")
    redis_db: int = int(os.getenv("REDIS_DB", "0"))

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # Kafka
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092")

    # CORS
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8080", "*"]

    # JWT (향후 인증 추가 시)
    jwt_secret: str = os.getenv("JWT_SECRET", "your-super-secret-key-change-in-production")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24  # 24시간

    class Config:
        env_file = ".env"


settings = Settings()
