"""
统一配置管理 —— pydantic-settings
优先级：环境变量 > .env 文件 > 代码默认值
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---- 支持从项目根目录的 .env 文件读 ----
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """应用配置，字段名即环境变量名（大写）"""

    # ============ 基础 ============
    APP_NAME: str = "rag-bot"
    APP_ENV: str = "dev"                    # dev / staging / prod
    DEBUG: bool = True
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8008

    # ============ 数据库 ============
    DB_DRIVER: str = "postgresql"           # postgresql / sqlite
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "rag_bot"
    DB_USER: str = "postgres"
    DB_PASSWORD: str = ""                   # 从 .env / 环境变量注入，勿写死
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_SQLITE_PATH: str = str(_PROJECT_ROOT / "data" / "chat.db")

    # ============ Redis ============
    REDIS_URL: str = ""                     # 留空则禁用，限流/session 退化为内存

    # ============ 安全 ============
    JWT_SECRET: str = ""                    # 从 .env / 环境变量注入，勿写死
    JWT_ALG: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 360
    JWT_REFRESH_EXPIRE_DAYS: int = 30

    CORS_ORIGINS: str = "http://localhost:8008,http://127.0.0.1:8008"

    # ============ 密码策略 ============
    PASSWORD_MIN_LEN: int = 8
    PASSWORD_REQUIRE_UPPER: bool = True
    PASSWORD_REQUIRE_LOWER: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = False
    PASSWORD_HISTORY_CHECK: int = 3         # 最近 N 次密码不能重复

    # ============ 登录锁定 ============
    LOGIN_MAX_ATTEMPTS: int = 5
    LOGIN_LOCKOUT_MINUTES: int = 10

    # ============ LLM / Ollama ============
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_DEFAULT_MODEL: str = "qwen2.5:7b"
    OLLAMA_FALLBACK_MODEL: str = "qwen2.5:3b"
    LLM_TIMEOUT: int = 120

    # ============ RAG ============
    VECTOR_DB_PATH: str = str(_PROJECT_ROOT / "data" / "chroma")
    MAX_CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.5

    # ============ 日志 ============
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = str(_PROJECT_ROOT / "data" / "logs" / "rag.log")
    LOG_MAX_BYTES: int = 10 * 1024 * 1024   # 10MB
    LOG_BACKUP_COUNT: int = 5

    # ============ Prometheus / Tracing ============
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090                # 预留独立 metrics 端口（目前挂主进程）
    TRACING_ENABLED: bool = False
    OTLP_ENDPOINT: str = "http://localhost:4317"
    OTLP_SERVICE_NAME: str = "rag-bot"

    # ============ 告警 ============
    ALERT_WEBHOOK_URL: str = ""             # 飞书/Slack webhook
    ALERT_ERROR_RATE_THRESHOLD: float = 0.05  # 5% 错误率触发
    ALERT_P99_LATENCY_THRESHOLD_MS: int = 3000

    # ============ 异步任务 ============
    ASYNC_QUEUE_ENABLED: bool = True
    ASYNC_WORKER_CONCURRENCY: int = 4

    # ============ 内容安全 ============
    CONTENT_SAFETY_ENABLED: bool = False
    CONTENT_BLOCKED_WORDS: str = ""         # 逗号分隔

    model_config = SettingsConfigDict(
        env_file=str(_PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ---------- 校验 ----------
    @model_validator(mode="after")
    def _check_secrets(self):
        """非 dev 环境必须显式配置数据库口令与 JWT 密钥，避免空值/默认值上线"""
        if self.APP_ENV != "dev":
            if not self.DB_PASSWORD:
                raise ValueError("DB_PASSWORD 未配置：请在 .env 中设置")
            if not self.JWT_SECRET or "CHANGE-ME" in self.JWT_SECRET:
                raise ValueError("JWT_SECRET 未配置或仍为默认值：请在 .env 中设置随机字符串")
        return self

    # ---------- 计算属性 ----------
    @computed_field  # type: ignore[prop-decorator]
    @property
    def DATABASE_URL(self) -> str:
        """SQLAlchemy 兼容的连接串"""
        if self.DB_DRIVER == "sqlite":
            return f"sqlite:///{self.DB_SQLITE_PATH}"
        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASSWORD}"
            f"@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """单例获取配置"""
    return Settings()
