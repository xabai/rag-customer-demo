"""
深度健康检查 —— /api/health 返回 DB + 向量库 + LLM 三项探针

生产级应该每分钟跑一次，任何一项 fail 就给告警系统触发
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .config import get_settings
_settings = get_settings()

from fastapi import APIRouter

_log = logging.getLogger("rag.health")
router = APIRouter()


def _check_db() -> dict:
    """检查数据库连接（SELECT 1）"""
    start = time.time()
    try:
        from .db_engine import engine
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        return {"status": "ok", "latency_ms": int((time.time() - start) * 1000)}
    except Exception as e:
        return {"status": "fail", "latency_ms": int((time.time() - start) * 1000),
                "error": str(e)[:200]}


def _check_llm(timeout: float = 5.0) -> dict:
    """检查 Ollama / LLM 可达性（同步 httpx）"""
    start = time.time()
    try:
        import httpx
        url = f"{_settings.OLLAMA_HOST.rstrip('/')}/api/tags"
        with httpx.Client(timeout=timeout) as c:
            resp = c.get(url)
            ok = resp.status_code == 200
        return {"status": "ok" if ok else "fail",
                "latency_ms": int((time.time() - start) * 1000)}
    except ImportError:
        return {"status": "fail", "latency_ms": int((time.time() - start) * 1000),
                "error": "httpx not installed"}
    except Exception as e:
        return {"status": "fail", "latency_ms": int((time.time() - start) * 1000),
                "error": str(e)[:200]}


def _check_vector_db() -> dict:
    """检查向量库（Chroma）"""
    start = time.time()
    try:
        import chromadb
        client = chromadb.PersistentClient(path=_settings.VECTOR_DB_PATH)
        collections = client.list_collections()
        return {"status": "ok", "latency_ms": int((time.time() - start) * 1000),
                "collections": len(collections)}
    except ImportError:
        return {"status": "skip", "latency_ms": int((time.time() - start) * 1000),
                "error": "chromadb not installed"}
    except Exception as e:
        return {"status": "fail", "latency_ms": int((time.time() - start) * 1000),
                "error": str(e)[:200]}


def _check_redis() -> dict:
    """检查 Redis（如果配置了）"""
    if not _settings.REDIS_URL:
        return {"status": "skip", "error": "Redis not configured"}
    start = time.time()
    try:
        import redis as _redis
        r = _redis.from_url(_settings.REDIS_URL, socket_timeout=2)
        r.ping()
        return {"status": "ok", "latency_ms": int((time.time() - start) * 1000)}
    except ImportError:
        return {"status": "skip", "error": "redis-py not installed"}
    except Exception as e:
        return {"status": "fail", "latency_ms": int((time.time() - start) * 1000),
                "error": str(e)[:200]}


@router.get("/api/health")
async def health_check(detailed: bool = True) -> dict[str, Any]:
    """深度健康检查端点"""
    from .metrics import heartbeat
    heartbeat()

    checks = {
        "service": {"status": "ok", "app": _settings.APP_NAME, "env": _settings.APP_ENV},
        "database": _check_db(),
        "llm": _check_llm(),
        "vector_db": _check_vector_db(),
        "redis": _check_redis(),
    }
    overall = "ok" if all(c["status"] in ("ok", "skip") for c in checks.values()) else "degraded"
    checks["overall"] = overall
    return checks
