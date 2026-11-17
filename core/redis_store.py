"""
Redis 适配层 —— session 白名单 + SlidingWindow 限流

设计：
- Redis 可用则用，不可用自动 fallback 到 SQLite + 内存（已有的实现）
- 统一 import：`from core.redis_store import redis, is_redis_enabled`
- 横向扩展时所有 worker 共享 Redis，session 和限流状态全局同步

key 规范：
  rag:session:{session_id}  → {user_id, expires_at}     (TTL = 剩余秒数)
  rag:rate_limit:{path}:{ip} → ZADD timestamp 排序集合   (ZREMRANGEBYSCORE 清理旧)
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .config import get_settings

_settings = get_settings()
_log = logging.getLogger("rag.redis_store")

# 延迟 import（Redis 不可用时不崩）
redis = None
is_redis_enabled = False

if _settings.REDIS_URL:
    try:
        import redis as _redis  # type: ignore
        redis = _redis.from_url(_settings.REDIS_URL, decode_responses=True,
                                 socket_timeout=2, socket_connect_timeout=2)
        redis.ping()
        is_redis_enabled = True
        _log.info("Redis connected → %s", _settings.REDIS_URL)
    except Exception as e:
        _log.warning("Redis init failed, fallback to local: %s", e)
        redis = None
        is_redis_enabled = False
else:
    _log.info("REDIS_URL not set, using fallback (SQLite sessions + in-memory rate limit)")


# ================================================================
# Session 白名单（横向扩展必须全局同步）
# ================================================================

def redis_add_session(session_id: str, user_id: int, expires_at: float,
                      ip: str = "", user_agent: str = "") -> bool:
    """登记一个 session，TTL 自动根据 expires_at 计算"""
    if not is_redis_enabled or not redis:
        return False
    ttl = max(1, int(expires_at - time.time()))
    key = f"rag:session:{session_id}"
    try:
        redis.hset(key, mapping={
            "user_id": str(user_id),
            "expires_at": str(expires_at),
            "ip": ip,
            "user_agent": (user_agent or "")[:500],
        })
        redis.expire(key, ttl)
        return True
    except Exception as e:
        _log.warning("redis_add_session failed: %s", e)
        return False


def redis_session_exists(session_id: str) -> bool:
    """检查 session 是否存在且未过期"""
    if not is_redis_enabled or not redis:
        return False
    key = f"rag:session:{session_id}"
    try:
        # 如果 key 不存在自动过期
        return redis.exists(key) > 0
    except Exception:
        return False


def redis_revoke_session(session_id: str) -> bool:
    """注销 session"""
    if not is_redis_enabled or not redis:
        return False
    try:
        redis.delete(f"rag:session:{session_id}")
        return True
    except Exception:
        return False


def redis_revoke_all_user_sessions(user_id: int) -> int:
    """踢掉某用户所有 session（force-logout）"""
    if not is_redis_enabled or not redis:
        return 0
    try:
        # 扫 rag:session:* 里 user_id 匹配的 —— Redis 没有原生按 field 查询 hash，
        # 所以维护一个反向索引 set rag:user_sessions:{user_id}
        members = redis.smembers(f"rag:user_sessions:{user_id}")
        count = 0
        for sid in members:
            redis.delete(f"rag:session:{sid}")
            count += 1
        redis.delete(f"rag:user_sessions:{user_id}")
        return count
    except Exception as e:
        _log.warning("redis_revoke_all failed: %s", e)
        return 0


# ================================================================
# 限流 —— Sliding Window 用 Redis ZSET 实现
# ================================================================

class RedisSlidingWindowLimiter:
    """Redis Sliding Window 限流（替代内存 deque 实现）"""

    def __init__(self):
        self.enabled = is_redis_enabled and redis is not None

    def check(self, key: str, limit: int, window_seconds: float) -> tuple[bool, int]:
        """返回 (allowed, remaining)"""
        if not self.enabled:
            return True, limit  # 没 Redis 就放行（靠内存 fallback）
        now = time.time()
        try:
            zkey = f"rag:rl:{key}"
            # 清掉窗口外的旧记录
            redis.zremrangebyscore(zkey, "-inf", now - window_seconds)
            # 当前窗口内的请求数
            count = redis.zcard(zkey)
            if count >= limit:
                return False, 0
            # 记录本次请求，score=timestamp，value=uuid
            redis.zadd(zkey, {f"{now}:{id(self)}": now})
            redis.expire(zkey, int(window_seconds) + 10)
            return True, limit - int(count) - 1
        except Exception as e:
            _log.warning("rl_check failed: %s", e)
            return True, limit  # Redis 挂了不阻断


limiter = RedisSlidingWindowLimiter()


# ================================================================
# 通用 KV（健康检查 + 预热脚本用）
# ================================================================

def redis_set(key: str, value: Any, ttl: int = 0) -> bool:
    if not is_redis_enabled or not redis:
        return False
    try:
        if ttl > 0:
            redis.setex(key, ttl, str(value))
        else:
            redis.set(key, str(value))
        return True
    except Exception:
        return False


def redis_get(key: str) -> str | None:
    if not is_redis_enabled or not redis:
        return None
    try:
        return redis.get(key)
    except Exception:
        return None
