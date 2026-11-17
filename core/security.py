# -*- coding: utf-8 -*-
"""安全中间件：限流 + 审计日志 + 敏感操作二次验证

不引入 slowapi/redis 等新依赖，纯 Python + SQLite 足够单机部署用。
"""
from __future__ import annotations

import functools
import inspect
import threading
import time
from collections import defaultdict, deque
from typing import Callable

from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from .desensitize import desensitize as _desens  # A2 脱敏

import db
from core.logger import get_logger, set_trace_id, get_trace_id, log_performance

logger = get_logger("security")


# ============================================================================
# 1. 限流：内存滑动窗口（per-IP）
# ============================================================================

class SlidingWindowRateLimiter:
    """每个 (key, time_window) 维护一个 deque，存最近请求时间戳"""

    def __init__(self):
        self._buckets: dict[tuple[str, float], deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, limit: int, window_seconds: float) -> tuple[bool, int, float]:
        """
        返回 (allowed, remaining, retry_after_seconds)
        """
        now = time.time()
        cutoff = now - window_seconds
        bucket = self._buckets[(key, window_seconds)]

        with self._lock:
            # 清窗口外的旧戳
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                retry = window_seconds - (now - bucket[0])
                return False, 0, max(1.0, retry)
            bucket.append(now)
            remaining = limit - len(bucket)
            return True, remaining, 0.0


rate_limiter = SlidingWindowRateLimiter()


# 各接口限流策略：(limit, window_seconds)
RATE_LIMITS: dict[str, tuple[int, float]] = {
    "/api/auth/login":          (20,   60),    # 20 次 / 分钟（防暴力破解；锁定阈值另设 5 次）
    "/api/auth/register":       (3,    300),   # 3 次 / 5 分钟
    "/api/chat":                (10,   60),    # 10 次 / 分钟（推理贵）
    "/api/retrieval/test":      (10,   60),
    "/api/upload":              (20,   60),    # 20 次 / 分钟（CPU/存储密集）
    "/api/upload_image":        (20,   60),
    "/api/users":               (30,   60),
    "/api/users/2":             (30,   60),    # PUT/DELETE/POST force-logout 都匹配
    "/api/memories":            (120,  60),
    "/api/agents":              (60,   60),
}


def _get_rate_limit_config(path: str, method: str) -> tuple[int, float] | None:
    # 精确匹配 → 前缀匹配（带 {id} 的路径配父路径）
    if path in RATE_LIMITS:
        return RATE_LIMITS[path]
    # 前缀匹配（去掉末尾 /xxx 片段，匹配 /api/users/{id} → /api/users）
    for pattern, conf in RATE_LIMITS.items():
        if pattern.endswith("/2"):  # 特殊标记通配
            base = pattern[:-2]
            if path.startswith(base):
                return conf
    return None


def _client_ip(request: Request) -> str:
    """取客户端真实 IP（信任 X-Forwarded-For 的第一个）"""
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    if request.headers.get("x-real-ip"):
        return request.headers["x-real-ip"].strip()
    return request.client.host if request.client else "unknown"


# ============================================================================
# 2. 审计中间件：自动为所有请求写入 audit_logs + trace_id
# ============================================================================

# 敏感动作路径 → audit action 代码映射（稳定枚举，前端筛选靠这个）
ACTION_MAP: dict[str, str] = {
    "/api/auth/login":                          "auth.login",
    "/api/auth/register":                       "auth.register",
    "/api/auth/logout":                         "auth.logout",
    "/api/auth/change-password":                "auth.change_password",
    "/api/users":                               "user.list|user.create",
    "/api/users/2":                             "user.update|user.delete",
    "/api/users/2/toggle":                      "user.toggle",
    "/api/users/2/force-logout":                "user.force_logout",
    "/api/upload":                              "kb.upload",
    "/api/upload_url":                          "kb.upload_url",
    "/api/kb":                                  "kb.create|kb.delete|kb.list",
    "/api/documents":                           "kb.doc.list|kb.doc.delete",
    "/api/agents":                              "agent.list|agent.create",
    "/api/agents/2":                            "agent.update|agent.delete",
    "/api/agents/2/toggle":                     "agent.toggle",
    "/api/memories":                            "memory.list",
    "/api/memories/2":                          "memory.delete",
    "/api/settings":                            "sys.settings.update|sys.settings.read",
    "/api/qa_log/export":                       "sys.export.qa_log",
    "/api/sessions/2":                          "sys.export.session",
    "/api/chat":                                "chat.request",
}


def _resolve_action(path: str, method: str) -> tuple[str, str, str]:
    """返回 (action, resource_type, resource_id)"""
    # 先找通配
    base_id = ""
    clean = path
    parts = path.split("/")
    # /api/users/123 → base=/api/users, id=123
    for i, seg in enumerate(parts):
        if seg.isdigit():
            base_id = seg
            parts = parts[:i] + ["2"] + parts[i+1:]
            clean = "/".join(parts)
            break

    if clean in ACTION_MAP:
        candidates = ACTION_MAP[clean].split("|")
        # candidates 里按 method 后缀匹配
        method_map = {"GET": "read|list", "POST": "create", "PUT": "update|toggle", "DELETE": "delete"}
        suffixes = method_map.get(method, "")
        for c in candidates:
            if c.endswith(tuple(suffixes.split("|"))) or c.split(".")[-1] == method.lower():
                action = c
                break
        else:
            action = candidates[0]
        resource_type = action.split(".")[0]
        return action, resource_type, base_id

    # 默认：method + path
    return f"{method.lower()} {path}", "api", base_id


class AuditMiddleware(BaseHTTPMiddleware):
    """
    为每个请求：
    1. 生成 trace_id（挂到 request.state）
    2. 调用限流检查
    3. 正常放行；异常也会写一条审计（result=fail）
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # ---- 1. trace_id ----
        tid = f"{int(time.time()*1000):x}{__import__('uuid').uuid4().hex[:6]}"
        request.state.trace_id = tid
        set_trace_id(tid)

        # ---- 2. 限流（敏感接口）----
        ip = _client_ip(request)
        conf = _get_rate_limit_config(request.url.path, request.method)
        if conf:
            limit, window = conf
            allowed, remaining, retry = rate_limiter.check(f"{ip}:{request.url.path}", limit, window)
            if not allowed:
                after = int(retry)
                # 审计：限流被拒
                action, rtype, rid = _resolve_action(request.url.path, request.method)
                db.write_audit(
                    action=f"rate_limited.{action}",
                    resource_type=rtype, resource_id=rid,
                    ip=ip, result="fail",
                    detail=_desens(f"IP {ip} hit {request.method} {request.url.path} limit {limit}/{window}s"),
                )
                logger.warning("[rate-limited] %s %s from %s", request.method, request.url.path, ip)
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"请求过于频繁，请 {after} 秒后重试", "retry_after": after},
                    headers={"Retry-After": str(after)},
                )

        # ---- 3. 调用下游 ----
        start = time.time()
        try:
            response = await call_next(request)
            elapsed_ms = int((time.time() - start) * 1000)
            request.state.elapsed_ms = elapsed_ms
            request.state.status_code = response.status_code

            # 对非公开接口，写审计（登录除外，后面单独处理）
            auth_paths = ("/api/auth/login", "/api/auth/register", "/api/models", "/api/health", "/")
            if request.url.path not in auth_paths:
                actor = getattr(request.state, "actor", None)
                action, rtype, rid = _resolve_action(request.url.path, request.method)
                db.write_audit(
                    action=action,
                    resource_type=rtype, resource_id=rid,
                    actor_user_id=actor.get("id") if actor else None,
                    actor_username=actor.get("username") if actor else None,
                    ip=ip,
                    user_agent=request.headers.get("user-agent", "")[:200],
                    result="success" if response.status_code < 400 else "fail",
                    detail=_desens(f"status={response.status_code} elapsed={elapsed_ms}ms"),
                )
            return response

        except HTTPException as exc:
            elapsed_ms = int((time.time() - start) * 1000)
            actor = getattr(request.state, "actor", None)
            action, rtype, rid = _resolve_action(request.url.path, request.method)
            db.write_audit(
                action=action,
                resource_type=rtype, resource_id=rid,
                actor_user_id=actor.get("id") if actor else None,
                actor_username=actor.get("username") if actor else None,
                ip=ip,
                user_agent=request.headers.get("user-agent", "")[:200],
                result="fail",
                detail=_desens(f"HTTPException {exc.status_code}: {exc.detail}"),
            )
            return JSONResponse(status_code=exc.status_code, content={"detail": str(exc.detail)})

        except Exception as exc:
            elapsed_ms = int((time.time() - start) * 1000)
            logger.exception("[unhandled] %s %s: %s", request.method, request.url.path, exc)
            actor = getattr(request.state, "actor", None)
            action, rtype, rid = _resolve_action(request.url.path, request.method)
            db.write_audit(
                action="error.unhandled." + action,
                resource_type=rtype, resource_id=rid,
                actor_user_id=actor.get("id") if actor else None,
                actor_username=actor.get("username") if actor else None,
                ip=ip,
                user_agent=request.headers.get("user-agent", "")[:200],
                result="error",
                detail=_desens(f"{exc.__class__.__name__}: {exc}"),
            )
            return JSONResponse(
                status_code=500,
                content={"detail": "内部错误", "trace_id": tid},
            )


# ============================================================================
# 3. 敏感操作二次验证（装饰器）
# ============================================================================

def require_reverify(password_field: str = "password"):
    """
    装饰器：在接口执行前强制验证当前登录用户的密码。
    用法：@require_reverify("current_password") 要求请求体里有 {"current_password": "xxx"}

    安全原则：
    - 改密码、强制下线、删管理员/超级管理员、改系统配置 这些必须加
    - 装饰器内部自己读 request.body() + parse JSON
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            request: Request | None = kwargs.get("request")
            if request is None:
                # FastAPI 也可能把 request 当参数注入
                for a in args:
                    if isinstance(a, Request):
                        request = a; break

            actor = getattr(request.state, "actor", None) if request else None
            if not actor:
                raise HTTPException(401, "未登录")

            # 读 body + 找 password 字段
            try:
                body = await request.json()
            except Exception:
                raise HTTPException(400, "请求体格式错误")
            pwd = body.get(password_field)
            if not pwd:
                raise HTTPException(400, "需要当前密码验证")

            from core.auth import verify_password
            raw = db.get_user_by_username(actor["username"])
            if not raw or not verify_password(pwd, raw.get("password", "")):
                # 二次验证失败也算登录失败（加锁）
                db.record_login_failure(actor["username"], ip=_client_ip(request))
                db.write_audit(
                    action="auth.reverify_failed",
                    actor_user_id=actor["id"], actor_username=actor["username"],
                    ip=_client_ip(request),
                    result="fail", detail="二次验证密码错误",
                )
                raise HTTPException(400, "当前密码错误")

            return await func(*args, **kwargs)
        return wrapper
    return decorator
