# -*- coding: utf-8 -*-
"""RBAC 认证模块：bcrypt 密码 + JWT access token + 服务端 session 白名单

角色：user / admin / superadmin
- JWT 签发后写入 user_sessions 表（带 session_id + exp + revoked）
- 强制下线 = 批量把 revoked 置 1，所有 JWT 立即失效
"""
from __future__ import annotations

import os
import sys
import time
import uuid
import threading

# 把项目根目录加入 sys.path，以便 import db（db.py 在项目根目录而非 core 包内）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import bcrypt
import jwt

import db as _db

# --- 配置 ---
JWT_SECRET = os.environ.get("RAG_JWT_SECRET", "rag-bot-dev-secret-change-in-prod-2026")
JWT_ALGO = "HS256"
ACCESS_TOKEN_TTL = 21600        # 15 分钟
SESSION_TOKEN_TTL = 604800    # 7 天（刷新窗口）


# ---------- 密码 ----------
def hash_password(pwd: str) -> str:
    """bcrypt 哈希，永不存明文"""
    return bcrypt.hashpw(pwd.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(pwd: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pwd.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


# ---------- JWT ----------
def issue_access_token(user_id: int, username: str, role: str, session_id: str) -> str:
    now = int(time.time())
    payload = {
        "sub": str(user_id),  # JWT RFC 要求 sub 必须是 string
        "uid": user_id,        # 原始 int 保留方便 cast
        "username": username,
        "role": role,
        "sid": session_id,
        "iat": now,
        "exp": now + ACCESS_TOKEN_TTL,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


# ---------- 登录/注销 ----------
def authenticate(username: str, password: str, ip: str = "", user_agent: str = "") -> dict | None:
    """验证用户名+密码 → 签发 session_id + 写白名单 → 返回 token

    带锁定机制：
    - 账号被锁 → 返回 {"locked": True, "lockout_seconds": N}
    - 密码错误 → 累加 failed_attempts，达阈值自动锁
    - 密码正确 → 清零 failed_attempts
    """
    # 0. 检查锁定（防暴力）
    if _db.check_account_locked(username):
        # 审计：锁定时尝试登录
        try:
            _db.write_audit(
                action="auth.login.blocked_locked",
                actor_username=username,
                ip=ip, user_agent=user_agent,
                result="fail", detail="账号处于锁定状态"
            )
        except Exception: pass
        return {"locked": True, "lockout_seconds": 600}

    # 1. 查用户
    user = _db.get_user_by_username(username)
    if not user or not verify_password(password, user.get("password", "")):
        # 密码错 → 记一次失败（顺带写 auth_failures 表）
        status = _db.record_login_failure(username, ip=ip, user_agent=user_agent)
        detail = f"密码错误 {status['remaining']} 次后锁定" if not status["locked"] else f"已锁定 {status['lockout_seconds']} 秒"
        try:
            _db.write_audit(
                action="auth.login.failed",
                actor_username=username, ip=ip, user_agent=user_agent,
                result="fail", detail=detail
            )
        except Exception: pass
        return None

    if not user.get("enabled", True):
        try:
            _db.write_audit(
                action="auth.login.blocked_disabled",
                actor_user_id=user["id"], actor_username=user["username"],
                ip=ip, user_agent=user_agent,
                result="fail", detail="账号已禁用"
            )
        except Exception: pass
        return {"disabled": True}

    # 2. 清零失败计数（登录成功）
    _db.reset_failed_attempts(username)

    # 3. 写服务端 session 白名单
    session_id = uuid.uuid4().hex[:16]
    _db.create_user_session(user["id"], session_id, ACCESS_TOKEN_TTL)
    # B2 Redis session 同步（横向扩展）
    try:
        from core.redis_store import redis_add_session as _redis_add
        _redis_add(session_id, user["id"], time.time() + ACCESS_TOKEN_TTL,
                   ip=ip, user_agent=user_agent)
    except Exception:
        pass  # Redis 不可用不阻断
    token = issue_access_token(user["id"], user["username"], user["role"], session_id)

    # 4. 审计：登录成功
    try:
        _db.write_audit(
            action="auth.login.success",
            actor_user_id=user["id"], actor_username=user["username"],
            ip=ip, user_agent=user_agent,
            result="success",
        )
    except Exception: pass

    return {"token": token, "username": user["username"], "role": user["role"],
            "user_id": user["id"]}


def logout(token: str) -> None:
    """注销单个 session（前端主动退出）"""
    payload = decode_token(token)
    if payload:
        _db.revoke_user_session(payload.get("sid"))


def force_logout_user(user_id: int) -> int:
    """强制下线某用户所有 session，返回失效数量"""
    return _db.revoke_all_user_sessions(user_id)


def verify_token(token: str) -> dict | None:
    """验证 JWT 签名 + 检查服务端白名单未被撤销"""
    payload = decode_token(token)
    if not payload:
        return None
    sid = payload.get("sid")
    if not sid or not _db.is_session_valid(sid):
        return None
    return payload


# ---------- 角色权限 ----------
ROLE_HIERARCHY = {"user": 0, "admin": 1, "superadmin": 2}


def role_can(actual: str, required: str) -> bool:
    return ROLE_HIERARCHY.get(actual, -1) >= ROLE_HIERARCHY.get(required, -1)
