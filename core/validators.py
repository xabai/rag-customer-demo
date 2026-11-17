"""
输入校验 —— 统一 Pydantic 模型 + 正则白名单

设计：
- 每个 API endpoint 有独立的 Request Model
- 所有字符串字段自动 strip + 长度/字符集校验
- 注入 FastAPI ValidationError → 全局异常处理
"""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ============ 通用校验器 ============

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_.-]+$")
EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
PATH_TRAVERSAL_RE = re.compile(r"\.\./|\\x00|%00", re.IGNORECASE)


def _safe_str(v: Any) -> str:
    if not isinstance(v, str):
        raise ValueError("必须是字符串")
    v = v.strip()
    if PATH_TRAVERSAL_RE.search(v):
        raise ValueError("包含非法字符（路径穿越/null 字节）")
    return v


# ============ Auth ============

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64, description="用户名")
    password: str = Field(..., min_length=1, max_length=255, description="密码")

    @field_validator("username")
    @classmethod
    def _check_username(cls, v: str) -> str:
        v = _safe_str(v)
        if not USERNAME_RE.match(v):
            raise ValueError("用户名只能包含字母、数字、下划线、点、连字符")
        return v


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=1, max_length=255)
    email: str | None = Field(None, max_length=255)

    @field_validator("username")
    @classmethod
    def _u(cls, v):
        v = _safe_str(v)
        if not USERNAME_RE.match(v):
            raise ValueError("用户名只能包含字母、数字、下划线、点、连字符")
        return v

    @field_validator("email")
    @classmethod
    def _e(cls, v):
        if v:
            v = _safe_str(v)
            if not EMAIL_RE.match(v):
                raise ValueError("邮箱格式不正确")
        return v


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=1, max_length=255)
    new_password: str = Field(..., min_length=1, max_length=255)


class ReverifyRequest(BaseModel):
    """所有二次验证都用这个模型：body 必须带 password"""
    password: str = Field(..., min_length=1, max_length=255, description="当前密码")


# ============ User Management ============

class CreateUserRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    password: str = Field(..., min_length=1, max_length=255)
    role: str = Field(default="user", pattern="^(user|agent|admin|superadmin)$")
    email: str | None = None
    display_name: str | None = None


class UpdateUserRequest(BaseModel):
    role: str | None = Field(None, pattern="^(user|agent|admin|superadmin)$")
    email: str | None = None
    display_name: str | None = None
    enabled: bool | None = None


class ToggleUserRequest(ReverifyRequest):
    enabled: bool = Field(...)


# ============ Chat ============

class ChatRequest(BaseModel):
    session_id: int | None = None
    message: str = Field(..., min_length=1, max_length=8000)
    agent_id: int | None = None
    use_kb: bool = True

    @field_validator("message")
    @classmethod
    def _m(cls, v):
        v = _safe_str(v)
        if len(v) > 8000:
            raise ValueError("消息过长（最多 8000 字符）")
        return v


# ============ Knowledge Base ============

class KBCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str | None = Field(None, max_length=512)


class DocUploadRequest(BaseModel):
    kb_id: int = Field(...)
    url: str | None = Field(None, max_length=1024)


# ============ Agent ============

class AgentCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str | None = Field(None, max_length=512)
    system_prompt: str | None = Field(None, max_length=2000)
    model: str = Field(default="qwen2.5:7b", max_length=64)
    enabled: bool = True


# ============ FAQ / Intent ============

class FaqRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    answer: str = Field(..., min_length=1, max_length=5000)


class IntentRuleRequest(BaseModel):
    intent: str = Field(..., min_length=1, max_length=64)
    pattern: str = Field(..., min_length=1, max_length=1000)
    response: str | None = Field(None, max_length=2000)
    priority: int = Field(default=0, ge=-100, le=100)
    enabled: bool = True


# ============ Settings ============

class SettingsUpdateRequest(BaseModel):
    rag_default_model: str | None = Field(None, max_length=64)
    rag_top_k: int | None = Field(None, ge=1, le=50)
    rag_similarity_threshold: float | None = Field(None, ge=0.0, le=1.0)
    rag_max_chunk_size: int | None = Field(None, ge=64, le=4096)
    ollama_host: str | None = Field(None, max_length=255)


# ============ 通用 query 参数校验 ============

class AuditQuery(BaseModel):
    limit: int = Field(default=200, ge=1, le=2000)
    action: str | None = Field(None, max_length=64)
    actor: str | None = Field(None, max_length=64)
    keyword: str | None = Field(None, max_length=128)
    since: float | None = None
    until: float | None = None


class LogQuery(BaseModel):
    limit: int = Field(default=200, ge=1, le=2000)
    level: str | None = Field(None, pattern="^(ERROR|WARNING|INFO|DEBUG)$")
    keyword: str | None = Field(None, max_length=128)


# ============ 工具：安全 URL ============

URL_BLACKLIST_SCHEMES = {"file", "ftp", "gopher", "data", "javascript", "vbscript"}

def is_safe_url(url: str) -> bool:
    """防止 SSRF"""
    if not url or not url.startswith(("http://", "https://")):
        return False
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        if parsed.scheme in URL_BLACKLIST_SCHEMES:
            return False
        # 禁止内网 IP（简单检查）
        host = (parsed.hostname or "").lower()
        if host.startswith(("127.", "0.", "10.", "172.16.", "192.168.", "169.254.", "localhost", "::1", "fe80:")):
            return False
        return True
    except Exception:
        return False
