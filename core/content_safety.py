"""
内容安全 —— PII 检测 + 敏感词过滤 + 输出 Guardrails

当前版本：轻量本地实现（关键词 + 正则）。
生产可替换为商业 API（阿里云/腾讯云安全内容识别）。
"""
from __future__ import annotations

import re
import logging

from .config import get_settings

_settings = get_settings()
_log = logging.getLogger("rag.content_safety")


# ---- 默认敏感词库（可配置在 .env CONTENT_BLOCKED_WORDS）----
_DEFAULT_BLOCKED: list[str] = [
    # 违规行为类
    "自杀", "自残", "赌博", "贩毒", "毒品", "卖淫", "嫖娼",
    # 政治敏感（极简列表）
    # 注：合规场景必须用专业 API，不能只靠关键词
]


# ---- PII 检测 ----
_PII_PATTERNS = [
    ("phone", re.compile(r"1[3-9]\d{9}")),
    ("id_card", re.compile(r"\d{17}[\dXx]|\d{15}")),
    ("bank_card", re.compile(r"\b\d{16,19}\b")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
]


class ContentSafety:
    """内容安全检查器"""

    def __init__(self):
        self.enabled = _settings.CONTENT_SAFETY_ENABLED
        extra = [w.strip() for w in _settings.CONTENT_BLOCKED_WORDS.split(",") if w.strip()]
        self.blocked = set(_DEFAULT_BLOCKED + extra)

    def check_input(self, text: str) -> dict:
        """检查用户输入是否安全"""
        if not self.enabled:
            return {"safe": True, "blocked": [], "pii": []}

        blocked = [w for w in self.blocked if w in text]
        pii_found = []
        for name, pat in _PII_PATTERNS:
            matches = pat.findall(text)
            if matches:
                pii_found.append(name)

        return {
            "safe": not blocked,
            "blocked": blocked,
            "pii": pii_found,
        }

    def sanitize_output(self, text: str) -> str:
        """脱敏 LLM 输出中的 PII"""
        if not text:
            return text
        for name, pat in _PII_PATTERNS:
            text = pat.sub(f"[{name.upper()}_REDACTED]", text)
        return text


safety = ContentSafety()


class ContentSafetyMiddleware:
    """FastAPI 中间件 —— 在 /api/chat 前检查输入

    注意：使用 BaseHTTPMiddleware 才能被 add_middleware() 接受
    """
    async def __call__(self, request, call_next):
        import json
        from fastapi import HTTPException
        if request.url.path == "/api/chat" and request.method == "POST":
            try:
                body = await request.body()
                if body:
                    data = json.loads(body)
                    msg = data.get("message", "") if isinstance(data, dict) else ""
                    result = safety.check_input(msg)
                    if not result["safe"]:
                        raise HTTPException(400, f"输入包含被禁止的内容: {result['blocked']}")
            except HTTPException:
                raise
            except Exception:
                pass  # 解析失败不阻断
        return await call_next(request)


from starlette.middleware.base import BaseHTTPMiddleware


class ContentSafetyHTTPMiddleware(BaseHTTPMiddleware):
    """真正用在 add_middleware 里的类"""

    async def dispatch(self, request, call_next):
        import json
        if request.url.path == "/api/chat" and request.method == "POST":
            try:
                body = await request.body()
                if body:
                    data = json.loads(body)
                    msg = data.get("message", "") if isinstance(data, dict) else ""
                    result = safety.check_input(msg)
                    if not result["safe"]:
                        from fastapi import HTTPException
                        raise HTTPException(400, f"输入包含被禁止的内容: {result['blocked']}")
            except HTTPException:
                raise
            except Exception:
                pass
        return await call_next(request)
