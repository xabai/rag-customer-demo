"""
多模型路由 —— 主模型失败自动降级到 Fallback

策略：
1. 优先用 Agent 指定的 model
2. 失败（超时/HTTP 5xx/连接失败）自动切 fallback_model
3. 所有调用记录到 cost_logs + metrics
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from .config import get_settings
from .metrics import LLM_CALLS, LLM_TOKENS, record_llm_call

_settings = get_settings()
_log = logging.getLogger("rag.multimodel")


class ModelRouter:
    """模型路由器"""

    def __init__(self):
        self.default = _settings.OLLAMA_DEFAULT_MODEL
        self.fallback = _settings.OLLAMA_FALLBACK_MODEL
        self.host = _settings.OLLAMA_HOST
        self.timeout = _settings.LLM_TIMEOUT

    def build_url(self, model: str | None = None, fallback: str | None = None) -> list[dict]:
        """返回候选模型列表（按优先级）"""
        candidates = []
        primary = model or self.default
        candidates.append({"model": primary, "is_primary": True})
        fb = fallback or self.fallback
        if fb and fb != primary:
            candidates.append({"model": fb, "is_primary": False})
        return candidates


router = ModelRouter()


async def chat_with_router(
    messages: list[dict],
    model: str | None = None,
    fallback: str | None = None,
    stream: bool = False,
    **kwargs,
) -> dict | AsyncIterator:
    """
    带自动降级的 LLM 调用。
    返回: {"content": ..., "model": ..., "tokens": {...}, "latency_ms": ...}
    """
    import time
    import httpx
    from tenacity import retry, stop_after_attempt, retry_if_exception_type

    candidates = router.build_url(model, fallback)
    last_err = None
    start = time.time()

    for cand in candidates:
        m = cand["model"]
        try:
            url = f"{router.host.rstrip('/')}/api/chat"
            payload = {
                "model": m,
                "messages": messages,
                "stream": stream,
                "options": {"temperature": kwargs.get("temperature", 0.7)},
            }
            timeout = httpx.Timeout(router.timeout)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code >= 400:
                    raise Exception(f"HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()

            latency = int((time.time() - start) * 1000)
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            record_llm_call(m, "ok", prompt_tokens, completion_tokens)

            return {
                "content": data.get("message", {}).get("content", ""),
                "model": m,
                "tokens": {"prompt": prompt_tokens, "completion": completion_tokens,
                           "total": prompt_tokens + completion_tokens},
                "latency_ms": latency,
                "from_fallback": not cand["is_primary"],
            }
        except Exception as e:
            last_err = e
            _log.warning("Model %s failed: %s. %s",
                         m, e, "trying fallback..." if not cand["is_primary"] else "no more fallback")
            record_llm_call(m, "error", 0, 0)
            continue

    # 所有候选都失败
    record_llm_call((model or router.default), "all_failed", 0, 0)
    raise RuntimeError(f"All models failed. Last error: {last_err}")
