"""
Prometheus Metrics Middleware —— 暴露 /metrics 端点

暴露指标：
- http_requests_total{method,path,status}  Counter
- http_request_duration_seconds{method,path}  Histogram（分位值）
- http_requests_in_progress  Gauge
- http_errors_total{method,path}  Counter
- llm_calls_total{model,status}  Counter
- llm_tokens_total{model}  Counter
- rag_kb_documents_total  Gauge
- db_pool_active / db_pool_idle  Gauge
"""
from __future__ import annotations

import time
from typing import Callable

from fastapi import Request, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# ---------- Registry ----------
REGISTRY = CollectorRegistry()

# ---------- HTTP Metrics ----------
HTTP_REQUESTS = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
    registry=REGISTRY,
)
HTTP_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)
HTTP_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Currently in-flight HTTP requests",
    registry=REGISTRY,
)
HTTP_ERRORS = Counter(
    "http_errors_total",
    "Total 5xx errors",
    ["method", "path"],
    registry=REGISTRY,
)

# ---------- LLM / RAG Metrics ----------
LLM_CALLS = Counter(
    "llm_calls_total",
    "Total LLM calls",
    ["model", "status"],
    registry=REGISTRY,
)
LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Total tokens consumed",
    ["model", "type"],  # type: prompt / completion / total
    registry=REGISTRY,
)
RAG_RETRIEVAL = Histogram(
    "rag_retrieval_duration_seconds",
    "RAG retrieval latency",
    ["strategy"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0),
    registry=REGISTRY,
)

# ---------- System ----------
KB_DOCS = Gauge(
    "rag_kb_documents_total",
    "Total documents in all KBs",
    registry=REGISTRY,
)
HEARTBEAT = Gauge(
    "rag_service_heartbeat",
    "Service heartbeat (1=live)",
    registry=REGISTRY,
)


# ---------- Middleware ----------
from starlette.middleware.base import BaseHTTPMiddleware


class MetricsMiddleware(BaseHTTPMiddleware):
    """全局 metrics 中间件"""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # 跳过 /metrics 本身
        if request.url.path == "/metrics":
            return await call_next(request)

        method = request.method
        path = request.url.path
        import re
        norm = re.sub(r"/\d+", "/{id}", path)

        HTTP_IN_PROGRESS.inc()
        start = time.time()
        try:
            response = await call_next(request)
            duration = time.time() - start
            status = response.status_code
        except Exception:
            duration = time.time() - start
            HTTP_DURATION.labels(method=method, path=norm).observe(duration)
            HTTP_IN_PROGRESS.dec()
            raise

        HTTP_DURATION.labels(method=method, path=norm).observe(duration)
        HTTP_REQUESTS.labels(method=method, path=norm, status=str(status)).inc()
        HTTP_IN_PROGRESS.dec()
        if status >= 500:
            HTTP_ERRORS.labels(method=method, path=norm).inc()

        return response


async def metrics_middleware(request: Request, call_next: Callable) -> Response:
    """函数形式保留（兼容），实际靠 MetricsMiddleware 类使用"""
    return await MetricsMiddleware.__wrapped__(MetricsMiddleware, request, call_next)  # type: ignore


# ---------- Metrics Endpoint ----------
async def metrics_endpoint() -> Response:
    """GET /metrics —— Prometheus 抓取端点"""
    return PlainTextResponse(
        generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )


# ---------- 辅助：记录 LLM 调用 ----------
def record_llm_call(model: str, status: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
    LLM_CALLS.labels(model=model, status=status).inc()
    LLM_TOKENS.labels(model=model, type="prompt").inc(prompt_tokens)
    LLM_TOKENS.labels(model=model, type="completion").inc(completion_tokens)
    LLM_TOKENS.labels(model=model, type="total").inc(prompt_tokens + completion_tokens)


def record_rag_retrieval(strategy: str, duration: float) -> None:
    RAG_RETRIEVAL.labels(strategy=strategy).observe(duration)


def set_kb_docs(count: int) -> None:
    KB_DOCS.set(count)


def heartbeat() -> None:
    HEARTBEAT.set(1)
