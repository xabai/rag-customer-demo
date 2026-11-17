"""
OpenTelemetry Tracing —— 可选开关

默认不启用（TRACING_ENABLED=False），避免无 OTLP Collector 时报错。
启用后通过 OTLP gRPC 发送到 Collector（通常是 Jaeger/Zipkin/Tempo/OTLP Gateway）。
"""
from __future__ import annotations

import logging
from contextvars import ContextVar

from .config import get_settings

_settings = get_settings()

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")
_log = logging.getLogger("rag.tracing")


def setup_tracing() -> None:
    """初始化 OpenTelemetry（幂等，多次调用安全）"""
    if not _settings.TRACING_ENABLED:
        _log.info("Tracing disabled (TRACING_ENABLED=False)")
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": _settings.OTLP_SERVICE_NAME})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=_settings.OTLP_ENDPOINT, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)

        _log.info("OpenTelemetry tracing initialized → %s", _settings.OTLP_ENDPOINT)
    except ImportError as e:
        _log.warning("OpenTelemetry packages not installed: %s", e)
    except Exception as e:
        _log.warning("Tracing setup failed: %s (continuing without tracing)", e)


def get_tracer(name: str):
    """获取命名 Tracer"""
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except Exception:
        return None


def generate_trace_id() -> str:
    """生成 16 字符 hex trace_id（与 logger 共享）"""
    import secrets
    tid = secrets.token_hex(8)
    trace_id_var.set(tid)
    return tid


def current_trace_id() -> str:
    return trace_id_var.get()
