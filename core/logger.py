# -*- coding: utf-8 -*-
"""统一日志框架：标准 logging + 文件轮转 + 控制台分级 + 自动 trace_id

设计原则：
1. 全局统一入口：get_logger(__name__) 即可
2. 双输出：DEBUG+ 落文件（轮转 10MB×5），INFO+ 打印控制台
3. 自动 trace_id：Formatter 里从 contextvar 注入，不 monkey-patch LogRecordFactory（兼容性坑）
4. 兼容 print：redirect_stdout 把 print 也吞到 logger.info
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from contextvars import ContextVar
from uuid import uuid4

# ======== 结构化 trace_id（请求级唯一标识）=======
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")


def set_trace_id(tid: str):
    trace_id_var.set(tid)


def get_trace_id() -> str:
    return trace_id_var.get()


# ======== 自定义 Formatter：自动注入 trace_id ========
class _TraceFormatter(logging.Formatter):
    def format(self, record):
        try:
            record.trace_id = get_trace_id()
        except Exception:
            record.trace_id = "-"
        return super().format(record)


# ======== 初始化（只跑一次）=======
_initialized = False
_init_lock = threading.Lock()
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "logs")
LOG_FILE = os.path.join(LOG_DIR, "rag.log")


def setup_logger():
    """应用启动时调一次，后续所有 get_logger() 拿到的都是已配置的 logger"""
    global _initialized
    with _init_lock:
        if _initialized:
            return
        _initialized = True

        os.makedirs(LOG_DIR, exist_ok=True)

        # 根 logger：设为 DEBUG，每个 handler 自己控级别
        root = logging.getLogger("rag")
        root.setLevel(logging.DEBUG)
        root.propagate = False

        # ---- 1. 轮转文件 handler（DEBUG 全量）----
        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(_TraceFormatter(
            "%(asctime)s [%(levelname)-7s] [%(trace_id)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        root.addHandler(fh)

        # ---- 2. 控制台 handler（INFO 以上）----
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch.setFormatter(_TraceFormatter(
            "%(asctime)s [%(levelname)-7s] %(message)s",
            datefmt="%H:%M:%S"
        ))
        root.addHandler(ch)

        # ---- 3. print → logger.info（兼容历史 print 语句）----
        # 注意：不能 monkey-patch sys.stdout！uvicorn/fastapi 会调 sys.stdout.isatty()/fileno()
        # 来判断是否输出彩色，而 redirector 不实现这些方法会导致 AttributeError 崩溃。
        # 所以我们只做 wrap：保留原始 stdout，额外让 print 内容也进 logger。
        import builtins as _builtins
        _orig_print = _builtins.print
        def _patched_print(*args, **kwargs):
            _orig_print(*args, **kwargs)
            msg = " ".join(str(a) for a in args)
            if msg.strip():
                root.info(msg.strip())
        _builtins.print = _patched_print


def get_logger(name: str) -> logging.Logger:
    """模块级使用：logger = get_logger(__name__)"""
    if not _initialized:
        setup_logger()
    return logging.getLogger(f"rag.{name}")


# ======== 便捷工具 ========
def log_exception(logger: logging.Logger, msg: str):
    """记录带完整堆栈的异常"""
    import traceback
    logger.error("%s\n%s", msg, traceback.format_exc())


def log_performance(logger: logging.Logger, stage: str, elapsed_ms: float, **extra):
    """记录性能指标"""
    extra_str = " ".join(f"{k}={v}" for k, v in extra.items())
    logger.info("[perf] %s took %.1fms  %s", stage, elapsed_ms, extra_str)


def read_logs(limit: int = 200, level: str = "", keyword: str = "") -> list[dict]:
    """超级管理员看板用：读日志文件"""
    if not os.path.exists(LOG_FILE):
        return []
    records = []
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        lines = lines[-limit * 2:]
        for line in reversed(lines):
            line = line.rstrip()
            if not line: continue
            if level and f"[{level.upper()}]" not in line: continue
            if keyword and keyword.lower() not in line.lower(): continue
            records.append({"line": line})
            if len(records) >= limit: break
        records.reverse()
    except Exception:
        pass
    return records
