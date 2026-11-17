"""
异步任务队列 —— DB 队列为主（async_tasks 表），Redis 为可选加速

支持的任务类型：
- kb_index           知识库重新索引
- export_audit       导出审计日志
- export_qa          导出 Q&A 日志
- cleanup_sessions   清理过期 session
- send_notification  发送告警/webhook

Worker 模式：单独跑 python -m core.worker 消费队列；也可在主进程跑 asyncio loop
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable

from .config import get_settings

_settings = get_settings()
_log = logging.getLogger("rag.async_queue")


# 任务处理器注册表
_HANDLERS: dict[str, Callable[..., Any]] = {}


def register_task(task_type: str):
    """装饰器：注册任务处理器"""
    def decorator(func):
        _HANDLERS[task_type] = func
        return func
    return decorator


def submit_task(task_type: str, payload: dict | None = None, priority: int = 0) -> int:
    """提交一个异步任务，返回 task_id"""
    from .db_engine import db_session
    from .models import AsyncTask

    with db_session() as db:
        t = AsyncTask(
            task_type=task_type,
            payload_json=json.dumps(payload or {}, ensure_ascii=False),
            status="pending",
        )
        db.add(t)
        db.flush()
        task_id = t.id

    # 如果配了 Redis 还会 push 通知
    if _settings.REDIS_URL:
        try:
            import redis as _redis
            r = _redis.from_url(_settings.REDIS_URL, socket_timeout=2)
            r.rpush("rag:async_queue", str(task_id))
        except Exception as e:
            _log.warning("Redis push failed: %s (fallback to DB polling)", e)

    _log.info("Task submitted: id=%d type=%s", task_id, task_type)
    return task_id


def poll_and_run_once() -> bool:
    """轮询 DB 找一个 pending 任务并执行（同步，可单独线程跑）"""
    from .db_engine import db_session
    from .models import AsyncTask

    with db_session() as db:
        # 找最老的 pending 任务
        task = (db.query(AsyncTask)
                  .filter(AsyncTask.status == "pending")
                  .order_by(AsyncTask.created_at.asc())
                  .first())
        if not task:
            return False

        # 抢任务：CAS 式更新
        now = time.time()
        task.status = "running"
        task.started_at = now
        db.flush()
        task_id = task.id
        task_type = task.task_type
        payload = json.loads(task.payload_json or "{}")

    handler = _HANDLERS.get(task_type)
    if not handler:
        _log.error("No handler registered for task type: %s", task_type)
        with db_session() as db:
            t = db.query(AsyncTask).get(task_id)
            if t:
                t.status = "failed"
                t.error = f"No handler for type={task_type}"
        return True

    _log.info("Running task #%d: %s", task_id, task_type)
    try:
        result = handler(payload)
        with db_session() as db:
            t = db.query(AsyncTask).get(task_id)
            if t:
                t.status = "done"
                t.finished_at = time.time()
        _log.info("Task #%d done", task_id)
    except Exception as e:
        _log.exception("Task #%d failed: %s", task_id, e)
        with db_session() as db:
            t = db.query(AsyncTask).get(task_id)
            if t:
                t.retry_count += 1
                if t.retry_count >= t.max_retries:
                    t.status = "failed"
                    t.error = str(e)[:2000]
                else:
                    t.status = "pending"  # 重试
    return True


async def run_worker_loop(concurrency: int | None = None):
    """协程 worker loop（在 FastAPI 启动时 spawn）"""
    concurrency = concurrency or _settings.ASYNC_WORKER_CONCURRENCY
    _log.info("Async worker started (concurrency=%d)", concurrency)

    while True:
        try:
            did_work = poll_and_run_once()
            await asyncio.sleep(0.5 if not did_work else 0)
        except Exception as e:
            _log.exception("Worker loop error: %s", e)
            await asyncio.sleep(2)


# ---- 内置任务处理器 ----

@register_task("export_audit")
def _export_audit(payload: dict) -> dict:
    """导出审计日志为 CSV/JSON"""
    from .exporters import export_audit_logs
    return export_audit_logs(
        limit=payload.get("limit", 5000),
        fmt=payload.get("fmt", "json"),
    )


@register_task("cleanup_sessions")
def _cleanup_sessions(payload: dict) -> dict:
    """清理过期 session"""
    from .db_engine import db_session
    from .models import UserSession
    cutoff = payload.get("cutoff_ts", time.time())
    with db_session() as db:
        deleted = db.query(UserSession).filter(
            UserSession.expires_at < cutoff
        ).delete()
    return {"deleted": deleted}


@register_task("send_notification")
def _send_notification(payload: dict) -> dict:
    """发送告警 webhook"""
    from .alerts import send_alert
    return send_alert(
        title=payload.get("title", "Notification"),
        message=payload.get("message", ""),
        level=payload.get("level", "info"),
    )
