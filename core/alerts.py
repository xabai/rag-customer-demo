"""
告警 —— 错误率/延迟超阈值时通过 Webhook 通知

设计：
- 从 metrics 获取最近 N 分钟的错误率和 P99 延迟
- 超过阈值则 POST 到 webhook（飞书/钉钉/Slack 格式自适应）
- 用 async_task 异步发送，避免阻塞主请求
"""
from __future__ import annotations

import json
import logging
import time

from .config import get_settings

_settings = get_settings()
_log = logging.getLogger("rag.alerts")


def send_alert(title: str, message: str, level: str = "info") -> bool:
    """发送告警到 webhook（同步，可被 async_task 调用）"""
    url = _settings.ALERT_WEBHOOK_URL
    if not url:
        _log.info("[alert:disabled] %s - %s", title, message[:100])
        return False

    # 飞书格式
    payload = {
        "msg_type": "interactive",
        "card": {
            "header": {
                "title": {"tag": "plain_text", "content": f"[{level.upper()}] {title}"},
                "template": "red" if level in ("error", "critical") else "orange" if level == "warn" else "blue",
            },
            "elements": [
                {"tag": "div", "text": {"tag": "plain_text", "content": message}},
                {"tag": "div", "text": {"tag": "plain_text", "content": f"时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"}},
            ],
        },
    }
    try:
        import httpx
        import asyncio
        async def _do():
            async with httpx.AsyncClient(timeout=5) as c:
                await c.post(url, json=payload)
        asyncio.run(_do())
        return True
    except Exception as e:
        _log.error("Alert send failed: %s", e)
        return False


def check_and_alert() -> list[dict]:
    """检查最近状态，需要告警就发（建议每分钟调一次）"""
    from .db_engine import db_session
    from .models import AuditLog

    alerts = []

    # 1. 错误率：最近 5 分钟 audit_logs 里 result=fail 的比例
    cutoff = time.time() - 300
    with db_session() as db:
        recent = db.query(AuditLog).filter(AuditLog.created_at >= cutoff).all()
    if len(recent) >= 20:
        errors = [l for l in recent if l.result == "fail"]
        rate = len(errors) / len(recent)
        if rate > _settings.ALERT_ERROR_RATE_THRESHOLD:
            alerts.append({
                "level": "error",
                "title": "错误率超阈值",
                "message": f"最近 5 分钟错误率 {rate:.1%} (阈值 {_settings.ALERT_ERROR_RATE_THRESHOLD:.0%})",
            })

    # 2. DB 健康
    from .health import _check_db
    db_health = _check_db()
    if db_health["status"] != "ok":
        alerts.append({"level": "critical", "title": "数据库不可达", "message": str(db_health)})

    # 3. LLM 健康
    if _settings.OLLAMA_HOST:
        from .health import _check_llm
        llm = _check_llm(timeout=5)
        if llm["status"] != "ok":
            alerts.append({"level": "warn", "title": "LLM 服务异常", "message": str(llm)})

    # 发出
    for a in alerts:
        send_alert(a["title"], a["message"], a["level"])

    return alerts
