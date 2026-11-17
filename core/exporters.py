"""
审计日志导出 —— CSV / JSON 格式

数据量大时建议用 async_task 异步导出，这里提供同步接口供快速查询
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime

from .db_engine import db_session
from .models import AuditLog


AUDIT_COLUMNS = [
    "id", "time", "actor_user_id", "actor_username",
    "action", "resource_type", "resource_id",
    "ip", "user_agent", "result", "detail", "trace_id",
]


def _row_to_dict(row: AuditLog) -> dict:
    return {
        "id": row.id,
        "time": datetime.fromtimestamp(row.created_at).isoformat() if row.created_at else "",
        "actor_user_id": row.actor_user_id,
        "actor_username": row.actor_username,
        "action": row.action,
        "resource_type": row.resource_type,
        "resource_id": row.resource_id,
        "ip": row.ip,
        "user_agent": row.user_agent,
        "result": row.result,
        "detail": row.detail,
        "trace_id": row.trace_id,
    }


def export_audit_logs(limit: int = 5000, fmt: str = "json",
                      action: str | None = None, since: float | None = None) -> dict | bytes:
    """
    返回：
      fmt="json" → dict{filename, content}
      fmt="csv"  → bytes (UTF-8 with BOM，Excel 兼容)
    """
    with db_session() as db:
        q = db.query(AuditLog).order_by(AuditLog.created_at.desc())
        if action:
            q = q.filter(AuditLog.action.like(f"%{action}%"))
        if since:
            q = q.filter(AuditLog.created_at >= since)
        q = q.limit(limit)
        rows = [_row_to_dict(r) for r in q.all()]

    if fmt == "json":
        content = json.dumps(rows, ensure_ascii=False, indent=2)
        return {"filename": f"audit_logs_{int(time.time())}.json", "content": content,
                "count": len(rows)}

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=AUDIT_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
        csv_bytes = buf.getvalue().encode("utf-8-sig")  # BOM for Excel
        return {"filename": f"audit_logs_{int(time.time())}.csv", "content": csv_bytes,
                "count": len(rows)}

    raise ValueError(f"Unknown format: {fmt}")


# 时间戳 hack（避免额外 import）
import time


def export_cost_logs(user_id: int | None = None, days: int = 30,
                     fmt: str = "json") -> dict | bytes:
    """导出成本日志"""
    from .db_engine import db_session
    from .models import CostLog
    from datetime import datetime, timedelta

    cutoff = (datetime.now() - timedelta(days=days)).timestamp()
    with db_session() as db:
        q = db.query(CostLog).filter(CostLog.created_at >= cutoff)
        if user_id:
            q = q.filter(CostLog.user_id == user_id)
        rows = q.order_by(CostLog.created_at.desc()).limit(10000).all()

    data = []
    for r in rows:
        data.append({
            "time": datetime.fromtimestamp(r.created_at).isoformat(),
            "model": r.model, "prompt_tokens": r.prompt_tokens,
            "completion_tokens": r.completion_tokens,
            "total_tokens": r.total_tokens,
            "cost_usd": float(r.cost_usd),
        })

    if fmt == "json":
        return {"filename": f"cost_logs_{int(time.time())}.json",
                "content": json.dumps(data, ensure_ascii=False, indent=2),
                "count": len(data)}
    # csv
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(data[0].keys()) if data else ["time"])
    writer.writeheader()
    for r in data:
        writer.writerow(r)
    return {"filename": f"cost_logs_{int(time.time())}.csv",
            "content": buf.getvalue().encode("utf-8-sig"), "count": len(data)}
