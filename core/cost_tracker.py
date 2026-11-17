"""
成本追踪 —— 每次 LLM 调用落库 cost_logs

Ollama 免费，token → cost_usd 系数设为 0。
将来换商业模型（OpenAI/Anthropic）只需更新 PRICING 字典。
"""
from __future__ import annotations

from datetime import datetime, timedelta

from .db_engine import db_session
from .models import CostLog


# ---------- 模型定价表（每 1K token，USD）----------
PRICING: dict[str, dict[str, float]] = {
    # 格式: "model_name_contains": {"prompt": X, "completion": Y}
    "qwen":       {"prompt": 0.0, "completion": 0.0},       # 本地 Ollama 免费
    "llama":      {"prompt": 0.0, "completion": 0.0},
    "mistral":    {"prompt": 0.0, "completion": 0.0},
    "deepseek":   {"prompt": 0.0, "completion": 0.0},
    "gpt-4o":     {"prompt": 0.005, "completion": 0.015},
    "gpt-4":      {"prompt": 0.03, "completion": 0.06},
    "gpt-3.5":    {"prompt": 0.0005, "completion": 0.0015},
    "claude-3-opus": {"prompt": 0.015, "completion": 0.075},
    "claude-3-sonnet": {"prompt": 0.003, "completion": 0.015},
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """估算调用成本（USD）"""
    pricing = None
    for key, p in PRICING.items():
        if key in model.lower():
            pricing = p
            break
    if pricing is None:
        pricing = {"prompt": 0.0, "completion": 0.0}  # 未知模型不计费
    return (prompt_tokens / 1000) * pricing["prompt"] + \
           (completion_tokens / 1000) * pricing["completion"]


def log_cost(user_id: int | None, session_id: int | None, model: str,
             prompt_tokens: int, completion_tokens: int) -> float:
    """写一条 cost_log，返回估算成本"""
    cost = estimate_cost(model, prompt_tokens, completion_tokens)
    with db_session() as db:
        db.add(CostLog(
            user_id=user_id,
            session_id=session_id,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            cost_usd=cost,
        ))
    return cost


def get_user_cost(user_id: int, days: int = 7) -> dict:
    """查询用户的成本统计"""
    cutoff = (datetime.now() - timedelta(days=days)).timestamp()
    with db_session() as db:
        from sqlalchemy import func
        total = (db.query(func.sum(CostLog.total_tokens),
                          func.sum(CostLog.cost_usd))
                   .filter(CostLog.user_id == user_id,
                           CostLog.created_at >= cutoff)
                   .one())
        by_model = (db.query(CostLog.model,
                             func.sum(CostLog.total_tokens),
                             func.sum(CostLog.cost_usd))
                      .filter(CostLog.user_id == user_id,
                              CostLog.created_at >= cutoff)
                      .group_by(CostLog.model)
                      .all())
    return {
        "days": days,
        "total_tokens": int(total[0] or 0),
        "total_cost_usd": float(total[1] or 0.0),
        "by_model": [{"model": m, "tokens": int(t or 0), "cost_usd": float(c or 0)}
                     for m, t, c in by_model],
    }


def get_global_cost(days: int = 30) -> dict:
    """全局成本看板"""
    cutoff = (datetime.now() - timedelta(days=days)).timestamp()
    with db_session() as db:
        from sqlalchemy import func
        total = (db.query(func.sum(CostLog.total_tokens),
                          func.sum(CostLog.cost_usd),
                          func.count(CostLog.id))
                   .filter(CostLog.created_at >= cutoff)
                   .one())
    return {
        "days": days,
        "total_tokens": int(total[0] or 0),
        "total_cost_usd": float(total[1] or 0.0),
        "total_calls": int(total[2] or 0),
    }
