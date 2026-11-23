"""
db_pg.py — PostgreSQL ORM 版核心函数
被 db.py 条件覆盖（当 DB_DRIVER=postgresql 时）
签名 100% 兼容 SQLite 版本，app.py 一行不动
"""
from __future__ import annotations
import sys, json, time
from typing import Optional

from core.db_engine import SessionLocal
from sqlalchemy import select, func, update, delete
from core import models as _m


# ── 辅助 ──
def _pg(): return SessionLocal()

def _to_dict(row) -> dict:
    if row is None: return {}
    if hasattr(row, "_mapping"): return dict(row._mapping)
    d = {}
    for c in row.__table__.columns:
        d[c.name] = getattr(row, c.name)
    return d

def _pj(v):
    if v is None or v == "": return None
    if isinstance(v, (list, dict)): return v
    try: return json.loads(v)
    except: return None


# ── Users ──

def get_user_by_username(username: str) -> Optional[dict]:
    with _pg() as s:
        row = s.execute(select(_m.User).where(_m.User.username == username)).scalar_one_or_none()
        if row is None: return None
        d = _to_dict(row)
        d["password"] = d.pop("password_hash", "")
        return d

def get_user_by_id(user_id: int) -> Optional[dict]:
    with _pg() as s:
        row = s.execute(select(_m.User).where(_m.User.id == user_id)).scalar_one_or_none()
        if row is None: return None
        d = _to_dict(row)
        d["password"] = d.pop("password_hash", "")
        return d

def list_users() -> list[dict]:
    with _pg() as s:
        rows = s.execute(select(_m.User).order_by(_m.User.id)).scalars().all()
        result = []
        for r in rows:
            d = _to_dict(r)
            d["password"] = d.pop("password_hash", "")
            result.append(d)
        return result

def create_user(username: str, password: str, role: str = "user") -> Optional[int]:
    from core.auth import hash_password
    with _pg() as s:
        u = _m.User(username=username, password_hash=hash_password(password),
                    role=role, enabled=True,
                    created_at=time.time(), updated_at=time.time(), tenant_id=1)
        s.add(u); s.commit(); s.refresh(u)
        return u.id

def update_user(user_id: int, **fields) -> bool:
    with _pg() as s:
        u = s.get(_m.User, user_id)
        if not u: return False
        for k, v in fields.items():
            # password 在 SQLite 是明文，PG 是 password_hash
            if k == "password":
                from core.auth import hash_password
                u.password_hash = hash_password(v)
            elif hasattr(u, k):
                setattr(u, k, v)
        u.updated_at = time.time(); s.commit(); return True

def delete_user(user_id: int) -> bool:
    with _pg() as s:
        u = s.get(_m.User, user_id)
        if not u: return False
        s.delete(u); s.commit(); return True

def toggle_user(user_id: int, enabled: bool) -> bool:
    return update_user(user_id, enabled=enabled)

def change_password(user_id: int, new_password: str) -> bool:
    return update_user(user_id, password=new_password)

def list_roles() -> list[dict]:
    return [
        {"name": "superadmin", "display_name": "超级管理员", "description": "全部权限", "level": 100, "builtin": True, "enabled": True},
        {"name": "admin",      "display_name": "管理员",     "description": "知识库/用户管理", "level": 80, "builtin": True, "enabled": True},
        {"name": "agent",      "display_name": "Agent",     "description": "内部服务", "level": 50, "builtin": True, "enabled": True},
        {"name": "user",       "display_name": "普通用户",  "description": "对话+历史", "level": 10, "builtin": True, "enabled": True},
    ]


# ── Sessions ──

def create_session(title: str = "新对话", user_id=None) -> str:
    with _pg() as s:
        ses = _m.ChatSession(title=title, user_id=user_id,
                             created_at=time.time(), updated_at=time.time(), tenant_id=1)
        s.add(ses); s.commit(); s.refresh(ses)
        return str(ses.id)

def list_sessions(user_id=None, page=1, page_size=None) -> list[dict]:
    with _pg() as s:
        q = select(_m.ChatSession).order_by(_m.ChatSession.created_at.desc())
        if user_id: q = q.where(_m.ChatSession.user_id == user_id)
        if page_size: q = q.limit(page_size).offset((page-1)*page_size)
        rows = s.execute(q).scalars().all()
        return [{"id": str(r.id), "title": r.title, "user_id": r.user_id,
                 "created_at": r.created_at} for r in rows]

def rename_session(session_id, title: str):
    with _pg() as s:
        ses = s.get(_m.ChatSession, int(session_id))
        if ses: ses.title = title; ses.updated_at = time.time(); s.commit()

def delete_session(session_id):
    with _pg() as s:
        ses = s.get(_m.ChatSession, int(session_id))
        if ses: s.delete(ses); s.commit()


# ── Messages ──

def add_message(session_id, role, content, images=None, sources=None):
    with _pg() as s:
        sid = int(session_id) if str(session_id).isdigit() else None
        if sid is None: return None
        msg = _m.ChatMessage(session_id=sid, role=role, content=content,
                             images=images if images else None,
                             sources=sources if sources else None,
                             created_at=time.time(), updated_at=time.time(), tenant_id=1)
        s.add(msg); s.commit(); s.refresh(msg)
        return msg.id

def get_messages(session_id, limit=None) -> list[dict]:
    with _pg() as s:
        sid = int(session_id) if str(session_id).isdigit() else None
        if sid is None: return []
        q = select(_m.ChatMessage).where(_m.ChatMessage.session_id == sid).order_by(_m.ChatMessage.created_at)
        if limit: q = q.limit(limit)
        rows = s.execute(q).scalars().all()
        result = []
        for r in rows:
            result.append({
                "id": r.id, "session_id": str(session_id),
                "role": r.role, "content": r.content,
                "created_at": r.created_at,
                "images": _pj(r.images), "sources": _pj(r.sources),
            })
        return result


# ── KB ──

def create_kb(name, description="", chunk_config=None):
    with _pg() as s:
        kb = _m.KnowledgeBase(name=name, description=description,
                             collection_name=f"kb_{name}",
                             chunk_config=chunk_config,
                             created_at=time.time(), updated_at=time.time(), tenant_id=1)
        s.add(kb); s.commit(); s.refresh(kb)
        return {"id": kb.id, "name": name, "description": description}

def get_kb(kid):
    with _pg() as s:
        row = s.execute(select(_m.KnowledgeBase).where(_m.KnowledgeBase.id == kid)).scalar_one_or_none()
        if row is None: return None
        d = _to_dict(row)
        d["chunk_config"] = _pj(d.get("chunk_config"))
        return d

def list_kb(page=1, page_size=None) -> list[dict]:
    with _pg() as s:
        q = select(_m.KnowledgeBase).order_by(_m.KnowledgeBase.created_at.desc())
        if page_size: q = q.limit(page_size).offset((page-1)*page_size)
        rows = s.execute(q).scalars().all()
        result = []
        for r in rows:
            d = _to_dict(r)
            d["chunk_config"] = _pj(d.get("chunk_config"))
            result.append(d)
        return result

def delete_kb(kid):
    with _pg() as s:
        kb = s.get(_m.KnowledgeBase, kid)
        if kb: s.delete(kb); s.commit()

def update_kb(kid, name=None, description=None, chunk_config=None):
    with _pg() as s:
        kb = s.get(_m.KnowledgeBase, kid)
        if not kb: return False
        if name is not None: kb.name = name
        if description is not None: kb.description = description
        if chunk_config is not None: kb.chunk_config = chunk_config
        kb.updated_at = time.time(); s.commit(); return True


# ── Audit ──

def write_audit(action, resource_type="", resource_id="", result="success", detail="",
                actor_user_id=None, actor_username=None, ip="", user_agent=""):
    with _pg() as s:
        rid = int(resource_id) if resource_id and str(resource_id).isdigit() else None
        log = _m.AuditLog(action=action, resource_type=resource_type, resource_id=rid,
                          result=result, detail=detail,
                          actor_user_id=actor_user_id, actor_username=actor_username,
                          ip=ip, user_agent=user_agent,
                          created_at=time.time(), updated_at=time.time(), tenant_id=1)
        s.add(log); s.commit(); return log.id

def list_audit_logs(
    limit: int = 200, action: str = "", actor: str = "", keyword: str = "",
    page: int = 1, page_size: int | None = None,
):
    """完全兼容 db.py 的签名 + 两种返回值模式"""
    from sqlalchemy import func, and_, or_
    with _pg() as s:
        # 构建查询
        q = select(_m.AuditLog)
        if action:
            q = q.where(_m.AuditLog.action.like(f"%{action}%"))
        if actor:
            q = q.where(or_(
                _m.AuditLog.actor_username.like(f"%{actor}%"),
                _m.AuditLog.actor_user_id == int(actor) if actor.isdigit() else _m.AuditLog.actor_user_id == -1,
            ))
        if keyword:
            q = q.where(or_(
                _m.AuditLog.detail.like(f"%{keyword}%"),
                _m.AuditLog.action.like(f"%{keyword}%"),
                _m.AuditLog.resource_type.like(f"%{keyword}%"),
            ))
        q = q.order_by(_m.AuditLog.created_at.desc())

        if page_size is None:
            rows = s.execute(q.limit(limit)).scalars().all()
            return _audit_rows_to_dicts(rows)

        # 分页模式
        count_q = select(func.count()).select_from(q.subquery())
        total = s.execute(count_q).scalar() or 0
        ps = max(1, page_size)
        p = max(1, page)
        rows = s.execute(q.limit(ps).offset((p - 1) * ps)).scalars().all()

    import math as _m2
    return {
        "items": _audit_rows_to_dicts(rows),
        "total": total,
        "page": p,
        "page_size": ps,
        "pages": max(1, _m2.ceil(total / ps)),
    }

def _audit_rows_to_dicts(rows):
    return [
        {
            "id": r.id,
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.created_at)),
            "actor_user_id": r.actor_user_id,
            "actor_username": r.actor_username or "(未登录)",
            "action": r.action,
            "resource_type": r.resource_type or "",
            "resource_id": r.resource_id or "",
            "ip": r.ip or "",
            "user_agent": (r.user_agent or "")[:60],
            "result": r.result,
            "detail": r.detail or "",
        }
        for r in rows
    ]


# ── Session auth ──

def create_user_session(user_id: int, session_id: str, ttl_seconds: int):
    with _pg() as s:
        es = _m.UserSession(user_id=user_id, session_id=session_id,
                            expires_at=time.time()+ttl_seconds, created_at=time.time())
        s.add(es); s.commit()

def revoke_user_session(session_id: str):
    with _pg() as s:
        es = s.execute(select(_m.UserSession).where(_m.UserSession.session_id == session_id)).scalar_one_or_none()
        if es: es.revoked = True; s.commit()

def revoke_all_user_sessions(user_id: int) -> int:
    with _pg() as s:
        rows = s.execute(select(_m.UserSession).where(_m.UserSession.user_id == user_id)).scalars().all()
        for r in rows: r.revoked = True
        s.commit(); return len(rows)

def is_session_valid(session_id: str) -> bool:
    with _pg() as s:
        es = s.execute(select(_m.UserSession).where(_m.UserSession.session_id == session_id)).scalar_one_or_none()
        if not es or es.revoked: return False
        return (es.expires_at or 0) > time.time()

def check_account_locked(username: str) -> bool:
    u = get_user_by_username(username)
    return bool(u and u.get("lockout_until") and u["lockout_until"] > time.time())

def record_login_failure(username: str, ip: str = "", user_agent: str = "") -> dict:
    """兼容 db.py：返回 {"locked", "remaining", "lockout_seconds"}"""
    MAX_ATTEMPTS = 5
    LOCKOUT_MINUTES = 10

    with _pg() as s:
        fl = _m.LoginFailure(username=username, ip=ip, user_agent=user_agent,
                              created_at=time.time(), updated_at=time.time())
        s.add(fl)

    u = get_user_by_username(username)
    if not u:
        return {"locked": False, "remaining": MAX_ATTEMPTS, "lockout_seconds": 0}

    new_count = (u.get("failed_attempts") or 0) + 1
    if new_count >= MAX_ATTEMPTS:
        lockout_until = time.time() + LOCKOUT_MINUTES * 60
        update_user(u["id"], failed_attempts=new_count, lockout_until=lockout_until)
        return {"locked": True, "remaining": 0, "lockout_seconds": LOCKOUT_MINUTES * 60}
    else:
        update_user(u["id"], failed_attempts=new_count)
        return {"locked": False, "remaining": MAX_ATTEMPTS - new_count, "lockout_seconds": 0}

def reset_failed_attempts(username: str) -> None:
    u = get_user_by_username(username)
    if u: update_user(u["id"], failed_attempts=0, lockout_until=None)

def list_auth_failures(
    username: str = "", ip: str = "", limit: int = 100,
    page: int = 1, page_size: int | None = None,
):
    """兼容 db.py：支持 page_size=None 返回 list，否则返回分页 dict"""
    from sqlalchemy import func, or_
    with _pg() as s:
        q = select(_m.LoginFailure)
        if username: q = q.where(_m.LoginFailure.username == username)
        if ip: q = q.where(_m.LoginFailure.ip == ip)
        q = q.order_by(_m.LoginFailure.created_at.desc())

        if page_size is None:
            rows = s.execute(q.limit(limit)).scalars().all()
            return _failure_rows_to_dicts(rows)

        total = s.execute(select(func.count()).select_from(q.subquery())).scalar() or 0
        ps = max(1, page_size)
        p = max(1, page)
        rows = s.execute(q.limit(ps).offset((p - 1) * ps)).scalars().all()

    import math as _m2
    return {
        "items": _failure_rows_to_dicts(rows),
        "total": total,
        "page": p, "page_size": ps,
        "pages": max(1, _m2.ceil(total / ps)),
    }

def _failure_rows_to_dicts(rows):
    return [
        {
            "id": r.id,
            "username": r.username,
            "ip": r.ip or "",
            "user_agent": (r.user_agent or "")[:60],
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r.created_at)),
        }
        for r in rows
    ]
