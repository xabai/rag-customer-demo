"""
细粒度 ACL 权限检查

两层机制：
1. role_permissions —— 角色→权限（RBAC，如 "user.read"、"kb.write"）
2. resource_acls —— 用户/角色→资源（ACL，哪个人/角色对哪个 KB/Agent 有权）

使用：
    require_permission("kb.write")
    check_resource_access(user, resource_type="kb", resource_id=1, action="write")
"""
from __future__ import annotations

from functools import wraps
from typing import Callable

from fastapi import Depends, HTTPException, Request

from .db_engine import get_db
from .models import ResourceACL, RolePermission


def _has_role_permission(role: str, permission: str, db) -> bool:
    if role == "superadmin":
        return True
    # 通配符 "*" 表示该角色拥有全部权限
    star = db.query(RolePermission).filter_by(role=role, permission="*").first()
    if star:
        return True
    return db.query(RolePermission).filter_by(role=role, permission=permission).first() is not None


def has_permission(user: dict | None, permission: str, db=None) -> bool:
    """检查用户是否拥有指定权限"""
    if not user:
        return False
    role = user.get("role", "")
    if role == "superadmin":
        return True

    if db is None:
        from .db_engine import SessionLocal
        db = SessionLocal()
        try:
            return _has_role_permission(role, permission, db)
        finally:
            db.close()
    return _has_role_permission(role, permission, db)


def check_resource_access(user: dict, resource_type: str, resource_id: int,
                          action: str = "read", db=None) -> bool:
    """检查用户对特定资源是否有权限"""
    if not user:
        return False
    role = user.get("role", "")
    if role in ("superadmin", "admin"):
        return True

    action_map = {"read": "can_read", "write": "can_write",
                  "delete": "can_delete", "share": "can_share"}
    flag = action_map.get(action, "can_read")

    if db is None:
        from .db_engine import SessionLocal
        db = SessionLocal()
        close_db = True
    else:
        close_db = False

    try:
        # 用户级 ACL
        acl = db.query(ResourceACL).filter_by(
            subject_type="user", subject_id=user["id"],
            resource_type=resource_type, resource_id=resource_id
        ).first()
        if acl and getattr(acl, flag, False):
            return True

        # 角色级 ACL
        acl = db.query(ResourceACL).filter_by(
            subject_type="role", subject_role=role,
            resource_type=resource_type, resource_id=resource_id
        ).first()
        if acl and getattr(acl, flag, False):
            return True

        # Agent 特殊：agent 角色对它关联的 KB 应该有 read
        return False
    finally:
        if close_db:
            db.close()


def require_permission(permission: str):
    """FastAPI 依赖 —— 需要指定权限才能访问"""
    def _check(request: Request, user: dict = Depends("app.get_current_user"), db=Depends(get_db)):
        if not _has_role_permission(user.get("role", ""), permission, db):
            raise HTTPException(403, f"权限不足：需要 {permission}")
        return user
    return _check


def require_resource_access(resource_type: str, action: str = "read",
                            extract_id: Callable | None = None):
    """FastAPI 依赖 —— 需要对特定资源有权限"""
    def _check(request: Request, user: dict = Depends("app.get_current_user"), db=Depends(get_db)):
        rid = extract_id(request) if extract_id else request.path_params.get("id")
        if rid is None:
            return user
        rid = int(rid)
        if not check_resource_access(user, resource_type, rid, action, db=db):
            raise HTTPException(403, f"对 {resource_type}#{rid} 无 {action} 权限")
        return user
    return _check
