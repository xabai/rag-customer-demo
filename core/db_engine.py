"""
SQLAlchemy 引擎 + Session 管理

策略：
- 统一 asyncio 安全的 SessionLocal（同步，uvicorn 多 worker 各自有 engine）
- FastAPI 用依赖注入 get_db()
- SQLite 和 PostgreSQL 用同一套 API，只换 DB_DRIVER
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .config import get_settings
from .models import Base


_settings = get_settings()

# ---------- Engine ----------
_engine_kwargs: dict = {}
if _settings.DB_DRIVER == "sqlite":
    # SQLite 不能多线程，check_same_thread=False；而且 SQLite 不支持 pool_size
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    _engine_kwargs["pool_pre_ping"] = True
else:
    _engine_kwargs["pool_size"] = _settings.DB_POOL_SIZE
    _engine_kwargs["max_overflow"] = _settings.DB_MAX_OVERFLOW
    _engine_kwargs["pool_pre_ping"] = True
    _engine_kwargs["pool_recycle"] = 3600  # PG 连接超时默认 8h，主动回收避免断连


def _build_engine() -> Engine:
    eng = create_engine(_settings.DATABASE_URL, **_engine_kwargs, future=True)

    # SQLite 的 FK 必须每个连接开启
    if _settings.DB_DRIVER == "sqlite":
        @event.listens_for(eng, "connect")
        def _enable_fk(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.close()

    return eng


engine = _build_engine()

# ---------- Session ----------
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


# ---------- 依赖注入 ----------
def get_db() -> Iterator[Session]:
    """FastAPI 依赖 —— 自动 commit/rollback/close"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def db_session() -> Iterator[Session]:
    """非 FastAPI 场景用（脚本、定时任务）"""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ---------- 初始化 ----------
def init_db(create_defaults: bool = True) -> None:
    """建表（幂等，用 Alembic 迁移后只作为 fallback）"""
    import logging
    log = logging.getLogger("rag.db")

    log.info("Creating tables (if not exist)...")
    Base.metadata.create_all(bind=engine)
    log.info("Tables ensured.")

    if create_defaults:
        _create_default_data()


def _create_default_data() -> None:
    """首次启动时创建默认组织 + superadmin + 角色权限"""
    from .auth import hash_password
    from .models import (
        Organization, User, RolePermission, Setting, Agent, AgentKB,
    )

    with db_session() as db:
        # 默认组织
        if db.query(Organization).count() == 0:
            org = Organization(name="默认组织", code="DEFAULT")
            db.add(org)
            db.flush()

        # 默认角色权限（细粒度）
        _DEFAULT_PERMS: dict[str, list[str]] = {
            "superadmin": ["*"],
            "admin": [
                "kb.read", "kb.write", "kb.delete", "kb.share",
                "agent.read", "agent.write", "agent.delete",
                "memory.read", "memory.write", "memory.delete",
                "user.read", "user.update", "user.disable", "user.force_logout",
                "audit.read", "audit.export",
                "settings.read",
            ],
            "agent": [
                "kb.read", "kb.write",
                "memory.read", "memory.write",
                "chat.send",
            ],
            "user": [
                "chat.send",
                "memory.read", "memory.write", "memory.delete",
                "session.read", "session.delete",
            ],
        }
        for role, perms in _DEFAULT_PERMS.items():
            for p in perms:
                exists = db.query(RolePermission).filter_by(role=role, permission=p).first()
                if not exists:
                    db.add(RolePermission(role=role, permission=p))

        # 默认用户
        if db.query(User).count() == 0:
            db.add(User(
                username="admin",
                password_hash=hash_password("admin123"),
                role="superadmin",
                email="admin@rag.local",
                enabled=True,
            ))
            db.add(User(
                username="user",
                password_hash=hash_password("user123"),
                role="user",
                email="user@rag.local",
                enabled=True,
            ))

        # 默认设置
        _DEFAULT_SETTINGS = [
            ("rag.default_model", "qwen2.5:7b", "默认对话模型"),
            ("rag.top_k", "5", "检索 top_k"),
            ("rag.similarity_threshold", "0.5", "相似度阈值"),
            ("rag.max_chunk_size", "512", "分块最大长度"),
            ("content.safety.enabled", "false", "内容安全开关"),
            ("async.worker.concurrency", "4", "异步任务并发数"),
        ]
        for key, value, desc in _DEFAULT_SETTINGS:
            if not db.query(Setting).filter_by(key=key).first():
                db.add(Setting(key=key, value=value, description=desc))


def table_names() -> list[str]:
    """返回所有表名（用于健康检查）"""
    from sqlalchemy import inspect
    insp = inspect(engine)
    return sorted(insp.get_table_names())
