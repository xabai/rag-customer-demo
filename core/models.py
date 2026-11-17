"""
SQLAlchemy ORM 模型 —— 统一所有表结构

设计要点：
- 每个表带 id (BigInteger PK) + created_at + updated_at + deleted（软删除）
- UUID 用 pg-uuid 扩展，主键用 BigInteger 自增
- 审计相关表带 actor_user_id/actor_username 便于 JOIN 查询
- 多租户：所有业务表带 tenant_id（当前版本单租户固定为 1，预留）
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger, Boolean, CheckConstraint, DateTime, Float, ForeignKey,
    Index, Integer, String, Text, UniqueConstraint, func, event,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------- 时间戳工具 ----------
def _utcnow_ts() -> float:
    return time.time()

def _utcnow_dt() -> datetime:
    return datetime.now(timezone.utc)


# ---------- 基类 ----------
class Base(DeclarativeBase):
    """所有 ORM 模型的基类"""
    pass


class _TimestampMixin:
    """自动时间戳 mixin（使用 Python time.time() 以兼容 SQLite/PG）"""
    created_at: Mapped[float] = mapped_column(Float, default=_utcnow_ts, nullable=False)
    updated_at: Mapped[float] = mapped_column(Float, default=_utcnow_ts,
                                              onupdate=_utcnow_ts, nullable=False)


class _TenantMixin:
    """多租户预留（当前版本所有租户 ID = 1）"""
    tenant_id: Mapped[int] = mapped_column(Integer, default=1, nullable=False, index=True)


# ================================================================
# 1. 组织架构
# ================================================================
class Organization(Base, _TimestampMixin):
    """组织/公司"""
    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    code: Mapped[str] = mapped_column(String(32), nullable=False, unique=True, index=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)


class Department(Base, _TimestampMixin):
    """部门"""
    __tablename__ = "departments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    org_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("organizations.id", ondelete="CASCADE"), index=True)
    parent_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("departments.id", ondelete="SET NULL"), nullable=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    __table_args__ = (UniqueConstraint("org_id", "name", name="uq_dept_org_name"),)


# ================================================================
# 2. 用户 / 角色 / 权限（RBAC + ACL）
# ================================================================
class User(Base, _TimestampMixin, _TenantMixin):
    """用户表 —— 新增部门/组织关联 + 锁定字段 + 密码历史"""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="user", index=True)
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    # 组织架构关联（新增）
    org_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("organizations.id"), nullable=True)
    dept_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("departments.id"), nullable=True)

    # 登录锁定（已迁移自旧表）
    failed_attempts: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    lockout_until: Mapped[float | None] = mapped_column(Float, nullable=True)

    # 最后登录
    last_login_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    last_login_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # 软删除预留
    deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (CheckConstraint("role IN ('user','agent','admin','superadmin','owner')", name="ck_user_role"),)


class PasswordHistory(Base, _TimestampMixin):
    """密码历史 —— 用于防止最近 N 次密码重复"""
    __tablename__ = "password_history"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)

    __table_args__ = (Index("ix_pw_history_user_time", "user_id", "created_at"),)


class RolePermission(Base):
    """角色-权限 映射（细粒度 RBAC）"""
    __tablename__ = "role_permissions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    role: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    permission: Mapped[str] = mapped_column(String(128), nullable=False, index=True)

    __table_args__ = (UniqueConstraint("role", "permission", name="uq_role_perm"),)


class ResourceACL(Base, _TimestampMixin):
    """资源级 ACL —— 哪个用户/角色对哪个资源有权限"""
    __tablename__ = "resource_acls"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    # 主体：user 或 role
    subject_type: Mapped[str] = mapped_column(String(8), nullable=False)  # 'user' / 'role'
    subject_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)  # user_id 或 null
    subject_role: Mapped[str | None] = mapped_column(String(32), nullable=True)  # role 名
    # 资源
    resource_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)  # 'kb' / 'agent' / 'session'
    resource_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    # 权限
    can_read: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    can_write: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    can_delete: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    can_share: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (
        CheckConstraint("subject_type IN ('user','role')", name="ck_acl_subject"),
        Index("ix_acl_resource", "resource_type", "resource_id"),
    )


# ================================================================
# 3. 认证：Session / 登录审计
# ================================================================
class UserSession(Base, _TimestampMixin):
    """服务端 Session 白名单 —— JWT 签发后在此登记，logout/kick 时删除"""
    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    expires_at: Mapped[float] = mapped_column(Float, nullable=False, index=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False, index=True)


class LoginFailure(Base, _TimestampMixin):
    """登录失败审计"""
    __tablename__ = "auth_failures"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (Index("ix_auth_fail_time", "created_at"),)


# ================================================================
# 4. 审计日志（全局）
# ================================================================
class AuditLog(Base, _TimestampMixin, _TenantMixin):
    """全局安全审计日志"""
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    actor_user_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    actor_username: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    resource_type: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)
    resource_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True, index=True)
    ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[str] = mapped_column(String(16), nullable=False, default="success")  # success / fail / warn
    detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)

    __table_args__ = (
        Index("ix_audit_time_action", "created_at", "action"),
        Index("ix_audit_trace", "trace_id"),
    )


# ================================================================
# 5. 知识库 / 文档
# ================================================================
class KnowledgeBase(Base, _TimestampMixin, _TenantMixin):
    """知识库"""
    __tablename__ = "kb"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    collection_name: Mapped[str] = mapped_column(String(128), nullable=False)
    doc_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    owner_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=True, index=True)


class Document(Base, _TimestampMixin, _TenantMixin):
    """知识库文档"""
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    kb_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("kb.id", ondelete="CASCADE"), index=True, nullable=False)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    doc_type: Mapped[str] = mapped_column(String(16), nullable=False)  # pdf / docx / url / txt
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    index_status: Mapped[str] = mapped_column(String(16), default="done", nullable=False)  # pending / processing / done / failed
    index_error: Mapped[str | None] = mapped_column(Text, nullable=True)


# ================================================================
# 6. 会话 / 消息
# ================================================================
class ChatSession(Base, _TimestampMixin, _TenantMixin):
    """聊天会话"""
    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    user_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    user_agent_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("agents.id"), nullable=True, index=True)


class ChatMessage(Base, _TimestampMixin, _TenantMixin):
    """会话消息"""
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("sessions.id", ondelete="CASCADE"), index=True, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user / assistant / system
    content: Mapped[Text] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model: Mapped[str | None] = mapped_column(String(64), nullable=True)


# ================================================================
# 7. Agent
# ================================================================
class Agent(Base, _TimestampMixin, _TenantMixin):
    """智能体"""
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    system_prompt: Mapped[str | None] = mapped_column(Text, nullable=True)
    model: Mapped[str] = mapped_column(String(64), nullable=False, default="qwen2.5:7b")
    fallback_model: Mapped[str | None] = mapped_column(String(64), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    owner_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=True, index=True)


class AgentKB(Base):
    """Agent ↔ KB 多对多"""
    __tablename__ = "agent_kb"

    agent_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("agents.id", ondelete="CASCADE"), primary_key=True)
    kb_id: Mapped[int] = mapped_column(BigInteger, ForeignKey("kb.id", ondelete="CASCADE"), primary_key=True)


# ================================================================
# 8. 记忆
# ================================================================
class Memory(Base, _TimestampMixin, _TenantMixin):
    """用户记忆"""
    __tablename__ = "memories"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True)
    session_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("sessions.id", ondelete="CASCADE"), index=True, nullable=True)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, index=True)  # fact / preference / context
    content: Mapped[str] = mapped_column(Text, nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)


# ================================================================
# 9. FAQ / 意图规则 / Ticket / 反馈
# ================================================================
class Faq(Base, _TimestampMixin, _TenantMixin):
    """FAQ"""
    __tablename__ = "faq"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class IntentRule(Base, _TimestampMixin, _TenantMixin):
    """意图识别规则"""
    __tablename__ = "intent_rules"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    intent: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    pattern: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str | None] = mapped_column(Text, nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    priority: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class Ticket(Base, _TimestampMixin, _TenantMixin):
    """工单/客服"""
    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="open", nullable=False, index=True)
    assignee_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=True, index=True)


class Feedback(Base, _TimestampMixin, _TenantMixin):
    """用户反馈"""
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True)
    rating: Mapped[int | None] = mapped_column(Integer, nullable=True)  # 1-5
    comment: Mapped[str | None] = mapped_column(Text, nullable=True)
    qa_log_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)


class QALog(Base, _TimestampMixin, _TenantMixin):
    """Q&A 日志（对话级）"""
    __tablename__ = "qa_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str | None] = mapped_column(String(32), nullable=True)  # vector / bm25 / hybrid / faq / intent
    tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Handoff(Base, _TimestampMixin, _TenantMixin):
    """人工转交"""
    __tablename__ = "handoffs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True, index=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False, index=True)


class Unanswered(Base, _TimestampMixin, _TenantMixin):
    """未回答问题"""
    __tablename__ = "unanswered"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    session_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)


# ================================================================
# 10. 异步任务
# ================================================================
class AsyncTask(Base, _TimestampMixin, _TenantMixin):
    """异步任务队列表（DB 作为 fallback queue）"""
    __tablename__ = "async_tasks"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    task_type: Mapped[str] = mapped_column(String(64), nullable=False, index=True)  # kb_index / export / email
    payload_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="pending", nullable=False, index=True)  # pending / running / done / failed
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    started_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    finished_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    max_retries: Mapped[int] = mapped_column(Integer, default=3, nullable=False)

    __table_args__ = (Index("ix_async_status_type", "status", "task_type"),)


# ================================================================
# 11. 成本追踪
# ================================================================
class CostLog(Base, _TimestampMixin, _TenantMixin):
    """每次 LLM 调用成本"""
    __tablename__ = "cost_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("users.id"), nullable=True, index=True)
    session_id: Mapped[int | None] = mapped_column(BigInteger, ForeignKey("sessions.id"), nullable=True, index=True)
    model: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    __table_args__ = (
        Index("ix_cost_user_day", "user_id", "created_at"),
        Index("ix_cost_model", "model"),
    )


# ================================================================
# 12. 系统配置
# ================================================================
class Setting(Base):
    """KV 型系统配置"""
    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    key: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    value: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(String(255), nullable=True)
    updated_at: Mapped[float] = mapped_column(Float, default=_utcnow_ts, nullable=False)
