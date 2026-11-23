# -*- coding: utf-8 -*-
"""FastAPI 入口：知识库管理 + 多会话流式聊天"""
import asyncio
import json
import os
import re
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator

import config
import db
from core import auth as _auth
from core import chunker, intent, memory, ollama_client, parser, rag, settings, tools
from core.logger import setup_logger, get_logger
from core.security import AuditMiddleware, _client_ip, require_reverify
from core.vector_store import store

# ======== 新增：企业级模块 ========
from core.config import get_settings as _settings
from core.db_engine import init_db as orm_init_db
from core import metrics as _metrics
from core.health import router as _health_router
from core import tracing as _tracing
from core import async_queue as _async_queue
from core import content_safety as _content_safety
from core import password_policy as _pw_policy
from core import cost_tracker as _cost
from core import exporters as _exporters
from core import validators as _v
from core import acl as _acl
from core import alerts as _alerts

# ======== 初始化统一日志 ========
setup_logger()
_log = get_logger("app")
_settings = _settings()

# ======== Tracing（可选）=======
_tracing.setup_tracing()

# ======== FastAPI 应用 ========
app = FastAPI(
    title="RAG AI 客服",
    version="2.0.0-enterprise",
    description="RAG AI 客服 —— 企业级安全加固版",
    docs_url="/docs",   # 保留 Swagger UI 作为 API 文档
    redoc_url="/redoc",
)

# ---------- CORS ----------
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=_settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- 中间件（注意顺序：最先加的最后执行）----------
app.add_middleware(AuditMiddleware)                               # ① 审计
app.add_middleware(_metrics.MetricsMiddleware)                     # ② Prometheus metrics
app.add_middleware(_content_safety.ContentSafetyHTTPMiddleware)    # ③ 内容安全

# ---------- 数据库 ----------
if _settings.DB_DRIVER != "postgresql":
    db.init_db()  # SQLite 初始化（PG 模式下跳过）
if _settings.DB_DRIVER == "postgresql":
    orm_init_db(create_defaults=True)
    _log.info("PostgreSQL ORM initialized (26 tables)")
else:
    _log.info("Using SQLite (DB_DRIVER=%s)", _settings.DB_DRIVER)

# ---------- 异步 Worker 启动 ----------
@app.on_event("startup")
async def _start_async_worker():
    if _settings.ASYNC_QUEUE_ENABLED:
        asyncio.create_task(_async_queue.run_worker_loop())
        _log.info("Async worker loop started")

# ---------- Metrics + Health 路由 ----------
@app.get("/metrics", include_in_schema=False)
async def _metrics_endpoint():
    return await _metrics.metrics_endpoint()

app.include_router(_health_router)  # /api/health

_log.info("Application startup complete (env=%s)", _settings.APP_ENV)


# ---------- RBAC Dependency ----------
async def get_current_user(request: Request, authorization: str = Header(default="")) -> dict:
    """从 Authorization: Bearer <token> 解析 JWT + 校验服务端 session 白名单"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "未登录")
    token = authorization[7:].strip()
    payload = _auth.verify_token(token)
    if not payload:
        raise HTTPException(401, "登录已失效，请重新登录")
    # sub 在 JWT 里必须是 string，额外存了 uid 方便转 int
    user_id = int(payload.get("uid") or payload.get("sub", 0))
    user = db.get_user_by_id(user_id)
    if not user:
        raise HTTPException(401, "用户不存在")
    if not user.get("enabled", True):
        raise HTTPException(403, "账号已被禁用")
    # 附带 payload.sid 方便主动 logout
    user["sid"] = payload.get("sid")
    # 挂到 request.state 供审计中间件读取
    request.state.actor = user
    return user


def require_role(required: str):
    """角色守卫：user < admin < superadmin"""
    def _check(user: dict = Depends(get_current_user)) -> dict:
        if not _auth.role_can(user.get("role", ""), required):
            raise HTTPException(403, f"需要 {required} 权限，当前为 {user.get('role')}")
        return user
    return _check


# ---------- Auth 路由 ----------
class LoginBody(BaseModel):
    username: str
    password: str


class RegisterBody(BaseModel):
    username: str
    password: str


class ChangePwdBody(BaseModel):
    old_password: str
    new_password: str


@app.post("/api/auth/login")
async def login(body: LoginBody, request: Request):
    ip = _client_ip(request)
    ua = request.headers.get("user-agent", "")
    result = _auth.authenticate(body.username, body.password, ip=ip, user_agent=ua)
    if result is None:
        raise HTTPException(401, "用户名或密码错误")
    if isinstance(result, dict) and result.get("locked"):
        return JSONResponse(
            status_code=423,
            content={
                "detail": "账号已被临时锁定，请稍后再试",
                "lockout_seconds": result.get("lockout_seconds", 600),
            },
        )
    if isinstance(result, dict) and result.get("disabled"):
        raise HTTPException(403, "账号已被禁用，请联系管理员")
    # 挂 actor 供审计中间件（login 是 public 接口中间件不自动挂）
    request.state.actor = {"id": result["user_id"], "username": result["username"], "role": result["role"]}
    return {
        "token": result["token"],
        "username": result["username"],
        "role": result["role"],
        "user_id": result["user_id"],
    }


@app.post("/api/auth/register")
async def register(body: RegisterBody):
    """开放注册：默认注册为 user 角色；如果没有任何用户，第一个注册自动升 superadmin"""
    if not body.username.strip() or not body.password.strip():
        raise HTTPException(400, "用户名和密码不能为空")
    # A3 密码策略校验
    valid, err = _pw_policy.policy.validate(body.password)
    if not valid:
        raise HTTPException(400, err)
    existing = db.get_user_by_username(body.username.strip())
    if existing:
        raise HTTPException(400, "用户名已存在")
    role = "superadmin" if len(db.list_users()) == 0 else "user"
    uid = db.create_user(body.username.strip(), body.password, role=role)
    if uid is None:
        raise HTTPException(500, "注册失败")
    # A3 记录密码历史
    _pw_policy.policy.record_password(uid, _auth.hash_password(body.password))
    return {"ok": True, "id": uid, "role": role}


@app.post("/api/auth/logout")
async def logout(user: dict = Depends(get_current_user), authorization: str = Header(default="")):
    token = authorization[7:].strip() if authorization.startswith("Bearer ") else ""
    if token:
        _auth.logout(token)
    return {"ok": True}


@app.get("/api/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return user


@app.post("/api/auth/change-password")
async def change_password(body: ChangePwdBody, user: dict = Depends(get_current_user)):
    raw = db.get_user_by_username(user["username"])
    if not raw or not _auth.verify_password(body.old_password, raw.get("password", "")):
        raise HTTPException(400, "原密码错误")
    # A3 密码策略 + 历史检查
    valid, err = _pw_policy.policy.validate(body.new_password)
    if not valid:
        raise HTTPException(400, err)
    history_ok, herr = _pw_policy.policy.check_history(body.new_password, user["id"])
    if not history_ok:
        raise HTTPException(400, herr)
    db.change_password(user["id"], body.new_password)
    _pw_policy.policy.record_password(user["id"], _auth.hash_password(body.new_password))
    return {"ok": True}


# ---------- 用户管理（admin+） ----------
class UserCreateBody(BaseModel):
    username: str
    password: str
    role: str = "user"  # user | admin | superadmin


class UserUpdateBody(BaseModel):
    username: str | None = None
    role: str | None = None
    enabled: bool | None = None
    password: str | None = None


@app.get("/api/users")
async def list_users(_: dict = Depends(require_role("admin"))):
    users = db.list_users()
    orgs, depts = _org_dept_maps()
    role_map = {r["name"]: r["display_name"] for r in db.list_roles()}
    for u in users:
        u["org_name"] = orgs.get(u.get("org_id")) or ""
        u["dept_name"] = depts.get(u.get("dept_id")) or ""
        u["role_label"] = role_map.get(u.get("role"), u.get("role"))
    return users


@app.post("/api/users")
async def create_user(body: UserCreateBody, admin: dict = Depends(require_role("admin"))):
    if body.role not in ("user", "admin", "superadmin"):
        raise HTTPException(400, "role 必须是 user / admin / superadmin")
    # 只有 superadmin 能创建 superadmin
    if body.role == "superadmin" and admin.get("role") != "superadmin":
        raise HTTPException(403, "只有 superadmin 能创建 superadmin")
    # A3 密码策略
    valid, err = _pw_policy.policy.validate(body.password)
    if not valid:
        raise HTTPException(400, err)
    uid = db.create_user(body.username.strip(), body.password, role=body.role)
    if uid:
        _pw_policy.policy.record_password(uid, _auth.hash_password(body.password))
    if uid is None:
        raise HTTPException(400, "用户名已存在")
    return {"ok": True, "id": uid}


@app.put("/api/users/{user_id}")
async def update_user(user_id: int, body: UserUpdateBody, admin: dict = Depends(require_role("admin"))):
    target = db.get_user_by_id(user_id)
    if not target:
        raise HTTPException(404, "用户不存在")
    # 权限门禁：admin 不能把别人升到比自己高的角色
    new_role = body.role
    if new_role and not _auth.role_can(admin["role"], new_role):
        raise HTTPException(403, f"无权授予 {new_role} 角色")
    # 不能修改 superadmin 的角色/禁用（除非自己是 superadmin）
    if target["role"] == "superadmin" and admin.get("role") != "superadmin":
        if new_role or body.enabled is not None:
            raise HTTPException(403, "无权修改 superadmin 账号")
    kwargs = {k: v for k, v in
              [("username", body.username), ("role", new_role),
               ("enabled", body.enabled), ("password", body.password)]
              if v is not None}
    ok = db.update_user(user_id, **kwargs)
    if not ok:
        raise HTTPException(400, "更新失败")
    return {"ok": True}


@app.delete("/api/users/{user_id}")
async def delete_user(user_id: int, admin: dict = Depends(require_role("admin"))):
    target = db.get_user_by_id(user_id)
    if not target:
        raise HTTPException(404, "用户不存在")
    # superadmin 不能被任何角色删除
    if target["role"] == "superadmin":
        raise HTTPException(403, "不能删除 superadmin")
    # admin 不能删比自己高/同级（非 superadmin 的 admin）
    if not _auth.role_can(admin["role"], target["role"]):
        raise HTTPException(403, f"无权删除 {target['role']} 用户")
    ok = db.delete_user(user_id)
    if not ok:
        raise HTTPException(400, "删除失败")
    return {"ok": True}


@app.post("/api/users/{user_id}/force-logout")
async def force_logout(user_id: int, body: dict, admin: dict = Depends(require_role("admin"))):
    # 二次验证：admin 输入自己的密码
    pwd = body.get("password", "")
    if not pwd:
        raise HTTPException(400, "需要密码二次验证")
    raw = db.get_user_by_username(admin["username"])
    if not raw or not _auth.verify_password(pwd, raw.get("password", "")):
        # 二次验证失败 → 记一次锁定计数
        db.record_login_failure(admin["username"])
        raise HTTPException(400, "密码错误")
    count = db.revoke_all_user_sessions(user_id)
    return {"ok": True, "revoked_sessions": count}


@app.post("/api/users/{user_id}/toggle")
async def toggle_user(user_id: int, body: dict, admin: dict = Depends(require_role("admin"))):
    # 二次验证
    pwd = body.get("password", "")
    if not pwd:
        raise HTTPException(400, "需要密码二次验证")
    raw = db.get_user_by_username(admin["username"])
    if not raw or not _auth.verify_password(pwd, raw.get("password", "")):
        db.record_login_failure(admin["username"])
        raise HTTPException(400, "密码错误")
    enabled = bool(body.get("enabled", True))
    ok = db.toggle_user(user_id, enabled)
    if not ok:
        raise HTTPException(400, "操作失败")
    if not enabled:
        db.revoke_all_user_sessions(user_id)
    return {"ok": True}

# 挂载知识库图片目录（前端用 /kb_images/xxx.png 直接引用）
from fastapi.staticfiles import StaticFiles as _SF
if os.path.isdir(config.KB_IMG_DIR):
    app.mount("/kb_images", _SF(directory=config.KB_IMG_DIR), name="kb_images")


# ---------- 知识库 ----------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...), kb_id: str = Form("default"),
                 chunk_config: str = Form(""), embed_model: str = Form(""),
                 _admin: dict = Depends(require_role("admin"))):
    import json as _json
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in config.ALLOWED_EXT:
        raise HTTPException(400, f"不支持的文件类型 {ext}，支持: {', '.join(sorted(config.ALLOWED_EXT))}")

    path = os.path.join(config.UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 1) 优先用前端传来的 chunk_config；没有则从 KB 的 chunk_config 取；都没有用默认值
        cfg = None
        if chunk_config:
            try:
                cfg = _json.loads(chunk_config)
            except Exception:
                cfg = None
        if not cfg and kb_id:
            try:
                kb_obj = db.get_kb(kb_id)
                if kb_obj and kb_obj.get("chunk_config"):
                    cfg = kb_obj["chunk_config"]
            except Exception:
                pass
        # 默认 cfg：父子分块·段落模式 + 默认分隔符（和 Dify 一致）
        if not cfg:
            cfg = {
                "mode": "parent_child",
                "parent_strategy": "paragraph",
                "parent_separator": "\n\n",
                "parent_max_length": 1024,
                "child_separator": None,
                "child_max_length": 300,
            }

        # 归一化：新旧字段 → 内部统一结构
        norm = parser._normalize_chunk_config(cfg)

        # 🔍 DEBUG：打印归一化后的 cfg
        print(f"[UPLOAD DEBUG] raw cfg = {cfg}")
        print(f"[UPLOAD DEBUG] norm cfg = {norm}")

        # 优先走版面感知管线（元素归一化 + 图文配对 + 版面分块 + 父子分块）
        use_layout = ext in (".pdf", ".pptx", ".ppt", ".docx", ".xlsx", ".xls")
        parent_ids = None  # 先初始化，保证两条管线都能赋值
        if use_layout:
            try:
                elements = parser.parse_file_layout(path)

                # build_layout_chunks 的 parent_separator：只要用户填了就传
                # - 父子·段落模式：传 parent_separator（用于切父块边界）
                # - 通用模式：传 separator（用于切 flat 块边界）
                # - 父子·全文模式：不传（整篇=1父块，layout 怎么切都无所谓）
                if norm["mode"] == "parent_child" and norm.get("parent_strategy") == "document":
                    # 全文模式：让它按版面/heading 随便切，后面 layout_chunks_to_records 会整篇拼
                    parent_size = None
                    parent_sep = None
                elif norm["mode"] == "parent_child":
                    # 父子·段落模式
                    parent_size = norm.get("parent_max_length", 1024)
                    parent_sep = norm.get("parent_separator") or None
                else:
                    # 通用模式：用户填的 separator 也用作 layout element 的边界切割
                    parent_size = None
                    parent_sep = norm.get("separator") or None

                chunks = parser.build_layout_chunks(
                    elements,
                    parent_size=parent_size,
                    parent_separator=parent_sep,
                    # build_layout_chunks 不处理子块分隔符（子块切分在 layout_chunks_to_records 里）
                    child_separator=None,
                )
                chunk_texts, parent_texts, chunk_images, layout_meta, parent_ids = \
                    parser.layout_chunks_to_records(chunks, chunk_config=norm)
                if not chunk_texts:
                    raise ValueError("layout 解析为空，降级")
            except Exception as layout_err:
                print(f"[upload] layout 管线失败，降级: {layout_err}")
                import traceback
                traceback.print_exc()
                use_layout = False

        if not use_layout:
            text = parser.parse_file(path)
            if not text or not text.strip():
                raise ValueError("文件为空或无法解析文本")

            mode = norm["mode"]

            if mode == "parent_child":
                # ====== 父子模式 ======
                parent_strategy = norm.get("parent_strategy", "paragraph")
                child_sep = norm.get("child_separator") or None
                child_max = norm.get("child_max_length", 300)
                overlap = 0  # Dify 逻辑：父子模式 overlap 强制 0

                if parent_strategy == "document":
                    # ---- 全文模式：整篇文档 = 1 个父块 ----
                    children = parser._split_parent_into_children(text, child_max, overlap,
                                                                  child_sep=child_sep)
                    chunk_texts = children
                    parent_texts = [text.strip()] * len(children)
                    parent_ids = ["parent_document"] * len(children)
                else:
                    # ---- 段落模式：先按 parent_separator 切父块，再按 child_separator 切子块 ----
                    parent_sep = norm.get("parent_separator")
                    parent_max = norm.get("parent_max_length", 1024)

                    if parent_sep:
                        sep_re = parser._compile_separator(parent_sep, hint=text[:500])
                        if sep_re:
                            raw_parts = sep_re.split(text)
                        else:
                            raw_parts = re.split(r'\n\s*\n', text)
                    else:
                        raw_parts = re.split(r'\n\s*\n', text)

                    raw_parts = [p.strip() for p in raw_parts if p and p.strip()]

                    # 聚合到 parent_max
                    parent_buf = ""
                    parent_chunks = []
                    for p in raw_parts:
                        if len(parent_buf) + len(p) <= parent_max:
                            parent_buf += p + "\n"
                        else:
                            if parent_buf:
                                parent_chunks.append(parent_buf.strip())
                            parent_buf = p + "\n"
                    if parent_buf:
                        parent_chunks.append(parent_buf.strip())

                    chunk_texts, parent_texts, parent_ids = [], [], []
                    for pi, pt in enumerate(parent_chunks):
                        children = parser._split_parent_into_children(pt, child_max, overlap,
                                                                      child_sep=child_sep)
                        for child in children:
                            chunk_texts.append(child)
                            parent_texts.append(pt)
                            parent_ids.append(f"parent_{pi}")
            else:
                # ====== 通用模式（flat）======
                sep = norm.get("separator")
                max_len = norm.get("max_length", 500)
                overlap = norm.get("overlap", 50)

                if sep:
                    sep_re = parser._compile_separator(sep, hint=text[:500])
                    if sep_re:
                        parts = sep_re.split(text)
                    else:
                        parts = [text]
                else:
                    parts = re.split(r'\n\s*\n', text)

                parts = [p.strip() for p in parts if p and p.strip()]
                if not parts:
                    parts = [text]

                # ← 关键修复：不要把 parts join 回去再硬切！
                # 每个 part 内部独立按 max_length + overlap 切，overlap 只在 part 内部生效
                chunk_texts = []
                for p in parts:
                    if len(p) <= max_len:
                        chunk_texts.append(p)
                    else:
                        sub_pieces = parser._split_parent_into_children(
                            p, max_len, overlap, child_sep=None
                        )
                        chunk_texts.extend(sub_pieces)
                parent_texts = None
                parent_ids = None

            chunk_images = None
            layout_meta = None

        if not chunk_texts:
            raise ValueError("未从文件中解析出有效文本")

        # 文本向量（bge-m3 / 可选换其他 embedding 模型）
        vectors = ollama_client.embed_texts(chunk_texts, model=embed_model or None)

        # 图片向量（qwen3-vl-embed）：含图片块才编码
        image_vectors = None
        if chunk_images:
            img_paths = [c.split(";")[0] if c else None for c in chunk_images]
            valid_paths = [(i, p) for i, p in enumerate(img_paths) if p and os.path.exists(p)]
            if valid_paths:
                all_iv = ollama_client.embed_images([p for _, p in valid_paths])
                # 对齐到 chunks 索引：无图片的 chunk 用零向量占位
                zeros = np.zeros((len(chunk_texts), config.VL_EMBED_DIM), dtype=np.float32)
                for idx, (orig_i, _) in enumerate(valid_paths):
                    if idx < len(all_iv):
                        zeros[orig_i] = all_iv[idx]
                image_vectors = zeros

        # ★ metadata：文件级 + chunk 级元数据
        doc_type = ext.lstrip(".")
        file_mtime = os.path.getmtime(path)
        file_size = os.path.getsize(path)
        shared_metadata = {
            "doc_type": doc_type,
            "source_path": path,
            "file_size": file_size,
            "file_mtime": file_mtime,
        }
        # 尝试从文件名/内容提取"生效日期"（过期检测）
        import re as _re
        def _extract_valid_dates(text):
            """从 chunk 文本里正则提取生效/过期日期 → valid_from, valid_to"""
            from datetime import datetime, timezone as _tz
            results = {}
            # "生效日期：2024-01-01" / "有效期至 2025年6月"
            date_pat = r"(\d{4})[-/年.](\d{1,2})[-/月.](\d{1,2})"
            m_from = _re.search(r"(生效|施行|启用|开始)[^0-9]*?" + date_pat, text)
            m_to = _re.search(r"(有效期|截止|结束|终止|到期)[^0-9]*?" + date_pat, text)
            if m_from:
                y, mo, d = m_from.group(2), m_from.group(3), m_from.group(4)
                try: results["valid_from"] = datetime(int(y), int(mo), int(d)).timestamp()
                except: pass
            if m_to:
                y, mo, d = m_to.group(2), m_to.group(3), m_to.group(4)
                try: results["valid_to"] = datetime(int(y), int(mo), int(d)).timestamp()
                except: pass
            return results
        # 批量提取：对每个 chunk 查一遍
        chunk_metadata_list = []
        for ct in chunk_texts:
            md = dict(shared_metadata)
            md.update(_extract_valid_dates(ct))
            chunk_metadata_list.append(md)

        # ★ 重名检测：同 kb_id 同文件名 → 原子替换（update_document）
        existing_doc_id = None
        for d in store.list_documents():
            if d.get("name") == file.filename and d.get("kb_id", "default") == kb_id:
                existing_doc_id = d["id"]
                break

        if existing_doc_id:
            print(f"[upload] 检测到重名文档 {file.filename} (kb={kb_id})，走 update_document 替换旧索引")
            doc_id = store.update_document(
                existing_doc_id, file.filename, chunk_texts, vectors, kb_id=kb_id,
                parent_texts=parent_texts, chunk_images=chunk_images,
                layout_meta=layout_meta, image_vectors=image_vectors,
                parent_ids=parent_ids, metadata=chunk_metadata_list,
            )
        else:
            doc_id = store.add_document(
                file.filename, chunk_texts, vectors, kb_id=kb_id,
                parent_texts=parent_texts, chunk_images=chunk_images,
                layout_meta=layout_meta, image_vectors=image_vectors,
                parent_ids=parent_ids, metadata=chunk_metadata_list,
            )
    except Exception as e:
        import traceback
        traceback.print_exc()
        os.remove(path)
        raise HTTPException(500, f"文档处理失败: {e}")

    return {"id": doc_id, "name": file.filename, "chunks": len(chunk_texts),
            "kb_id": kb_id, "layout": use_layout}


@app.get("/api/documents")
async def list_documents(page: int = 1, page_size: int | None = None, _admin: dict = Depends(require_role("admin"))):
    return store.list_documents(page=page, page_size=page_size)


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str, _admin: dict = Depends(require_role("admin"))):
    if not store.delete_document(doc_id):
        raise HTTPException(404, "文档不存在")
    return {"ok": True}


class DocMetadataUpdate(BaseModel):
    doc_type: str | None = None
    title: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    tags: str | None = None


@app.get("/api/documents/{doc_id}/metadata")
async def get_doc_metadata(doc_id: str, _admin: dict = Depends(require_role("admin"))):
    return store.get_document_metadata(doc_id)


@app.patch("/api/documents/{doc_id}/metadata")
async def patch_doc_metadata(doc_id: str, body: DocMetadataUpdate, request: Request,
                             _admin: dict = Depends(require_role("admin"))):
    """编辑文档元数据（不重新向量化），保存后立即影响检索过滤/过期判断"""
    patch = body.model_dump(exclude_unset=True)
    n = store.update_metadata(doc_id, patch)
    if not n:
        raise HTTPException(404, "文档不存在")
    try:
        db.write_audit(
            action="kb.metadata.update", resource_type="document", resource_id=doc_id,
            detail=json.dumps({"patch": patch, "chunks": n}, ensure_ascii=False),
            actor_user_id=_admin.get("id"), actor_username=_admin.get("username", ""),
            ip=request.client.host if request.client else "unknown",
        )
    except Exception as e:
        _log.warning("[audit] kb.metadata.update 写入失败: %s", e)
    return {"ok": True, "chunks": n}


@app.get("/api/documents/{doc_id}/chunks")
async def get_chunks(doc_id: str, _admin: dict = Depends(require_role("admin"))):
    """查看文档的所有分块内容"""
    chunks = store.get_document_chunks(doc_id)
    if not chunks:
        raise HTTPException(404, "文档不存在或无分块")
    return chunks


class ChunkUpdate(BaseModel):
    text: str


@app.put("/api/documents/{doc_id}/chunks/{chunk_index}")
async def update_chunk(doc_id: str, chunk_index: int, body: ChunkUpdate, _admin: dict = Depends(require_role("admin"))):
    """编辑某个分块内容，自动重新生成向量"""
    if not store.update_chunk(doc_id, chunk_index, body.text):
        raise HTTPException(404, "分块不存在")
    return {"ok": True}


class UrlImportBody(BaseModel):
    url: str
    kb_id: str = "default"
    chunk_size: int = 0
    chunk_overlap: int = -1
    chunk_mode: str = "flat"


def _html_to_text(html: str) -> str:
    """简单 HTML 正文提取：去 script/style/标签，保留段落结构"""
    html = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?is)<br\s*/?>", "\n", html)
    html = re.sub(r"(?is)</(p|div|li|tr|h[1-6]|section|article)>", "\n\n", html)
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    import html as html_mod
    text = html_mod.unescape(html)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)


@app.post("/api/upload_url")
async def upload_url(body: UrlImportBody, _admin: dict = Depends(require_role("admin"))):
    """从 URL 抓取网页正文并入库"""
    import requests as http
    url = body.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "URL 必须以 http:// 或 https:// 开头")
    try:
        r = http.get(url, timeout=20, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        r.raise_for_status()
    except Exception as e:
        raise HTTPException(502, f"网页抓取失败: {e}")

    text = _html_to_text(r.text)
    if len(text) < 20:
        raise HTTPException(400, "网页正文过短或为空，可能不支持抓取")

    # 用 URL 生成一个伪文件名，便于追踪来源
    from urllib.parse import urlparse
    parsed = urlparse(url)
    base = (parsed.netloc + parsed.path).strip("/").replace("/", "_") or "webpage"
    filename = f"web_{base[:80]}.txt"

    path = os.path.join(config.UPLOAD_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    try:
        cs = body.chunk_size if body.chunk_size > 0 else None
        co = body.chunk_overlap if body.chunk_overlap >= 0 else None
        if getattr(body, "chunk_mode", "flat") == "parent_child":
            tree = chunker.split_parent_child(text, chunk_overlap=co)
            chunks, parent_texts = [], []
            for node in tree:
                for child in node["children"]:
                    chunks.append(child)
                    parent_texts.append(node["parent"])
        else:
            chunks = chunker.split_text(text, chunk_size=cs, chunk_overlap=co)
            parent_texts = None
        if not chunks:
            raise ValueError("未从网页中提取出有效文本")
        vectors = ollama_client.embed_texts(chunks)
        doc_id = store.add_document(filename, chunks, vectors, kb_id=body.kb_id,
                                    parent_texts=parent_texts)
    except Exception as e:
        os.remove(path)
        raise HTTPException(500, f"网页入库失败: {e}")

    return {"id": doc_id, "name": filename, "chunks": len(chunks),
            "kb_id": body.kb_id, "source_url": url}


# ---------- 召回测试 ----------
class RetrievalTestBody(BaseModel):
    query: str
    kb_id: str | None = ""
    top_k: int = 5


@app.post("/api/retrieval/test")
async def retrieval_test(body: RetrievalTestBody, _user: dict = Depends(get_current_user)):
    """输入问题，实时返回命中的分块与得分（用于判断没答好是知识库问题还是模型问题）"""
    if not body.query.strip():
        raise HTTPException(400, "query 不能为空")
    results, elapsed = rag.retrieve_debug(
        body.query.strip(), top_k=max(1, min(body.top_k, 20)),
        kb_id=body.kb_id or None)
    return {"query": body.query, "elapsed_ms": elapsed,
            "total": len(results), "results": results}


# ---------- 知识库分组 ----------
class KbBody(BaseModel):
    name: str
    description: str = ""


@app.post("/api/kb")
async def create_kb(body: KbBody, _admin: dict = Depends(require_role("admin"))):
    return db.create_kb(body.name, body.description)


@app.get("/api/kb")
async def list_kb(page: int = 1, page_size: int | None = None, _admin: dict = Depends(require_role("admin"))):
    return db.list_kb(page=page, page_size=page_size)


@app.delete("/api/kb/{kb_id}")
async def delete_kb(kb_id: str, _admin: dict = Depends(require_role("admin"))):
    db.delete_kb(kb_id)
    return {"ok": True}


# ---------- 会话 ----------
@app.get("/api/sessions")
async def list_sessions(page: int = 1, page_size: int | None = None, _user: dict = Depends(get_current_user)):
    return db.list_sessions(user_id=_user["id"], page=page, page_size=page_size)


@app.post("/api/sessions")
async def create_session(_user: dict = Depends(get_current_user)):
    return {"id": db.create_session(user_id=_user["id"])}


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str, _user: dict = Depends(get_current_user)):
    return db.get_messages(session_id)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, _user: dict = Depends(get_current_user)):
    db.delete_session(session_id)
    return {"ok": True}


# ---------- 设置（提示词 + 敏感词 + 检索开关 + LLM 配置） ----------
class SettingsBody(BaseModel):
    # LLM Provider（统一入口）
    llm_provider: str = "ollama"
    llm_base: str = ""
    llm_api_key: str = ""
    llm_temperature: float = 0.7
    # LLM / Ollama（本地默认）
    ollama_base: str = ""
    chat_model: str = ""
    vision_model: str = ""
    embed_model: str = ""
    vl_embed_model: str = ""
    keep_alive_embed: int = -1
    keep_alive_vl_embed: int = -1
    keep_alive_chat: int = 0
    keep_alive_vision: int = 0
    openai_base: str = ""
    openai_api_key: str = ""
    openai_model: str = ""
    # 提示词 / 安全 / 检索
    kb_prompt: str = ""
    no_match_prompt: str = ""
    direct_prompt: str = ""
    sensitive_words: str = ""
    rerank_enabled: bool = True
    rewrite_enabled: bool = True
    rerank_vector_weight: float = 0.7
    rerank_candidates: int = 20
    hybrid_enabled: bool = False
    hybrid_vector_weight: float = 0.6
    faq_enabled: bool = True
    faq_threshold: float = 0.6
    intent_enabled: bool = True


@app.get("/api/settings")
async def get_settings(_admin: dict = Depends(require_role("admin"))):
    return settings.load_settings()


@app.put("/api/settings")
async def update_settings(body: SettingsBody, _admin: dict = Depends(require_role("admin"))):
    saved = settings.save_settings(body.model_dump())
    # 热更新内存中的 LLM 配置
    try:
        from core import ollama_client as _oc
        if hasattr(_oc, "reload_llm_config"):
            _oc.reload_llm_config()
    except Exception:
        pass
    return saved


@app.post("/api/settings/test_llm")
async def test_llm(body: dict, _admin: dict = Depends(require_role("admin"))):
    """测试 LLM 连通性，不保存任何设置。
    body: {provider, ollama_base, model, llm_base?, llm_api_key?}
    provider: ollama | openai_compat | baidu_ernie | aliyun_dashscope
    """
    from core import llm_provider
    provider_name = body.get("provider") or "ollama"
    model = body.get("model") or config.CHAT_MODEL

    # 组装各 provider 需要的参数
    if provider_name == "ollama":
        base = (body.get("ollama_base") or config.OLLAMA_BASE).rstrip("/")
        api_key = ""
    elif provider_name == "openai_compat":
        base = (body.get("llm_base") or body.get("openai_base") or "https://api.deepseek.com/v1").rstrip("/")
        api_key = body.get("llm_api_key") or body.get("openai_api_key") or ""
    elif provider_name == "baidu_ernie":
        # 百度特殊：body 里同时要有 api_key（Secret Key 字段用 llm_base 传）
        base = body.get("llm_base") or ""  # Secret Key
        api_key = body.get("llm_api_key") or ""  # API Key
    else:
        base = body.get("llm_base") or ""
        api_key = body.get("llm_api_key") or ""

    return llm_provider.test_connectivity(provider_name, base, api_key, model)


# ---------- 数据存储目录 ----------
class DataDirBody(BaseModel):
    path: str


@app.get("/api/settings/data_dir")
async def get_data_dir(_admin: dict = Depends(require_role("admin"))):
    return {"data_dir": config.DATA_DIR,
            "default_dir": os.path.join(config.BASE_DIR, "data")}


@app.post("/api/settings/migrate_data")
async def migrate_data(body: DataDirBody, _admin: dict = Depends(require_role("admin"))):
    """把数据目录整体迁移到新路径：复制数据 → 写指针文件 → 提示重启生效"""
    target = body.path.strip().strip('"')
    if not target:
        raise HTTPException(400, "目标路径不能为空")
    try:
        target = os.path.abspath(target)
    except Exception:
        raise HTTPException(400, "目标路径无效")
    src = os.path.abspath(config.DATA_DIR)
    if os.path.normcase(target) == os.path.normcase(src):
        raise HTTPException(400, "目标路径与当前数据目录相同")
    if os.path.normcase(target).startswith(os.path.normcase(src) + os.sep):
        raise HTTPException(400, "目标路径不能位于当前数据目录内部")
    try:
        os.makedirs(target, exist_ok=True)
    except Exception as e:
        raise HTTPException(400, f"无法创建目标目录: {e}")
    # 可写性校验
    try:
        probe = os.path.join(target, ".write_test")
        with open(probe, "w") as f:
            f.write("ok")
        os.remove(probe)
    except Exception as e:
        raise HTTPException(400, f"目标目录不可写: {e}")
    # 复制全部数据文件（chat.db 为短连接，空闲时复制安全）
    try:
        shutil.copytree(src, target, dirs_exist_ok=True)
    except Exception as e:
        raise HTTPException(500, f"数据复制失败: {e}")
    # 写指针文件，重启后 config.py 读取生效
    with open(os.path.join(config.BASE_DIR, "data_location.txt"), "w", encoding="utf-8") as f:
        f.write(target)
    return {"ok": True, "data_dir": target, "need_restart": True}


# ---------- FAQ 管理 ----------
class FaqBody(BaseModel):
    question: str
    answer: str


@app.get("/api/faq")
async def list_faq(page: int = 1, page_size: int | None = None, _admin: dict = Depends(require_role("admin"))):
    return db.list_faq(page=page, page_size=page_size)


@app.post("/api/faq")
async def add_faq(body: FaqBody, _admin: dict = Depends(require_role("admin"))):
    db.add_faq(body.question, body.answer)
    return {"ok": True}


@app.delete("/api/faq/{item_id}")
async def delete_faq(item_id: int, _admin: dict = Depends(require_role("admin"))):
    db.delete_faq(item_id)
    return {"ok": True}


# ---------- 意图规则管理 ----------
class IntentRuleBody(BaseModel):
    name: str
    keywords: str
    action: str
    reply: str = ""


@app.get("/api/intent-rules")
async def list_intent_rules(page: int = 1, page_size: int | None = None, _admin: dict = Depends(require_role("admin"))):
    return db.list_intent_rules(page=page, page_size=page_size)


@app.post("/api/intent-rules")
async def add_intent_rule(body: IntentRuleBody, _admin: dict = Depends(require_role("admin"))):
    db.add_intent_rule(body.name, body.keywords, body.action, body.reply)
    return {"ok": True}


@app.delete("/api/intent-rules/{rule_id}")
async def delete_intent_rule(rule_id: int, _admin: dict = Depends(require_role("admin"))):
    db.delete_intent_rule(rule_id)
    return {"ok": True}


@app.put("/api/intent-rules/{rule_id}/toggle")
async def toggle_intent_rule(rule_id: int, body: dict, _admin: dict = Depends(require_role("admin"))):
    db.toggle_intent_rule(rule_id, 1 if body.get("enabled") else 0)
    return {"ok": True}


# ---------- 工单管理 ----------
@app.get("/api/tickets")
async def list_tickets(page: int = 1, page_size: int | None = None, _admin: dict = Depends(require_role("admin"))):
    return db.list_tickets(page=page, page_size=page_size)


@app.put("/api/tickets/{ticket_id}/status")
async def update_ticket_status(ticket_id: int, body: dict, _admin: dict = Depends(require_role("admin"))):
    db.update_ticket_status(ticket_id, body.get("status", "open"))
    return {"ok": True}


# ---------- 转人工管理 ----------
@app.get("/api/handoffs")
async def list_handoffs(page: int = 1, page_size: int | None = None, _admin: dict = Depends(require_role("admin"))):
    return db.list_handoffs(page=page, page_size=page_size)


@app.put("/api/handoffs/{handoff_id}/status")
async def update_handoff_status(handoff_id: int, body: dict, _admin: dict = Depends(require_role("admin"))):
    db.update_handoff_status(handoff_id, body.get("status", "pending"))
    return {"ok": True}


# ---------- 未命中问题沉淀 ----------
@app.get("/api/unanswered")
async def list_unanswered(page: int = 1, page_size: int | None = None, _admin: dict = Depends(require_role("admin"))):
    return db.list_unanswered(page=page, page_size=page_size)


@app.delete("/api/unanswered/{item_id}")
async def delete_unanswered(item_id: int, _admin: dict = Depends(require_role("admin"))):
    db.delete_unanswered(item_id)
    return {"ok": True}


# ---------- 用户反馈 ----------
class FeedbackBody(BaseModel):
    session_id: str
    message_id: int
    rating: int  # 1=赞, -1=踩
    comment: str = ""


@app.post("/api/feedback")
async def add_feedback(body: FeedbackBody, _user: dict = Depends(get_current_user)):
    db.add_feedback(body.session_id, body.message_id, body.rating, body.comment)
    return {"ok": True}


@app.get("/api/feedback")
async def list_feedback(_admin: dict = Depends(require_role("admin"))):
    return db.list_feedback()


# ---------- 统计看板 ----------
@app.get("/api/stats")
async def get_stats(_admin: dict = Depends(require_role("admin"))):
    return db.get_stats()


@app.get("/api/stats/trend")
async def get_trend(days: int = 14, _admin: dict = Depends(require_role("admin"))):
    """看板图表数据：每日问答量、每日反馈、问答来源分布、耗时统计"""
    return {
        "daily_qa": db.get_daily_qa(days),
        "daily_feedback": db.get_daily_feedback(days),
        "qa_source": db.get_qa_source_distribution(),
        "latency": db.get_latency_stats(days),
    }


# ---------- 问答日志 ----------
@app.get("/api/qa_log")
async def get_qa_log(session_id: str = "", limit: int = 100,
                     page: int = 1, page_size: int | None = None,
                     _admin: dict = Depends(require_role("admin"))):
    return db.list_qa_log(session_id, limit, page=page, page_size=page_size)


@app.get("/api/qa_log/export")
async def export_badcases(_admin: dict = Depends(require_role("admin"))):
    """导出点踩或无命中的问答对（JSONL）"""
    rows = db.export_badcases()
    lines = []
    for r in rows:
        hits = json.loads(r.get("hits_json") or "[]")
        lines.append(json.dumps({
            "question": r["question"],
            "answer": r["answer"],
            "session_id": r["session_id"],
            "message_id": r["message_id"],
            "rating": r.get("rating"),
            "no_match": bool(r["no_match"]),
            "used_faq": bool(r["used_faq"]),
            "used_tool": bool(r["used_tool"]),
            "elapsed_ms": r["elapsed_ms"],
            "hits": hits,
            "created_at": r["created_at"],
        }, ensure_ascii=False))
    return StreamingResponse(
        iter(["\n".join(lines)]),
        media_type="application/jsonl",
        headers={"Content-Disposition": "attachment; filename=badcases.jsonl"},
    )


# ---------- 对话导出（Markdown） ----------
@app.get("/api/sessions/{session_id}/export")
async def export_session(session_id: str, _admin: dict = Depends(require_role("admin"))):
    """导出会话为 Markdown，图片以占位符标注（base64 太长不入文档）"""
    msgs = db.get_messages(session_id)
    if not msgs:
        raise HTTPException(404, "会话不存在或无消息")
    title = next((s["title"] for s in db.list_sessions() if s["id"] == session_id), session_id)
    lines = [f"# {title}\n",
             f"导出时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n\n---\n"]
    for m in msgs:
        role = "👤 用户" if m["role"] == "user" else "🤖 助手"
        lines.append(f"\n## {role}\n")
        if m.get("images"):
            lines.append(f"*[含 {len(m['images'])} 张图片]*\n")
        lines.append(m["content"] + "\n")
    md = "".join(lines)
    return StreamingResponse(
        iter([md]), media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f"attachment; filename=chat_{session_id}.md"},
    )


# ---------- 评测集（标注 + 回归测试） ----------
class EvalCaseBody(BaseModel):
    question: str
    expected: str
    kb_id: str = ""


@app.get("/api/eval/cases")
async def list_eval_cases(page: int = 1, page_size: int | None = None, _admin: dict = Depends(require_role("admin"))):
    return db.list_eval_cases(page=page, page_size=page_size)


@app.post("/api/eval/cases")
async def add_eval_case(body: EvalCaseBody, _admin: dict = Depends(require_role("admin"))):
    return {"id": db.add_eval_case(body.question, body.expected, "manual", body.kb_id)}


@app.post("/api/eval/cases/from_log")
async def add_eval_case_from_log(body: dict, _admin: dict = Depends(require_role("admin"))):
    """从问答日志/badcase 一键加入评测集"""
    return {"id": db.add_eval_case(body["question"], body.get("expected", body.get("answer", "")),
                                   "badcase", body.get("kb_id", ""))}


@app.put("/api/eval/cases/{case_id}")
async def update_eval_case(case_id: int, body: EvalCaseBody, _admin: dict = Depends(require_role("admin"))):
    db.update_eval_case(case_id, body.question, body.expected)
    return {"ok": True}


@app.delete("/api/eval/cases/{case_id}")
async def delete_eval_case(case_id: int, _admin: dict = Depends(require_role("admin"))):
    db.delete_eval_case(case_id)
    return {"ok": True}


@app.post("/api/eval/run")
async def run_eval(_admin: dict = Depends(require_role("admin"))):
    """回归测试：对每条评测用例跑一遍检索+生成，计算与期望答案的相似度
    返回每条用例的得分与平均分（语义相似度，0~1）"""
    import numpy as np
    cases = db.list_eval_cases()
    if not cases:
        raise HTTPException(400, "评测集为空，请先添加用例")
    results = []
    for c in cases:
        # 走与线上一致的检索管线
        context, sources, _imgs = rag.retrieve(c["question"], kb_id=c["kb_id"] or None)
        cfg = settings.load_settings()
        system = cfg["kb_prompt"].format(context=context) if context else cfg["no_match_prompt"]
        try:
            answer = ollama_client.chat_once(
                [{"role": "system", "content": system},
                 {"role": "user", "content": c["question"]}],
                timeout=120)
        except Exception as e:
            results.append({"id": c["id"], "question": c["question"], "expected": c["expected"],
                            "answer": f"[生成失败: {e}]", "score": 0, "hits": len(sources)})
            continue
        # 语义相似度：生成答案与期望答案各算向量求余弦
        try:
            vecs = ollama_client.embed_texts([answer, c["expected"]])
            score = float(np.dot(vecs[0], vecs[1]))
        except Exception:
            score = 0.0
        results.append({"id": c["id"], "question": c["question"], "expected": c["expected"],
                        "answer": answer, "score": round(score, 3), "hits": len(sources)})
    avg = round(sum(r["score"] for r in results) / len(results), 3)
    return {"avg_score": avg, "total": len(results), "results": results}


# ---------- 知识库健康度 ----------
@app.get("/api/kb/health")
async def kb_health(_admin: dict = Depends(require_role("admin"))):
    """每个文档的命中次数、平均分、最近命中时间；未命中过的文档标红"""
    doc_stats = db.get_kb_health()
    docs = store.list_documents()
    out = []
    for d in docs:
        st = doc_stats.get(d["name"])
        out.append({
            "doc_id": d["id"], "name": d["name"], "kb_id": d.get("kb_id", "default"),
            "chunks": d["chunks"],
            "hits": st["hits"] if st else 0,
            "avg_score": round(st["score_sum"] / st["hits"], 3) if st and st["hits"] else 0,
            "last_hit": time.strftime("%Y-%m-%d %H:%M", time.localtime(st["last_hit"])) if st else "从未命中",
            "never_hit": st is None,
        })
    out.sort(key=lambda x: x["hits"], reverse=True)
    return out


@app.post("/api/kb/preview")
async def kb_preview(
    file: UploadFile | None = File(None),
    text: str | None = Form(None),
    chunk_config: str | None = Form(None),
    preprocess_rules: str | None = Form(None),
    _admin: dict = Depends(require_role("admin")),
):
    """前端上传弹窗实时预览（统一 multipart/form-data）：
    - 纯文本：前端传 text=纯文本 字符串
    - 二进制：前端传 file=文件，后端用 parse_file 解
    """
    from core.parser import preview_chunks, parse_file

    cfg = {}
    rules = {}
    if chunk_config:
        try: cfg = json.loads(chunk_config)
        except: pass
    if preprocess_rules:
        try: rules = json.loads(preprocess_rules)
        except: pass

    if file is not None:
        raw = await file.read()
        import tempfile, os
        ext = os.path.splitext(file.filename or "")[1].lower()
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tf:
                tf.write(raw)
                tmp_path = tf.name
            parsed = parse_file(tmp_path)
            text_content = parsed.get("text", "") if isinstance(parsed, dict) else str(parsed or "")
            os.unlink(tmp_path)
        except Exception as e:
            raise HTTPException(400, f"预览解析失败: {e}")
    elif text:
        text_content = text
    else:
        raise HTTPException(400, "需要传 text 或 file")

    return preview_chunks(text_content, cfg, rules)


# ---------- 图片知识库 ----------
@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...), kb_id: str = Form("default"), _admin: dict = Depends(require_role("admin"))):
    """上传图片入知识库：视觉模型生成文字描述 → 描述向量化入库 → 检索命中时带出原图"""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in config.ALLOWED_IMG_EXT:
        raise HTTPException(400, f"不支持的图片类型 {ext}")
    import base64
    raw = await file.read()
    b64 = base64.b64encode(raw).decode()
    # 保存原图
    img_name = f"{int(time.time() * 1000)}{ext}"
    with open(os.path.join(config.KB_IMG_DIR, img_name), "wb") as f:
        f.write(raw)
    try:
        # 视觉模型生成检索用描述
        desc = ollama_client.chat_once(
            [{"role": "user", "content": "请详细描述这张图片的内容（100字以内），用于知识库检索。"}],
            model=config.VISION_MODEL, images=[b64], timeout=180)
        if not desc:
            raise ValueError("视觉模型未返回描述")
        chunk_text = f"[图片:{file.filename}] {desc}"
        vectors = ollama_client.embed_texts([chunk_text])
        doc_id = store.add_document(file.filename, [chunk_text], vectors,
                                    kb_id=kb_id, chunk_images=[img_name])
    except Exception as e:
        os.remove(os.path.join(config.KB_IMG_DIR, img_name))
        raise HTTPException(500, f"图片入库失败: {e}")
    return {"id": doc_id, "name": file.filename, "desc": desc, "kb_id": kb_id}


@app.get("/api/kb_image/{img_name}")
async def get_kb_image(img_name: str, _admin: dict = Depends(require_role("admin"))):
    """获取知识库图片（前端预览用）"""
    # 防路径穿越
    safe = os.path.basename(img_name)
    path = os.path.join(config.KB_IMG_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(404, "图片不存在")
    return FileResponse(path)


# ---------- 记忆管理 ----------
@app.get("/api/memories")
async def list_memories(
    session_id: str = "", keyword: str = "", kind: str = "",
    page: int = 1, page_size: int | None = None,
    _admin: dict = Depends(require_role("admin"))
):
    return db.list_memories(
        session_id=session_id or None,
        keyword=keyword or None,
        kind=kind or None,
        page=page, page_size=page_size,
    )


@app.get("/api/memories/{mem_id}")
async def get_memory(mem_id: int, _admin: dict = Depends(require_role("admin"))):
    rows = db.list_memories()
    for m in rows:
        if m["id"] == mem_id:
            return m
    raise HTTPException(404, "记忆不存在")


@app.delete("/api/memories/{mem_id}")
async def delete_memory(mem_id: int, _admin: dict = Depends(require_role("admin"))):
    db.delete_memory(mem_id)
    return {"ok": True}


# ---------- Agent 管理 ----------
@app.get("/api/agents")
async def list_agents(page: int = 1, page_size: int | None = None, _admin: dict = Depends(require_role("admin"))):
    return db.list_agents(page=page, page_size=page_size)


@app.post("/api/agents")
async def create_agent(req: dict, _admin: dict = Depends(require_role("admin"))):
    agent_id = db.create_agent(
        name=req.get("name", "").strip(),
        description=req.get("description", ""),
        endpoint=req.get("endpoint", ""),
        auth_type=req.get("auth_type", "none"),
        auth_value=req.get("auth_value", ""),
        schema=req.get("schema_json", {}),
    )
    if not agent_id:
        raise HTTPException(400, "Agent 创建失败")
    return {"ok": True, "id": agent_id}


@app.put("/api/agents/{agent_id}")
async def update_agent(agent_id: int, req: dict, _admin: dict = Depends(require_role("admin"))):
    db.update_agent(
        agent_id,
        name=req.get("name"),
        description=req.get("description"),
        endpoint=req.get("endpoint"),
        auth_type=req.get("auth_type"),
        auth_value=req.get("auth_value"),
        schema=req.get("schema_json"),
    )
    return {"ok": True}


@app.post("/api/agents/{agent_id}/toggle")
async def toggle_agent(agent_id: int, enabled: bool = True, _admin: dict = Depends(require_role("admin"))):
    db.toggle_agent(agent_id, enabled)
    return {"ok": True}


@app.delete("/api/agents/{agent_id}")
async def delete_agent(agent_id: int, _admin: dict = Depends(require_role("admin"))):
    db.delete_agent(agent_id)
    return {"ok": True}


# ---------- 聊天（SSE 流式） ----------
class ChatRequest(BaseModel):
    session_id: str | int  # 前端可能传数字，统一转字符串
    message: str
    model: str = ""      # 为空则用默认模型
    use_kb: bool = True  # 是否检索知识库（关闭则直接与大模型对话）
    kb_id: str | None = ""  # 指定知识库，空则全部
    use_tools: bool = True  # 是否启用 Agent 工具调用
    enabled_tools: list[str] | None = None  # 允许使用的工具名列表，None 表示全部可用
    images: list[str] = []  # base64 编码图片（多模态），有图时自动切换视觉模型

    @field_validator("session_id", mode="before")
    @classmethod
    def _session_id_to_str(cls, v):
        return str(v)


@app.get("/api/tools")
async def list_tools(_admin: dict = Depends(require_role("admin"))):
    """返回可调用的 Agent 工具清单（供前端工具选择面板展示）"""
    return [{"name": name,
             "label": spec.get("label", name),
             "description": spec["description"]}
            for name, spec in tools.TOOLS.items()]


@app.get("/api/providers")
async def list_providers():
    """返回可用 LLM Provider 列表（前端设置页下拉用）"""
    from core.llm_provider import available_providers
    return available_providers()


# 名称命中这些关键字的一律视为向量/重排模型，不在对话模型下拉框中出现
_NON_CHAT_MODEL_KEYS = ("embed", "bge", "rerank", "minilm", "e5-", "gte-", "nomic")


class SynonymItem(BaseModel):
    key: str = ""
    synonyms: list[str] = []


def _synonyms_path() -> str:
    return os.path.join(config.DATA_DIR, "synonyms.json")


@app.get("/api/synonyms")
async def get_synonyms(_admin: dict = Depends(require_role("admin"))):
    """读取同义词词典（key=标准词, value=同义词列表）"""
    path = _synonyms_path()
    items = []
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            items = [{"key": k, "synonyms": v} for k, v in raw.items()
                     if not k.startswith("_") and isinstance(v, list)]
    except Exception as e:
        _log.warning("[synonyms] 读取失败: %s", e)
    return {"items": items}


@app.put("/api/synonyms")
async def save_synonyms(body: list[SynonymItem], request: Request,
                        _admin: dict = Depends(require_role("admin"))):
    """整体保存同义词词典；写入后 rag 侧按 mtime 自动热加载，立即生效"""
    data = {"_comment": "key=标准词, value=同义词数组。检索前命中 key 会扩展出多条查询，改完立即生效。"}
    for it in body:
        k = (it.key or "").strip()
        if not k:
            continue
        vals = []
        for s in it.synonyms:
            s = (s or "").strip()
            if s and s not in vals:
                vals.append(s)
        if vals:
            data[k] = vals

    path = _synonyms_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

    try:
        db.write_audit(
            action="kb.synonyms.update", resource_type="config", resource_id="synonyms",
            detail=json.dumps({"count": len(data) - 1}, ensure_ascii=False),
            actor_user_id=_admin.get("id"), actor_username=_admin.get("username", ""),
            ip=request.client.host if request.client else "unknown",
        )
    except Exception as e:
        _log.warning("[audit] kb.synonyms.update 写入失败: %s", e)
    return {"ok": True, "count": len(data) - 1}


@app.get("/api/models")
async def list_models():
    """读取 Ollama 已安装模型（过滤掉 embedding 模型）"""
    import requests as http

    try:
        r = http.get(f"{config.OLLAMA_BASE}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception:
        raise HTTPException(502, "无法连接 Ollama，请确认服务已启动")
    models = []
    for m in r.json().get("models", []):
        name = m.get("name", "")
        caps = m.get("capabilities") or []
        # 双重过滤：能力标记 + 名称关键字，避免自定义 GGUF 向量模型漏进对话列表
        if "embedding" in caps:
            continue
        if any(k in name.lower() for k in _NON_CHAT_MODEL_KEYS):
            continue
        models.append(name)
    return models


@app.get("/api/embed-models")
async def list_embed_models(_admin: dict = Depends(require_role("admin"))):
    """读取 Ollama 已安装的 embedding 模型列表（上传知识库用）"""
    import requests as http

    try:
        r = http.get(f"{config.OLLAMA_BASE}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception:
        raise HTTPException(502, "无法连接 Ollama，请确认服务已启动")
    models = []
    for m in r.json().get("models", []):
        name = m.get("name", "")
        caps = m.get("capabilities") or []
        is_embed = "embedding" in caps or any(k in name.lower() for k in _NON_CHAT_MODEL_KEYS)
        if not is_embed:
            continue
        dim = m.get("details", {}).get("embedding_length", 0)
        models.append({"name": name, "dim": dim})
    # 标记默认
    for m in models:
        m["default"] = (m["name"] == config.EMBED_MODEL)
    return models


# 敏感词命中时的固定回复
SENSITIVE_REPLY = "抱歉，您的问题包含敏感内容，无法回答。请换个问题试试。"


def _is_degenerate_repetition(text: str) -> bool:
    """检测小模型退化：短片段在尾部高频重复（如 principles principles…）"""
    if len(text) < 240:
        return False
    tail = text[-800:]
    for unit in (4, 6, 8, 12, 16):
        if len(tail) < unit * 12:
            continue
        probe = tail[-unit:]
        if not probe.strip():
            continue
        cnt = tail.count(probe)
        # 重复片段需同时满足"次数够多"和"占比过半"，避免正常文本误判
        if cnt >= 12 and cnt * unit >= len(tail) * 0.5:
            return True
    return False


def _fixed_response(text):
    """固定回复的 SSE 流"""
    yield f"data: {json.dumps({'type': 'token', 'content': text}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'type': 'done', 'sources': []}, ensure_ascii=False)}\n\n"


def _parse_tool_call(text):
    """从模型输出中解析工具调用 JSON，返回 (name, args) 或 None"""
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "tool" in data:
            return data["tool"], data.get("args", {})
    except Exception:
        pass
    # 兜底：从文本里截取 JSON 片段。args 可能嵌套花括号，
    # 所以按花括号配对扫描，而不是用 [^}]* 这种无法处理嵌套的正则
    start = text.find('{"tool"')
    if start < 0:
        start = text.find('{ "tool"')
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[start:i + 1])
                    if "tool" in data:
                        return data["tool"], data.get("args", {})
                except Exception:
                    pass
                break
    return None


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request, _user: dict = Depends(get_current_user)):
    question = req.message.strip()
    if not question:
        raise HTTPException(400, "消息不能为空")

    _ip = request.client.host if request.client else "unknown"

    import uuid as _uuid
    trace_id = _uuid.uuid4().hex[:8]
    trace_start = time.time()
    stages = []  # 各环节记录

    def _stage(name, status, input_data="", output_data="", error="", detail=None):
        """记录一个环节的执行情况

        detail: 可选的完整详情（不截断），用于前端展开看完整内容
        """
        stages.append({
            "name": name,
            "elapsed_ms": int((time.time() - trace_start) * 1000),
            "status": status,  # pass | hit | skip | error
            "input": str(input_data)[:200] if input_data else "",
            "output": str(output_data)[:200] if output_data else "",
            "error": str(error)[:500] if error else "",
            "detail": detail,  # 完整详情（chunk 列表、完整 prompt 等）
        })

    history = db.get_messages(req.session_id, limit=30)
    if not history:  # 首条消息：用问题做会话标题
        db.rename_session(req.session_id, question[:20])
    # 图片持久化：user 消息同时存文字和图片，重启后可还原显示
    db.add_message(req.session_id, "user", question, images=req.images or None)
    # 发给模型的历史需去掉 images 字段（文本模型不接受该字段）
    history = [{"role": m["role"], "content": m["content"]} for m in history]

    # ⓪ 问候/闲聊快速拦截：纯问候短句不走 RAG，直接回复预设（避免向量噪声把"你好"匹配成退货政策）
    if intent.is_greeting(question):
        _stage("问候拦截", "hit", question, "预设话术")
        reply = intent.GREETING_REPLY
        msg_id = db.add_message(req.session_id, "assistant", reply, sources=[])
        db.add_qa_log(req.session_id, msg_id, question, reply,
                      [], 0, used_faq=False, used_tool=False, no_match=False)
        total_ms = int((time.time() - trace_start) * 1000)
        db.add_trace(trace_id, req.session_id, question, reply, total_ms, stages, error="")
        return StreamingResponse(_fixed_response(reply), media_type="text/event-stream")
    _stage("问候拦截", "pass")

    # ① 输入敏感词拦截：不调模型，直接回复
    if settings.contains_sensitive(question):
        _stage("敏感词检测", "hit", question, "命中敏感词")
        msg_id = db.add_message(req.session_id, "assistant", SENSITIVE_REPLY, sources=[])
        db.add_qa_log(req.session_id, msg_id, question, SENSITIVE_REPLY,
                      [], 0, used_faq=False, used_tool=False, no_match=False)
        total_ms = int((time.time() - trace_start) * 1000)
        db.add_trace(trace_id, req.session_id, question, SENSITIVE_REPLY, total_ms, stages, error="")
        return StreamingResponse(_fixed_response(SENSITIVE_REPLY), media_type="text/event-stream")
    _stage("敏感词检测", "pass", question)

    # ② FAQ 快捷回复：词面相似度命中，直接回复预设答案
    faq_hit = intent.match_faq(question)
    if faq_hit:
        answer, score = faq_hit
        _stage("FAQ匹配", "hit", question, answer[:50])
        msg_id = db.add_message(req.session_id, "assistant", answer, sources=[{"doc": "FAQ", "score": round(score, 3)}])
        db.add_qa_log(req.session_id, msg_id, question, answer,
                      [{"doc": "FAQ", "score": round(score, 3)}], 0,
                      used_faq=True, used_tool=False, no_match=False)
        total_ms = int((time.time() - trace_start) * 1000)
        db.add_trace(trace_id, req.session_id, question, answer, total_ms, stages, error="")
        return StreamingResponse(_fixed_response(answer), media_type="text/event-stream")
    _stage("FAQ匹配", "pass", question)

    # ③ 意图识别：仅跑关键词规则（精确命中才直接截断）
    #    exemplar 向量路由推迟到 RAG 之后作为兜底（RAG 也没召回到才触发）
    rule = intent.match_intent_rule_only(question)
    if rule:
        action = rule["action"]
        reply = rule["reply"] or "已收到您的请求，我们会尽快处理。"
        if action == "handoff":
            db.add_handoff(req.session_id, question)
        elif action == "ticket":
            db.add_ticket(req.session_id, question, rule["name"])
        _stage("意图识别", "hit", question, f"命中规则:{rule['name']}")
        msg_id = db.add_message(req.session_id, "assistant", reply, sources=[{"doc": f"意图规则:{rule['name']}", "score": 1.0}])
        db.add_qa_log(req.session_id, msg_id, question, reply,
                      [{"doc": f"意图规则:{rule['name']}", "score": 1.0}], 0,
                      used_faq=False, used_tool=False, no_match=False)
        total_ms = int((time.time() - trace_start) * 1000)
        db.add_trace(trace_id, req.session_id, question, reply, total_ms, stages, error="")
        return StreamingResponse(_fixed_response(reply), media_type="text/event-stream")
    _stage("意图识别", "pass", question, "规则未命中，让 RAG 先跑")

    import time as _time

    def event_stream():
        answer = ""
        chat_start = _time.time()
        gen_start = None
        try:
            # 阶段提示①：检索/工具探测耗时较长，必须放在流内发送，前端才能看到
            yield f"data: {json.dumps({'type': 'stage', 'content': '🔍 检索知识库…' if req.use_kb else '🤔 思考中…', 'elapsed_ms': int((_time.time() - chat_start) * 1000)}, ensure_ascii=False)}\n\n"

            # ④ 长期记忆：摘要压缩 + 跨会话记忆注入
            hist = memory.maybe_summarize(req.session_id, history, user_id=_user["id"])
            memory_ctx = memory.build_memory_context(user_id=_user["id"])
            _stage("记忆处理", "pass", "", f"注入{len(memory_ctx)}字" if memory_ctx else "无")

            # ⑤ 知识库检索 + 大模型生成
            messages, sources, matched, kb_images = rag.build_messages(
                question, hist, req.use_kb, req.model or None,
                kb_id=req.kb_id or None,
            )
            # 完整记录知识库检索详情（chunk 列表 + 分数 + 文档名 + 对应资料编号）
            # rag.retrieve 返回: text=原始chunk, parent=去重后的父块(LLM实际收到的), ref_num=[资料N]
            _kb_detail = [
                {
                    "doc": s.get("doc", ""),
                    "score": s.get("score", 0),
                    "chunk_index": s.get("chunk_index", -1),
                    "ref_num": s.get("ref_num", -1),
                    "chunk_text": (s.get("parent") or s.get("text") or "")[:300],
                }
                for s in (sources or [])
            ] if sources else None
            _stage("知识库检索", "pass", question,
                   f"命中{len(sources) if sources else 0}条" + (f"，最高分 {max((s.get('score',0) for s in sources), default=0):.3f}" if sources else ""),
                   detail=_kb_detail)

            # ★ 向量检索命中审计（检索后立即写，保证即使生成被打断也能记录）
            try:
                db.write_audit(
                    action="rag.retrieve",
                    resource_type="session", resource_id=str(req.session_id) if req.session_id else "",
                    result="success",
                    detail=json.dumps({
                        "question": question[:200],
                        "matched": bool(matched),
                        "sources": _kb_detail[:10] if _kb_detail else [],
                    }, ensure_ascii=False),
                    actor_user_id=_user.get("id"),
                    actor_username=_user.get("username", ""),
                    ip=_ip,
                )
            except Exception as _audit_err:
                _log.warning("[audit] 写入失败: %s", _audit_err)

            # ⑤-补：RAG 没召回到 → exemplar 向量路由兜底
            #    只有规则没命中 + RAG 没召回 + exemplar 触发，才走这个分支
            #    （解决 bge-m3 语义空间噪声导致 exemplar 在有资料时误截胡的问题）
            if not matched:
                try:
                    ex_rule = intent._match_exemplar(question)
                    if ex_rule:
                        _stage("意图识别(exemplar兜底)", "hit", question,
                               f"{ex_rule['name']} score={ex_rule.get('score', '?')}")
                        reply = ex_rule["reply"] or "已收到您的请求。"
                        yield f"data: {json.dumps({'type': 'token', 'content': reply}, ensure_ascii=False)}\n\n"
                        total_ms = int((_time.time() - chat_start) * 1000)
                        msg_id = db.add_message(req.session_id, "assistant", reply, sources=[{"doc": f"意图规则:{ex_rule['name']}", "score": ex_rule.get("score", 1.0)}])
                        db.add_qa_log(req.session_id, msg_id, question, reply,
                                      [{"doc": f"意图规则:{ex_rule['name']}", "score": ex_rule.get("score", 1.0)}], 0,
                                      used_faq=False, used_tool=False, no_match=True)
                        db.add_trace(trace_id, req.session_id, question, reply, total_ms, stages, error="")
                        yield f"data: {json.dumps({'type': 'done', 'sources': [], 'total_ms': total_ms, 'intent_exemplar': ex_rule['name']}, ensure_ascii=False)}\n\n"
                        return  # 生成器提前结束，不走后面的大模型生成
                except Exception as _ex:
                    print(f"[chat] exemplar 兜底失败，继续走常规流程: {_ex}")

            # 注入跨会话记忆到 system prompt
            if memory_ctx:
                messages[0]["content"] += memory_ctx

            # 图片知识库处理：
            #   用户自己上传了图（req.images）→ 走视觉模型，图片喂给 chat_stream
            #   RAG 召回到了 kb_images → 只在最终 sources 里带给前端渲染，
            #     不喂给大模型（避免触发视觉模型导致 context 超限/模型切换）
            all_images = list(req.images or [])

            # ⑥ 未命中沉淀：开启知识库但没检索到资料时，记录问题便于后续补充
            if req.use_kb and not matched:
                db.add_unanswered(question, req.session_id)

            # ⑦ Agent 工具调用，两级调度：
            #    ① 关键词直路由：system_info / gpu_info / get_current_time 这类零参数工具，
            #       不需要模型抽取参数，直接确定性执行，不依赖小模型判断（更快更稳）
            #    ② 模型探测：calculator / get_weather 需要从问题里抽取参数，才交给模型
            #    含图片时整体跳过（视觉模型的工具调用不稳定）
            tool_result = None
            tool_name = None
            tool_args = None
            # 用户在前端勾选的工具白名单：None 表示全部可用
            allowed = set(req.enabled_tools) if req.enabled_tools is not None else None

            def tool_allowed(name):
                return allowed is None or name in allowed

            if req.use_tools and not req.images:
                direct = tools.route_tool(question)
                if direct and not tool_allowed(direct):
                    _stage("Agent工具", "skip", question, f"{direct} 未在前端启用")
                    direct = None
                if direct:
                    tool_name, tool_args = direct, {}
                    tool_result = tools.execute_tool(direct, {})
                    _stage("Agent工具", "hit", question, f"{direct}（关键词直路由）")
                elif tools.should_probe(question):
                    messages[0]["content"] += tools.SYSTEM_PROMPT_SUFFIX
                    try:
                        # 工具探测用轻量小模型（PROBE_MODEL），比用大模型快一个数量级
                        probe = ollama_client.chat_once(
                            messages, config.PROBE_MODEL, timeout=120)
                        tool_call = _parse_tool_call(probe)
                        if tool_call:
                            tool_name, tool_args = tool_call
                            if not tool_allowed(tool_name):
                                _stage("Agent工具", "skip", question,
                                       f"{tool_name} 未在前端启用")
                                tool_name, tool_args = None, None
                            else:
                                tool_result = tools.execute_tool(tool_name, tool_args)
                                messages.append({"role": "assistant", "content": probe})
                                messages.append({"role": "user", "content": f"工具结果如下，请基于它回答用户问题：\n{tool_result}"})
                                _stage("Agent工具", "pass", question,
                                       f"{tool_name}({json.dumps(tool_args, ensure_ascii=False)})")
                        else:
                            # 把探测原始输出记进链路，便于排查"为什么没调工具"
                            _stage("Agent工具", "skip", question,
                                   f"未调用工具，探测输出: {probe[:120]}")
                    except Exception as e:
                        _stage("Agent工具", "error", question, "", str(e))
                        print(f"[chat] 工具调用探测失败，直接生成: {e}")
                else:
                    _stage("Agent工具", "skip", question, "无匹配工具，跳过")
            elif req.images:
                _stage("Agent工具", "skip", "", "含图片，跳过")
            else:
                _stage("Agent工具", "skip", "", "工具已关闭")

            # 工具执行结果注入对话，让模型基于真实数据作答
            if tool_result:
                messages.append({
                    "role": "user",
                    "content": f"工具 {tool_name} 的执行结果如下，请严格基于这些真实数据回答用户问题：\n{tool_result}",
                })
                # 把工具调用过程推给前端（结构化的工具名与参数）
                yield f"data: {json.dumps({'type': 'tool', 'tool': tool_name, 'args': tool_args}, ensure_ascii=False)}\n\n"
            # 阶段提示②：开始流式生成
            gen_start = _time.time()
            yield f"data: {json.dumps({'type': 'stage', 'content': '✍️ 正在生成回答…', 'elapsed_ms': int((_time.time() - chat_start) * 1000)}, ensure_ascii=False)}\n\n"

            degenerate = False
            for token in ollama_client.chat_stream(messages, req.model or None, images=all_images or None):
                answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
                if _is_degenerate_repetition(answer):
                    degenerate = True
                    break
            if degenerate:
                _stage("生成回答", "error", question, "检测到重复循环，已中断",
                       detail={"answer_head": answer[:200]})
                answer = ("抱歉，本次回答生成异常（出现重复循环），已自动中断。"
                          "请重试，或把问题描述得更具体一些。")
                yield f"data: {json.dumps({'type': 'replace', 'content': answer}, ensure_ascii=False)}\n\n"
            # ⑧ 输出敏感词检测：命中则整段替换为安全回复
            if settings.contains_sensitive(answer):
                answer = SENSITIVE_REPLY
                yield f"data: {json.dumps({'type': 'replace', 'content': answer}, ensure_ascii=False)}\n\n"

            _stage("生成回答", "pass", question, answer[:50],
                   detail={
                       "prompt_messages": [
                           {"role": m.get("role", ""), "content": str(m.get("content", ""))[:500]}
                           for m in (messages or [])
                       ],
                       "final_answer": answer,
                       "answer_length": len(answer),
                       "sources_count": len(sources or []),
                       "elapsed_ms": int((_time.time() - gen_start) * 1000) if gen_start else 0,
                       "model": req.model or config.CHAT_MODEL,
                   })

            msg_id = db.add_message(req.session_id, "assistant", answer, sources=sources if sources else [])
            # ⑨ 长期记忆：从本轮问答提取可记忆信息
            try:
                memory.extract_and_save(req.session_id, question, answer, user_id=_user["id"])
            except Exception as e:
                print(f"[chat] 记忆提取失败: {e}")
            # ⑩ 记录问答日志（含命中分块、耗时、是否走 FAQ/工具）
            total_ms = int((_time.time() - chat_start) * 1000)
            try:
                db.add_qa_log(req.session_id, msg_id, question, answer,
                              sources, total_ms,
                              used_faq=False, used_tool=bool(tool_result),
                              no_match=req.use_kb and not matched)
            except Exception as e:
                print(f"[chat] 问答日志记录失败: {e}")
            # 记录链路追踪
            db.add_trace(trace_id, req.session_id, question, answer, total_ms, stages, error="")

            # ★ 提前写检索审计（已在 L1598 处执行）

            # ★ 新增：回答质量评估（词面版快速评估，零延迟写入 audit log）
            try:
                from core.ragas_eval import evaluate_sample
                contexts = [s.get("parent") or s.get("text", "") for s in (sources or [])]
                ev = evaluate_sample(question, answer, contexts, latency_ms=total_ms)
                db.write_audit(
                    action="rag.answer_eval",
                    resource_type="session", resource_id=req.session_id,
                    result="success",
                    detail=json.dumps({
                        "faithfulness": ev.faithfulness,
                        "answer_relevance": ev.answer_relevance,
                        "context_precision": ev.context_precision,
                        "latency_ms": total_ms,
                    }, ensure_ascii=False),
                    actor_user_id=_user.get("id"),
                    actor_username=_user.get("username", ""),
                    ip=_ip,
                )
            except Exception as _audit_err:
                _log.warning("[audit] 写入失败: %s", _audit_err)
            # ⑪ 无匹配提示转人工：
            #    - 开启 KB 但没召回到有效资料（matched=False 或最高分 < 0.3 低质量片段）
            #    - 模型回复包含 no_match 关键词（说明资料确实不相关）
            done_payload = {"type": "done", "sources": sources, "total_ms": total_ms, "trace_id": trace_id}
            _max_score = max((s.get("score", 0) for s in (sources or [])), default=0.0)
            _no_match_kws = ("未提及", "暂无资料", "没有相关", "未找到", "无法找到",
                             "未收录", "暂未收录", "知识库中没有", "补充资料")
            _reply_indicates_no_match = any(k in answer for k in _no_match_kws)
            _low_score = _max_score < 0.3
            if req.use_kb and (_low_score or not matched or _reply_indicates_no_match):
                done_payload["handoff_suggest"] = True
                done_payload["handoff_message"] = (
                    "抱歉，我在知识库中没有找到足够相关的资料来回答这个问题。"
                    "是否需要我帮您联系人工客服？"
                    "（点击下方按钮即可转人工处理）"
                )
            yield f"data: {json.dumps(done_payload, ensure_ascii=False)}\n\n"
        except Exception as e:
            total_ms = int((_time.time() - chat_start) * 1000) if chat_start else 0
            db.add_trace(trace_id, req.session_id, question, answer, total_ms, stages, error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------- 转人工（用户确认后触发）----------
class HandoffBody(BaseModel):
    session_id: str
    question: str = ""          # 用户触发转人工的原始问题（可选）
    reason: str = "用户主动要求"  # 转单原因
    user_agent: str = ""


@app.post("/api/handoff")
async def confirm_handoff(body: HandoffBody, _user: dict = Depends(get_current_user)):
    """用户确认转人工后调用：创建转单记录 + 写审计日志 + 通知管理员"""
    # 1) 创建转单
    handoff_id = db.add_handoff(body.session_id, body.question or "(用户确认转人工)")
    # 2) 写审计日志
    try:
        db.write_audit(
            actor_username=_user["username"],
            actor_user_id=_user["id"],
            action="chat.handoff.confirm",
            resource_type="session",
            resource_id=body.session_id,
            ip="", user_agent=body.user_agent or "",
            result="success",
            detail=f"用户确认转人工 reason={body.reason}",
        )
    except Exception as e:
        print(f"[handoff] audit log fail: {e}")
    # 3) 通知管理员（Webhook + 审计日志里已有记录）
    try:
        from core.alerts import send_alert
        send_alert(
            title="新转人工请求",
            message=(
                f"会话 {body.session_id}\n"
                f"用户：{_user.get('username', '')}（ID {_user.get('id')}）\n"
                f"原因：{body.reason}\n"
                f"问题：{(body.question or '')[:100]}"
            ),
            level="info",
        )
    except Exception as e:
        print(f"[handoff] alert fail: {e}")

    return {
        "ok": True,
        "handoff_id": handoff_id,
        "message": "已为您创建转人工请求，客服人员会尽快处理，请稍候。",
    }


# ---------- 链路追踪 ----------
@app.get("/api/traces")
async def list_traces_api(limit: int = 50,
                          page: int = 1, page_size: int | None = None,
                          _admin: dict = Depends(require_role("admin"))):
    """获取链路追踪列表"""
    return db.list_traces(limit=limit, page=page, page_size=page_size)


@app.get("/api/traces/{trace_id}")
async def get_trace_api(trace_id: str, _admin: dict = Depends(require_role("admin"))):
    """获取单条链路详情"""
    trace = db.get_trace(trace_id)
    if not trace:
        raise HTTPException(404, "链路记录不存在")
    return trace


# ---------- 前端 ----------
STATIC_DIR = os.path.join(config.BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    # no-cache: 避免浏览器缓存旧页面导致前端更新不生效
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ============================================================================
# 超级管理员专属：安全审计 & 系统日志查询
# ============================================================================
@app.get("/api/audit-logs")
async def api_audit_logs(
    limit: int = 200, action: str = "", actor: str = "", keyword: str = "",
    page: int = 1, page_size: int | None = None,
    _: dict = Depends(require_role("superadmin")),
):
    return db.list_audit_logs(
        limit=limit, action=action, actor=actor, keyword=keyword,
        page=page, page_size=page_size,
    )


@app.get("/api/auth-failures")
async def api_auth_failures(
    username: str = "", ip: str = "", limit: int = 100,
    page: int = 1, page_size: int | None = None,
    _: dict = Depends(require_role("superadmin")),
):
    return db.list_auth_failures(username=username, ip=ip, limit=limit,
                                 page=page, page_size=page_size)


@app.get("/api/logs")
async def api_logs(
    limit: int = 500, level: str = "", keyword: str = "",
    page: int = 1, page_size: int | None = None,
    _: dict = Depends(require_role("superadmin")),
):
    from core.logger import read_logs
    # 分页模式先读全量，保证 total / 页数准确；非分页沿用 limit
    records = read_logs(limit=100000 if page_size else limit, level=level, keyword=keyword)
    if page_size is None:
        return records
    import math as _m
    ps = max(1, page_size)
    p = max(1, page)
    offset = (p - 1) * ps
    items = records[offset: offset + ps]
    return {
        "items": items,
        "total": len(records),
        "page": p, "page_size": ps,
        "pages": max(1, _m.ceil(len(records) / ps)),
    }


@app.get("/api/health")
async def health():
    """公开健康检查（负载均衡用）—— 深度版在 /api/health 由 health.router 提供"""
    # 保留兼容：简单版只返回 ok
    # 深度版被 include_router 先注册，这里作为 fallback 不会触发
    return {"status": "ok", "service": "rag-bot"}


# ============================================================================
# ✅ 企业级新 API
# ============================================================================

# ---- 密码策略（公开）----
@app.post("/api/auth/password-strength")
async def api_password_strength(body: dict):
    """前端实时提示密码强度"""
    pwd = body.get("password", "")
    valid, err = _pw_policy.policy.validate(pwd) if pwd else (False, "")
    strength = _pw_policy.check_password_strength(pwd)
    return {"valid": valid, "error": err, **strength}


# ---- 审计日志导出 ----
@app.get("/api/admin/audit/export")
async def api_export_audit(
    fmt: str = "json", limit: int = 5000, action: str = "",
    _: dict = Depends(require_role("superadmin")),
):
    result = _exporters.export_audit_logs(limit=limit, fmt=fmt, action=action or None)
    from fastapi.responses import Response
    if fmt == "csv":
        return Response(content=result["content"], media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={result['filename']}"})
    return result


@app.get("/api/admin/cost/export")
async def api_export_cost(
    user_id: int | None = None, days: int = 30, fmt: str = "json",
    _: dict = Depends(require_role("superadmin")),
):
    result = _exporters.export_cost_logs(user_id=user_id, days=days, fmt=fmt)
    from fastapi.responses import Response
    if fmt == "csv":
        return Response(content=result["content"], media_type="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={result['filename']}"})
    return result


# ---- 成本追踪看板 ----
@app.get("/api/admin/cost/global")
async def api_cost_global(days: int = 30, _: dict = Depends(require_role("superadmin"))):
    return _cost.get_global_cost(days=days)


@app.get("/api/admin/cost/me")
async def api_cost_me(days: int = 7, user: dict = Depends(get_current_user)):
    return _cost.get_user_cost(user["id"], days=days)


# ---- 异步任务 ----
@app.post("/api/admin/tasks")
async def api_submit_task(body: dict, _: dict = Depends(require_role("superadmin"))):
    task_type = body.get("type")
    if not task_type:
        raise HTTPException(400, "missing 'type'")
    tid = _async_queue.submit_task(task_type, payload=body.get("payload"))
    return {"task_id": tid}


@app.get("/api/admin/tasks")
async def api_list_tasks(limit: int = 50, _: dict = Depends(require_role("superadmin"))):
    from core.db_engine import db_session
    from core.models import AsyncTask
    with db_session() as db:
        rows = db.query(AsyncTask).order_by(AsyncTask.created_at.desc()).limit(limit).all()
    return {"tasks": [
        {"id": t.id, "type": t.task_type, "status": t.status,
         "retry": t.retry_count, "error": (t.error or "")[:200],
         "created_at": t.created_at, "finished_at": t.finished_at}
        for t in rows
    ]}


# ---- 权限检查（调试用）----
@app.get("/api/admin/permissions/check")
async def api_check_permission(permission: str, user: dict = Depends(get_current_user)):
    ok = _acl.has_permission(user, permission)
    return {"permission": permission, "granted": ok, "role": user.get("role")}


# ---- 告警检查 ----
@app.post("/api/admin/alerts/check")
async def api_alerts_check(_: dict = Depends(require_role("superadmin"))):
    return {"alerts": _alerts.check_and_alert()}


# ============================================================================
# A6 ACL: 给 KB/Agent/User 路由加权限守卫
# ============================================================================
# 用法示例（在需要的路由上加）：Depends(_acl.require_permission("kb.write"))

# ============================================================================
# A7 组织架构 / 角色 / 用户归属 CRUD
# ============================================================================
PERMISSION_CATALOG = [
    ("kb.read", "知识库：查看"),
    ("kb.write", "知识库：新建/编辑"),
    ("kb.delete", "知识库：删除"),
    ("doc.read", "文档：查看"),
    ("doc.upload", "文档：上传"),
    ("doc.delete", "文档：删除"),
    ("agent.read", "智能体：查看"),
    ("agent.write", "智能体：编辑"),
    ("session.read", "会话：查看"),
    ("session.manage", "会话：管理/转人工"),
    ("ticket.manage", "工单：处理"),
    ("user.read", "用户：查看"),
    ("user.write", "用户：编辑"),
    ("audit.read", "审计：查看"),
    ("audit.export", "审计：导出"),
    ("cost.read", "成本：查看"),
    ("org.manage", "组织架构：管理"),
    ("role.manage", "角色权限：管理"),
    ("system.config", "系统配置"),
]


def _org_dept_maps() -> tuple[dict, dict]:
    """读取组织/部门名称映射（PostgreSQL 不可用时返回空表，不影响用户列表）"""
    try:
        from core.db_engine import db_session
        from core.models import Department, Organization
        with db_session() as s:
            orgs = {o.id: o.name for o in s.query(Organization).all()}
            depts = {d.id: d.name for d in s.query(Department).all()}
        return orgs, depts
    except Exception:
        return {}, {}


# ---- 组织 ----
@app.get("/api/admin/orgs")
async def api_list_orgs(_: dict = Depends(require_role("superadmin"))):
    from core.db_engine import db_session
    from core.models import Department, Organization
    with db_session() as db:
        rows = db.query(Organization).order_by(Organization.id).all()
        dept_rows = db.query(Department).order_by(Department.id).all()
        result = {
            "orgs": [{"id": o.id, "name": o.name, "code": o.code, "enabled": o.enabled,
                      "dept_count": sum(1 for d in dept_rows if d.org_id == o.id)} for o in rows],
            "departments": [{"id": d.id, "org_id": d.org_id, "parent_id": d.parent_id,
                             "name": d.name} for d in dept_rows],
        }
    return result


@app.post("/api/admin/orgs")
async def api_create_org(body: dict, _: dict = Depends(require_role("superadmin"))):
    from core.db_engine import db_session
    from core.models import Organization
    name = (body.get("name") or "").strip()
    code = (body.get("code") or "").strip()
    if not name or not code:
        raise HTTPException(400, "name 和 code 必填")
    with db_session() as db:
        if db.query(Organization).filter_by(code=code).first():
            raise HTTPException(400, f"组织编码 {code} 已存在")
        o = Organization(name=name, code=code)
        db.add(o)
        db.flush()
        return {"id": o.id}


@app.put("/api/admin/orgs/{org_id}")
async def api_update_org(org_id: int, body: dict, _: dict = Depends(require_role("superadmin"))):
    from core.db_engine import db_session
    from core.models import Organization
    with db_session() as db:
        o = db.query(Organization).filter_by(id=org_id).first()
        if not o:
            raise HTTPException(404, "组织不存在")
        if body.get("name"):
            o.name = body["name"].strip()
        if body.get("code"):
            o.code = body["code"].strip()
        if body.get("enabled") is not None:
            o.enabled = bool(body["enabled"])
    return {"ok": True}


@app.delete("/api/admin/orgs/{org_id}")
async def api_delete_org(org_id: int, _: dict = Depends(require_role("superadmin"))):
    from core.db_engine import db_session
    from core.models import Department, Organization
    with db_session() as db:
        o = db.query(Organization).filter_by(id=org_id).first()
        if not o:
            raise HTTPException(404, "组织不存在")
        if db.query(Department).filter_by(org_id=org_id).count():
            raise HTTPException(400, "该组织下还有部门，请先删除部门")
        db.delete(o)
    return {"ok": True}


# ---- 部门 ----
@app.post("/api/admin/departments")
async def api_create_dept(body: dict, _: dict = Depends(require_role("superadmin"))):
    from core.db_engine import db_session
    from core.models import Department, Organization
    org_id = body.get("org_id")
    name = (body.get("name") or "").strip()
    if not org_id or not name:
        raise HTTPException(400, "org_id 和 name 必填")
    with db_session() as db:
        if not db.query(Organization).filter_by(id=int(org_id)).first():
            raise HTTPException(404, "组织不存在")
        if db.query(Department).filter_by(org_id=int(org_id), name=name).first():
            raise HTTPException(400, f"该组织下已有部门「{name}」")
        d = Department(org_id=int(org_id), name=name,
                       parent_id=body.get("parent_id"))
        db.add(d)
        db.flush()
        return {"id": d.id}


@app.put("/api/admin/departments/{dept_id}")
async def api_update_dept(dept_id: int, body: dict, _: dict = Depends(require_role("superadmin"))):
    from core.db_engine import db_session
    from core.models import Department
    with db_session() as db:
        d = db.query(Department).filter_by(id=dept_id).first()
        if not d:
            raise HTTPException(404, "部门不存在")
        if body.get("name"):
            d.name = body["name"].strip()
        if "parent_id" in body:
            d.parent_id = body["parent_id"]
    return {"ok": True}


@app.delete("/api/admin/departments/{dept_id}")
async def api_delete_dept(dept_id: int, _: dict = Depends(require_role("superadmin"))):
    from core.db_engine import db_session
    from core.models import Department
    with db_session() as db:
        d = db.query(Department).filter_by(id=dept_id).first()
        if not d:
            raise HTTPException(404, "部门不存在")
        db.query(Department).filter_by(parent_id=dept_id).update({"parent_id": None})
        db.delete(d)
    return {"ok": True}


# ---- 用户归属 ----
@app.put("/api/admin/users/{user_id}/assign")
async def api_assign_user(user_id: int, body: dict, _: dict = Depends(require_role("superadmin"))):
    ok = db.assign_user_org(user_id, body.get("org_id"), body.get("dept_id"))
    if not ok:
        raise HTTPException(404, "用户不存在")
    return {"ok": True}


# ---- 角色 ----
@app.get("/api/admin/roles")
async def api_list_roles(_: dict = Depends(require_role("superadmin"))):
    return {"roles": db.list_roles()}


@app.post("/api/admin/roles")
async def api_create_role(body: dict, _: dict = Depends(require_role("superadmin"))):
    name = (body.get("name") or "").strip()
    display_name = (body.get("display_name") or "").strip() or name
    if not name:
        raise HTTPException(400, "角色标识 name 必填")
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,31}", name):
        raise HTTPException(400, "角色标识需为 2-32 位小写字母/数字/下划线，且以字母开头")
    if db.get_role(name):
        raise HTTPException(400, f"角色 {name} 已存在")
    if not db.create_role(name, display_name, body.get("description", ""),
                          int(body.get("level") or 10)):
        raise HTTPException(400, "创建失败，角色标识可能重复")
    return {"ok": True, "name": name}


@app.put("/api/admin/roles/{role_name}")
async def api_update_role(role_name: str, body: dict, _: dict = Depends(require_role("superadmin"))):
    if not db.get_role(role_name):
        raise HTTPException(404, "角色不存在")
    ok = db.update_role(
        role_name,
        display_name=body.get("display_name"),
        description=body.get("description"),
        level=int(body["level"]) if body.get("level") is not None else None,
        enabled=body.get("enabled"),
    )
    if not ok:
        raise HTTPException(400, "没有需要更新的字段")
    return {"ok": True}


@app.delete("/api/admin/roles/{role_name}")
async def api_delete_role(role_name: str, _: dict = Depends(require_role("superadmin"))):
    ok, msg = db.delete_role(role_name)
    if not ok:
        raise HTTPException(400, msg)
    # 同步清理 PG 中的角色权限
    try:
        from core.db_engine import db_session
        from core.models import RolePermission
        with db_session() as s:
            s.query(RolePermission).filter_by(role=role_name).delete()
    except Exception:
        pass
    return {"ok": True}


@app.get("/api/admin/permissions/catalog")
async def api_permission_catalog(_: dict = Depends(require_role("superadmin"))):
    return {"catalog": [{"key": k, "label": v} for k, v in PERMISSION_CATALOG]}


@app.get("/api/admin/permissions/roles")
async def api_list_role_permissions(_: dict = Depends(require_role("superadmin"))):
    from sqlalchemy import func
    from core.db_engine import db_session
    from core.models import RolePermission
    with db_session() as db:
        rows = (db.query(RolePermission.role, func.array_agg(RolePermission.permission))
                .group_by(RolePermission.role).all())
    return {r: p for r, p in rows}


@app.post("/api/admin/permissions/roles/{role}/grant")
async def api_grant_permission(role: str, body: dict, _: dict = Depends(require_role("superadmin"))):
    from core.db_engine import db_session
    from core.models import RolePermission
    perm = (body.get("permission") or "").strip()
    if not perm:
        raise HTTPException(400, "permission 必填")
    with db_session() as db:
        if not db.query(RolePermission).filter_by(role=role, permission=perm).first():
            db.add(RolePermission(role=role, permission=perm))
    return {"ok": True, "role": role, "permission": perm}


@app.delete("/api/admin/permissions/roles/{role}/revoke")
async def api_revoke_permission(role: str, body: dict, _: dict = Depends(require_role("superadmin"))):
    from core.db_engine import db_session
    from core.models import RolePermission
    perm = (body.get("permission") or "").strip()
    with db_session() as db:
        db.query(RolePermission).filter_by(role=role, permission=perm).delete()
    return {"ok": True}


@app.put("/api/admin/permissions/roles/{role}")
async def api_set_role_permissions(role: str, body: dict, _: dict = Depends(require_role("superadmin"))):
    """全量替换某角色的权限集合（前端勾选后一次性保存）"""
    from core.db_engine import db_session
    from core.models import RolePermission
    perms = body.get("permissions")
    if not isinstance(perms, list):
        raise HTTPException(400, "permissions 必须是数组")
    if not db.get_role(role):
        raise HTTPException(404, "角色不存在")
    valid = {k for k, _ in PERMISSION_CATALOG} | {"*"}
    cleaned = sorted({p.strip() for p in perms if isinstance(p, str) and p.strip() in valid})
    with db_session() as s:
        s.query(RolePermission).filter_by(role=role).delete()
        for p in cleaned:
            s.add(RolePermission(role=role, permission=p))
    return {"ok": True, "role": role, "count": len(cleaned)}



# ============================================================================
# A8 APScheduler: 定时告警 + 过期 session 清理
# ============================================================================
try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    _scheduler = AsyncIOScheduler(timezone="UTC")

    @_scheduler.scheduled_job("interval", minutes=5, id="health_check")
    async def _job_health():
        """每 5 分钟跑一次告警检查"""
        try:
            _alerts.check_and_alert()
        except Exception as e:
            _log.warning("scheduler health_check error: %s", e)

    @_scheduler.scheduled_job("interval", minutes=30, id="cleanup_sessions")
    async def _job_cleanup():
        """每 30 分钟清理过期 session"""
        try:
            _async_queue.submit_task("cleanup_sessions", {"cutoff_ts": time.time()})
        except Exception as e:
            _log.warning("scheduler cleanup_sessions error: %s", e)

    @_scheduler.scheduled_job("interval", hours=24, id="kb_doc_scan")
    async def _job_kb_doc_scan():
        """每 24 小时扫描 uploads/ vs 向量索引，重建过期/变更文档

        逻辑:
          1. 遍历 vector_store.list_documents() 拿到所有 doc_id → doc_name → metadata
          2. 对每个 doc_name，检查 data/uploads/ 下同文件的 mtime
          3. 如果 mtime > metadata.file_mtime → 该文档被修改 → 重建索引
          4. 如果 uploads/ 里找不到对应文件 → 该文件被删 → 从向量库 remove
          5. 同步执行（不阻塞其他 job，24h 周期的扫描很轻）
        """
        try:
            import os
            from core import vector_store as _vs
            from core import parser as _parser
            from core import ollama_client

            _log.info("[kb_scan] 开始扫描 uploads/ vs 向量索引...")
            store = _vs.store
            upload_dir = getattr(_config, "UPLOAD_DIR", None) or os.path.join(_config.DATA_DIR, "uploads")
            if not os.path.isdir(upload_dir):
                _log.warning("[kb_scan] uploads dir 不存在，跳过")
                return

            doc_list = store.list_documents()
            to_rebuild, to_remove = [], []

            for d in doc_list:
                doc_name = d.get("name", "")
                doc_id = d.get("id", "")
                # 在 uploads/ 找同名文件
                found = None
                for root, _, files in os.walk(upload_dir):
                    for f in files:
                        if f == doc_name and not f.startswith("."):
                            found = os.path.join(root, f); break
                    if found: break

                if not found:
                    to_remove.append((doc_id, doc_name)); continue

                try:
                    mtime = os.path.getmtime(found)
                    chunks_of_doc = [c for c in store._chunks if c.get("doc_id") == doc_id]
                    if chunks_of_doc:
                        stored_mtime = chunks_of_doc[0].get("file_mtime", 0)
                        if mtime > stored_mtime:
                            to_rebuild.append((doc_id, doc_name, found, mtime))
                except Exception as _e:
                    _log.warning("[kb_scan] mtime 失败 %s: %s", found, _e)

            _log.info("[kb_scan] 扫描结果: rebuild=%d, remove=%d", len(to_rebuild), len(to_remove))

            for doc_id, doc_name in to_remove:
                try:
                    store.delete_document(doc_id)
                    _log.info("[kb_scan] 已删除向量: doc_id=%s name=%s", doc_id, doc_name)
                except Exception as _e:
                    _log.error("[kb_scan] remove 失败 %s: %s", doc_id, _e)

            for doc_id, doc_name, file_path, mtime in to_rebuild:
                try:
                    elements = _parser.parse_file_layout(file_path)
                    chunks = _parser.build_layout_chunks(elements)
                    chunk_texts, parent_texts, chunk_images, layout_meta, parent_ids = \
                        _parser.layout_chunks_to_records(chunks)
                    if not chunk_texts:
                        _log.warning("[kb_scan] 分块为空: %s", doc_name); continue
                    vectors = ollama_client.embed_texts(chunk_texts)
                    existing = [c for c in store._chunks if c.get("doc_id") == doc_id]
                    kb_id = existing[0].get("kb_id", "default") if existing else "default"
                    new_meta = {
                        "doc_type": os.path.splitext(file_path)[1].lstrip("."),
                        "source_path": file_path,
                        "file_size": os.path.getsize(file_path),
                        "file_mtime": mtime,
                    }
                    store.update_document(
                        doc_id, doc_name, chunk_texts, vectors,
                        kb_id=kb_id, parent_texts=parent_texts,
                        chunk_images=chunk_images, layout_meta=layout_meta,
                        parent_ids=parent_ids, metadata=new_meta,
                    )
                    _log.info("[kb_scan] 已重建: doc_id=%s name=%s (%d chunks)", doc_id, doc_name, len(chunk_texts))
                except Exception as _e:
                    _log.error("[kb_scan] rebuild 失败 %s: %s", doc_name, _e, exc_info=True)

        except Exception as _e:
            _log.warning("scheduler kb_doc_scan error: %s", _e, exc_info=True)

    # 接入 FastAPI lifespan
    @app.on_event("startup")
    async def _start_scheduler():
        _scheduler.start()
        _log.info("APScheduler started (health=5m, cleanup=30m)")

    @app.on_event("shutdown")
    async def _stop_scheduler():
        _scheduler.shutdown(wait=False)
        _log.info("APScheduler stopped")
except ImportError:
    _log.warning("apscheduler not installed, background jobs disabled (pip install apscheduler)")


# ============================================================================
# B3 RAGAS 轻量评估路由
# ============================================================================
@app.post("/api/admin/ragas/evaluate")
async def api_ragas_evaluate(body: dict, user: dict = Depends(get_current_user)):
    """评估一组样本的 RAG 质量（调用轻量版 ragas_eval）"""
    from core.ragas_eval import evaluate_batch, evaluate_sample
    samples = body.get("samples", [])
    if not samples:
        raise HTTPException(400, "samples 不能为空")
    result = evaluate_batch(samples[:50])  # 最多 50 个
    return result


@app.post("/api/admin/ragas/evaluate-current")
async def api_ragas_evaluate_current(body: dict, user: dict = Depends(get_current_user)):
    """评估单次 chat 结果（传 question / answer / contexts）"""
    from core.ragas_eval import evaluate_sample
    q = body.get("question", "")
    a = body.get("answer", "")
    ctxs = body.get("contexts", [])
    return evaluate_sample(q, a, ctxs)


# ============================================================================
# B2 Redis Session 适配（在登录时把 session_id 同步到 Redis）
# ============================================================================
# 改造 authenticate 让它同时调 redis_store.redis_add_session
# （通过 app 里的 login 路由在 _db.create_user_session 之后追加调 redis）


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8008)
