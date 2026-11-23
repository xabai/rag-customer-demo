# -*- coding: utf-8 -*-
"""统一数据层 —— SQLite 实现（当 DB_DRIVER=postgresql 时，核心函数被 db_pg 覆盖）"""
import json
import sqlite3
import threading
import time
import uuid

import config

_lock = threading.Lock()


def _conn():
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _lock, _conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
            CREATE TABLE IF NOT EXISTS unanswered (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                session_id TEXT,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS faq (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS intent_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                keywords TEXT NOT NULL,
                action TEXT NOT NULL DEFAULT 'ticket',
                reply TEXT DEFAULT '',
                enabled INTEGER DEFAULT 1
            );
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT NOT NULL,
                intent TEXT NOT NULL,
                status TEXT DEFAULT 'open',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS handoffs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS knowledge_bases (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                chunk_config TEXT DEFAULT '{"child_size":500,"child_overlap":50,"parent_size":2000,"split_mode":"smart"}',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS agents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                description TEXT DEFAULT '',
                enabled INTEGER DEFAULT 1,
                endpoint TEXT DEFAULT '',
                auth_type TEXT DEFAULT 'none',
                auth_value TEXT DEFAULT '',
                schema_json TEXT DEFAULT '{}',
                builtin INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                message_id INTEGER,
                rating INTEGER NOT NULL,
                comment TEXT DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS qa_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                message_id INTEGER,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                hits_json TEXT DEFAULT '[]',
                elapsed_ms INTEGER DEFAULT 0,
                used_faq INTEGER DEFAULT 0,
                used_tool INTEGER DEFAULT 0,
                no_match INTEGER DEFAULT 0,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_qa_log_session ON qa_log(session_id);
            CREATE TABLE IF NOT EXISTS eval_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT NOT NULL,
                expected TEXT NOT NULL,
                source TEXT DEFAULT 'manual',
                kb_id TEXT DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                enabled INTEGER DEFAULT 1,
                failed_attempts INTEGER DEFAULT 0,
                lockout_until REAL DEFAULT NULL,
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                revoked INTEGER DEFAULT 0,
                exp_at REAL NOT NULL,
                created_at REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                actor_user_id INTEGER,
                actor_username TEXT,
                action TEXT NOT NULL,
                resource_type TEXT,
                resource_id TEXT,
                ip TEXT,
                user_agent TEXT,
                result TEXT NOT NULL,
                detail TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_audit_logs_ts ON audit_logs(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
            CREATE TABLE IF NOT EXISTS auth_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                ip TEXT,
                user_agent TEXT,
                timestamp REAL NOT NULL
            );
            -- 角色元数据（角色名 + 展示名 + 等级 + 是否内置）
            -- 权限明细仍在 PostgreSQL role_permissions 表，这里只存角色本身
            CREATE TABLE IF NOT EXISTS roles (
                name TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                description TEXT DEFAULT '',
                level INTEGER DEFAULT 0,
                builtin INTEGER DEFAULT 0,
                enabled INTEGER DEFAULT 1,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_auth_failures_usr_ip ON auth_failures(username, ip);
        """)

        # 老表兼容：补齐 chunk_config 列（如果已经有就跳过）
        try:
            cols = [r[1] for r in conn.execute("PRAGMA table_info(knowledge_bases)").fetchall()]
            if "chunk_config" not in cols:
                conn.execute("ALTER TABLE knowledge_bases ADD COLUMN chunk_config TEXT DEFAULT '{}'")
        except Exception:
            pass

        # ======== 兼容迁移：已有库补缺失列 ========
        def _has_col(table, col):
            return any(r["name"] == col for r in conn.execute(f"PRAGMA table_info({table})"))

        for tbl, col, default in [
            ("sessions",  "user_id", "NULL"),
            ("memories",   "user_id", "NULL"),
            ("users",      "failed_attempts", "0"),
            ("users",      "lockout_until", "NULL"),
            ("users",      "org_id", "NULL"),
            ("users",      "dept_id", "NULL"),
        ]:
            if not _has_col(tbl, col):
                # SQLite ALTER TABLE 不支持参数化，必须硬编码
                conn.execute(f"ALTER TABLE {tbl} ADD COLUMN {col} INTEGER DEFAULT {default}")
        conn.execute("UPDATE sessions  SET user_id = 1 WHERE user_id IS NULL")
        conn.execute("UPDATE memories SET user_id = 1 WHERE user_id IS NULL")
        conn.execute("UPDATE users SET failed_attempts = 0 WHERE failed_attempts IS NULL")

        # 内置默认用户（admin，仅在首次启动创建）
        count = conn.execute("SELECT COUNT(*) c FROM users").fetchone()["c"]
        if count == 0:
            # 懒导入避免循环
            import bcrypt as _bcrypt
            admin_pwd = _bcrypt.hashpw(b"admin123", _bcrypt.gensalt()).decode("utf-8")
            user_pwd = _bcrypt.hashpw(b"user123", _bcrypt.gensalt()).decode("utf-8")
            conn.executemany(
                """INSERT INTO users (username, password, role, enabled, created_at)
                   VALUES (?, ?, ?, 1, ?)""",
                [
                    ("admin", admin_pwd, "superadmin", time.time()),
                    ("user",  user_pwd, "user",       time.time()),
                ],
            )
        # 内置默认角色（只在缺失时插入，不覆盖用户改过的展示名）
        conn.executemany(
            """INSERT OR IGNORE INTO roles
               (name, display_name, description, level, builtin, enabled, created_at)
               VALUES (?, ?, ?, ?, 1, 1, ?)""",
            [
                ("superadmin", "超级管理员", "系统最高权限，可管理组织、角色与审计", 100, time.time()),
                ("admin",      "管理员",     "知识库、会话、数据看板等后台管理", 50, time.time()),
                ("agent",      "客服坐席",   "处理会话与工单，只读知识库", 20, time.time()),
                ("user",       "普通用户",   "仅可使用在线客服对话", 10, time.time()),
            ],
        )
        # 内置默认意图规则（仅在表为空时插入）
        count = conn.execute("SELECT COUNT(*) c FROM intent_rules").fetchone()["c"]
        if count == 0:
            conn.executemany(
                "INSERT INTO intent_rules (name, keywords, action, reply) VALUES (?, ?, ?, ?)",
                [
                    ("转人工", "转人工,人工客服,真人客服,人工服务", "handoff",
                     "已为您转接人工客服，请稍候，客服会尽快接入本会话。"),
                    ("投诉建议", "投诉,举报,差评,维权", "ticket",
                     "已收到您的反馈，我们已创建工单并会尽快处理，请保持联系方式畅通。"),
                    ("退换售后", "退货,退款,换货,返修", "ticket",
                     "已为您创建售后工单，工作人员会尽快与您联系处理。"),
                ],
            )


# ---------- 会话 ----------
def create_session(title="新对话", user_id=None):
    sid = uuid.uuid4().hex[:12]
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, user_id, created_at) VALUES (?, ?, ?, ?)",
            (sid, title, user_id, time.time()),
        )
    return {"id": sid, "title": title}


def list_sessions(user_id=None, page=1, page_size=None):
    """page_size=None 时保持旧行为返回 list；否则返回 {items, total, page, page_size, pages}"""
    with _lock, _conn() as conn:
        where = ""
        params = []
        if user_id is not None:
            where = " WHERE user_id = ?"
            params.append(user_id)
        sql = "SELECT id, title, user_id, created_at FROM sessions" + where + " ORDER BY created_at DESC"
        if page_size is None:
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
        ps = max(1, int(page_size))
        p = max(1, int(page))
        total = conn.execute("SELECT COUNT(1) AS c FROM sessions" + where, params).fetchone()["c"]
        pages = max(1, (total + ps - 1) // ps)
        p = min(p, pages)
        offset = (p - 1) * ps
        rows = conn.execute(sql + " LIMIT ? OFFSET ?", params + [ps, offset]).fetchall()
        return {"items": [dict(r) for r in rows], "total": total, "page": p, "page_size": ps, "pages": pages}


def rename_session(session_id, title):
    with _lock, _conn() as conn:
        conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))


def delete_session(session_id):
    with _lock, _conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))


# ---------- 消息 ----------
def _ensure_extra_columns(conn):
    """老库兼容：messages 表缺少 images / sources 列时自动添加"""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(messages)").fetchall()]
    if "images" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN images TEXT")
    if "sources" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN sources TEXT")


def add_message(session_id, role, content, images=None, sources=None):
    import json as _json
    with _lock, _conn() as conn:
        _ensure_extra_columns(conn)
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content, created_at, images, sources) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, time.time(),
             _json.dumps(images, ensure_ascii=False) if images else None,
             _json.dumps(sources, ensure_ascii=False) if sources else None),
        )
        return cur.lastrowid


def get_messages(session_id, limit=None):
    import json as _json
    sql = "SELECT role, content, images, sources FROM messages WHERE session_id = ? ORDER BY id"
    if limit:
        sql = f"SELECT role, content, images, sources FROM (SELECT * FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT {int(limit)}) ORDER BY id"
    with _lock, _conn() as conn:
        _ensure_extra_columns(conn)
        rows = conn.execute(sql, (session_id,)).fetchall()
    result = []
    for r in rows:
        d = dict(r)
        d["images"] = _json.loads(d["images"]) if d.get("images") else None
        d["sources"] = _json.loads(d["sources"]) if d.get("sources") else None
        result.append(d)
    return result


# ---------- 未命中问题沉淀 ----------
def add_unanswered(question, session_id=None):
    """记录知识库未命中的问题（同一问题不重复记录）"""
    with _lock, _conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM unanswered WHERE question = ?", (question,)
        ).fetchone()
        if exists:
            return
        conn.execute(
            "INSERT INTO unanswered (question, session_id, created_at) VALUES (?, ?, ?)",
            (question, session_id, time.time()),
        )


def list_unanswered(page=1, page_size=None):
    """page_size=None 时保持旧行为返回 list；否则返回分页 dict"""
    with _lock, _conn() as conn:
        sql = "SELECT id, question, created_at FROM unanswered ORDER BY id DESC"
        if page_size is None:
            rows = conn.execute(sql).fetchall()
            return [
                {"id": r["id"], "question": r["question"],
                 "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}
                for r in rows
            ]
        ps = max(1, int(page_size))
        p = max(1, int(page))
        total = conn.execute("SELECT COUNT(1) AS c FROM unanswered").fetchone()["c"]
        pages = max(1, (total + ps - 1) // ps)
        p = min(p, pages)
        offset = (p - 1) * ps
        rows = conn.execute(sql + " LIMIT ? OFFSET ?", (ps, offset)).fetchall()
        items = [
            {"id": r["id"], "question": r["question"],
             "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}
            for r in rows
        ]
        return {"items": items, "total": total, "page": p, "page_size": ps, "pages": pages}


def delete_unanswered(item_id):
    with _lock, _conn() as conn:
        conn.execute("DELETE FROM unanswered WHERE id = ?", (item_id,))


# ---------- FAQ ----------
def add_faq(question, answer):
    with _lock, _conn() as conn:
        cur = conn.execute(
            "INSERT INTO faq (question, answer, created_at) VALUES (?, ?, ?)",
            (question, answer, time.time()),
        )
        return cur.lastrowid


def list_faq(page=1, page_size=None):
    """page_size=None 时保持旧行为返回 list；否则返回分页 dict"""
    with _lock, _conn() as conn:
        sql = "SELECT id, question, answer FROM faq ORDER BY id DESC"
        if page_size is None:
            rows = conn.execute(sql).fetchall()
            return [dict(r) for r in rows]
        ps = max(1, int(page_size))
        p = max(1, int(page))
        total = conn.execute("SELECT COUNT(1) AS c FROM faq").fetchone()["c"]
        pages = max(1, (total + ps - 1) // ps)
        p = min(p, pages)
        offset = (p - 1) * ps
        rows = conn.execute(sql + " LIMIT ? OFFSET ?", (ps, offset)).fetchall()
        return {"items": [dict(r) for r in rows], "total": total, "page": p, "page_size": ps, "pages": pages}


def delete_faq(item_id):
    with _lock, _conn() as conn:
        conn.execute("DELETE FROM faq WHERE id = ?", (item_id,))


# ---------- 意图规则 ----------
def list_intent_rules(enabled_only=False, page=1, page_size=None):
    """page_size=None 时保持旧行为返回 list；否则返回分页 dict"""
    where = " WHERE enabled = 1" if enabled_only else ""
    sql = "SELECT * FROM intent_rules" + where + " ORDER BY id"
    with _lock, _conn() as conn:
        if page_size is None:
            rows = conn.execute(sql).fetchall()
            return [dict(r) for r in rows]
        ps = max(1, int(page_size))
        p = max(1, int(page))
        total = conn.execute("SELECT COUNT(1) AS c FROM intent_rules" + where).fetchone()["c"]
        pages = max(1, (total + ps - 1) // ps)
        p = min(p, pages)
        offset = (p - 1) * ps
        rows = conn.execute(sql + " LIMIT ? OFFSET ?", (ps, offset)).fetchall()
        return {"items": [dict(r) for r in rows], "total": total, "page": p, "page_size": ps, "pages": pages}


def add_intent_rule(name, keywords, action, reply):
    with _lock, _conn() as conn:
        cur = conn.execute(
            "INSERT INTO intent_rules (name, keywords, action, reply) VALUES (?, ?, ?, ?)",
            (name, keywords, action, reply),
        )
        return cur.lastrowid


def delete_intent_rule(rule_id):
    with _lock, _conn() as conn:
        conn.execute("DELETE FROM intent_rules WHERE id = ?", (rule_id,))


def toggle_intent_rule(rule_id, enabled):
    with _lock, _conn() as conn:
        conn.execute("UPDATE intent_rules SET enabled = ? WHERE id = ?", (enabled, rule_id))


# ---------- 工单 ----------
def add_ticket(session_id, question, intent):
    with _lock, _conn() as conn:
        cur = conn.execute(
            "INSERT INTO tickets (session_id, question, intent, created_at) VALUES (?, ?, ?, ?)",
            (session_id, question, intent, time.time()),
        )
        return cur.lastrowid


def list_tickets(page=1, page_size=None):
    """page_size=None 时保持旧行为返回 list；否则返回分页 dict"""
    with _lock, _conn() as conn:
        sql = "SELECT * FROM tickets ORDER BY id DESC"
        if page_size is None:
            rows = conn.execute(sql).fetchall()
            return [
                {"id": r["id"], "question": r["question"], "intent": r["intent"],
                 "status": r["status"], "session_id": r["session_id"],
                 "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}
                for r in rows
            ]
        ps = max(1, int(page_size))
        p = max(1, int(page))
        total = conn.execute("SELECT COUNT(1) AS c FROM tickets").fetchone()["c"]
        pages = max(1, (total + ps - 1) // ps)
        p = min(p, pages)
        offset = (p - 1) * ps
        rows = conn.execute(sql + " LIMIT ? OFFSET ?", (ps, offset)).fetchall()
        items = [
            {"id": r["id"], "question": r["question"], "intent": r["intent"],
             "status": r["status"], "session_id": r["session_id"],
             "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}
            for r in rows
        ]
        return {"items": items, "total": total, "page": p, "page_size": ps, "pages": pages}


def update_ticket_status(ticket_id, status):
    with _lock, _conn() as conn:
        conn.execute("UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id))


# ---------- 转人工 ----------
def add_handoff(session_id, question):
    with _lock, _conn() as conn:
        cur = conn.execute(
            "INSERT INTO handoffs (session_id, question, created_at) VALUES (?, ?, ?)",
            (session_id, question, time.time()),
        )
        return cur.lastrowid


def list_handoffs(page=1, page_size=None):
    """page_size=None 时保持旧行为返回 list；否则返回分页 dict"""
    with _lock, _conn() as conn:
        sql = "SELECT * FROM handoffs ORDER BY id DESC"
        if page_size is None:
            rows = conn.execute(sql).fetchall()
            return [
                {"id": r["id"], "question": r["question"], "status": r["status"],
                 "session_id": r["session_id"],
                 "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}
                for r in rows
            ]
        ps = max(1, int(page_size))
        p = max(1, int(page))
        total = conn.execute("SELECT COUNT(1) AS c FROM handoffs").fetchone()["c"]
        pages = max(1, (total + ps - 1) // ps)
        p = min(p, pages)
        offset = (p - 1) * ps
        rows = conn.execute(sql + " LIMIT ? OFFSET ?", (ps, offset)).fetchall()
        items = [
            {"id": r["id"], "question": r["question"], "status": r["status"],
             "session_id": r["session_id"],
             "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}
            for r in rows
        ]
        return {"items": items, "total": total, "page": p, "page_size": ps, "pages": pages}


def update_handoff_status(handoff_id, status):
    with _lock, _conn() as conn:
        conn.execute("UPDATE handoffs SET status = ? WHERE id = ?", (status, handoff_id))


# ---------- 知识库分组 ----------
def create_kb(name, description="", chunk_config=None):
    import json as _json
    default_cfg = {"child_size": 500, "child_overlap": 50, "parent_size": 2000, "split_mode": "smart"}
    cfg_str = _json.dumps(chunk_config or default_cfg, ensure_ascii=False)
    kid = f"kb_{uuid.uuid4().hex[:8]}"
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO knowledge_bases (id, name, description, chunk_config, created_at) VALUES (?, ?, ?, ?, ?)",
            (kid, name, description, cfg_str, time.time()),
        )
    return {"id": kid, "name": name, "description": description, "chunk_config": chunk_config or default_cfg}


def get_kb(kid):
    with _lock, _conn() as conn:
        row = conn.execute(
            "SELECT * FROM knowledge_bases WHERE id = ?", (kid,)
        ).fetchone()
        if not row:
            return None
        import json as _json
        cfg_str = row["chunk_config"] or "{}"
        try:
            cfg = _json.loads(cfg_str)
        except Exception:
            cfg = {"child_size": 500, "child_overlap": 50, "parent_size": 2000, "split_mode": "smart"}
        return {
            "id": row["id"], "name": row["name"],
            "description": row["description"],
            "chunk_config": cfg,
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["created_at"])),
        }


def update_kb(kid, name=None, description=None, chunk_config=None):
    with _lock, _conn() as conn:
        if name is not None:
            conn.execute("UPDATE knowledge_bases SET name = ? WHERE id = ?", (name, kid))
        if description is not None:
            conn.execute("UPDATE knowledge_bases SET description = ? WHERE id = ?", (description, kid))
        if chunk_config is not None:
            import json as _json
            cfg_str = _json.dumps(chunk_config, ensure_ascii=False)
            conn.execute("UPDATE knowledge_bases SET chunk_config = ? WHERE id = ?", (cfg_str, kid))
        return True


def list_kb(page=1, page_size=None):
    """page_size=None 时保持旧行为返回 list；否则返回分页 dict"""
    import json as _json
    with _lock, _conn() as conn:
        sql = "SELECT id, name, description, chunk_config, created_at FROM knowledge_bases ORDER BY created_at ASC"
        if page_size is None:
            rows = conn.execute(sql).fetchall()
            result = []
            for r in rows:
                try:
                    cfg = _json.loads(r["chunk_config"] or "{}")
                except Exception:
                    cfg = {"child_size": 500, "child_overlap": 50, "parent_size": 2000, "split_mode": "smart"}
                result.append({
                    "id": r["id"], "name": r["name"], "description": r["description"],
                    "chunk_config": cfg,
                    "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"])),
                })
            return result
        ps = max(1, int(page_size))
        p = max(1, int(page))
        total = conn.execute("SELECT COUNT(1) AS c FROM knowledge_bases").fetchone()["c"]
        pages = max(1, (total + ps - 1) // ps)
        p = min(p, pages)
        offset = (p - 1) * ps
        rows = conn.execute(sql + " LIMIT ? OFFSET ?", (ps, offset)).fetchall()
        items = [
            {"id": r["id"], "name": r["name"], "description": r["description"],
             "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}
            for r in rows
        ]
        return {"items": items, "total": total, "page": p, "page_size": ps, "pages": pages}


def delete_kb(kid):
    with _lock, _conn() as conn:
        conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kid,))


# ---------- 记忆（摘要 + 跨会话偏好） ----------
def add_memory(session_id, kind, content, user_id=None):
    with _lock, _conn() as conn:
        cur = conn.execute(
            "INSERT INTO memories (session_id, user_id, kind, content, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, kind, content, time.time()),
        )
        return cur.lastrowid


def get_session_summary(session_id):
    """获取该会话的摘要记忆（最新一条）"""
    with _lock, _conn() as conn:
        row = conn.execute(
            "SELECT content FROM memories WHERE session_id = ? AND kind = 'summary' ORDER BY id DESC LIMIT 1",
            (session_id,),
        ).fetchone()
    return row["content"] if row else None


def list_cross_memories(user_id=None):
    """跨会话记忆（kind=preference/fact），注入到所有对话"""
    with _lock, _conn() as conn:
        sql = "SELECT id, content, created_at FROM memories WHERE kind IN ('preference','fact')"
        params = []
        if user_id is not None:
            sql += " AND (user_id = ? OR user_id IS NULL)"
            params.append(user_id)
        sql += " ORDER BY id DESC"
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def list_memories(user_id=None, session_id=None, keyword=None, kind=None, page=1, page_size=None):
    """列出记忆，支持按 user_id / session_id / kind 过滤

    page_size=None 时保持旧行为返回 list；否则返回 {items, total, page, page_size, pages}
    """
    sql = """
        SELECT m.id, m.session_id, m.user_id, m.kind, m.content, m.created_at,
               s.title AS session_title,
               u.username
        FROM memories m
        LEFT JOIN sessions s ON m.session_id = s.id
        LEFT JOIN users u ON m.user_id = u.id
        WHERE 1=1
    """
    # 用 COUNT(1) 避免复杂 SELECT 替换出错
    count_sql = """
        SELECT COUNT(1) AS c
        FROM memories m
        LEFT JOIN sessions s ON m.session_id = s.id
        LEFT JOIN users u ON m.user_id = u.id
        WHERE 1=1
    """
    params = []
    if user_id is not None:
        sql += " AND (m.user_id = ? OR m.user_id IS NULL)"
        count_sql += " AND (m.user_id = ? OR m.user_id IS NULL)"
        params.append(user_id)
    if session_id:
        sql += " AND m.session_id = ?"
        count_sql += " AND m.session_id = ?"
        params.append(session_id)
    if kind:
        sql += " AND m.kind = ?"
        count_sql += " AND m.kind = ?"
        params.append(kind)
    if keyword:
        sql += " AND m.content LIKE ?"
        count_sql += " AND m.content LIKE ?"
        params.append(f"%{keyword}%")
    sql += " ORDER BY m.id DESC"

    with _lock, _conn() as conn:
        if page_size is not None:
            total = conn.execute(count_sql, params).fetchone()["c"]
            offset = (page - 1) * page_size
            rows = conn.execute(sql + " LIMIT ? OFFSET ?", params + [page_size, offset]).fetchall()
        else:
            rows = conn.execute(sql, params).fetchall()
            total = None

    items = [
        {
            "id": r["id"],
            "session_id": r["session_id"],
            "session_title": r["session_title"] or "(已删除会话)",
            "user_id": r["user_id"],
            "username": r["username"] or "(未知)",
            "kind": r["kind"],
            "content": r["content"],
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))
        }
        for r in rows
    ]

    if page_size is not None:
        pages = max(1, (total + page_size - 1) // page_size)
        return {"items": items, "total": total, "page": page, "page_size": page_size, "pages": pages}
    return items


def delete_memory(mem_id):
    with _lock, _conn() as conn:
        conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))


# ---------- 用户反馈 ----------
def add_feedback(session_id, message_id, rating, comment=""):
    with _lock, _conn() as conn:
        cur = conn.execute(
            "INSERT INTO feedback (session_id, message_id, rating, comment, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, message_id, rating, comment, time.time()),
        )
        return cur.lastrowid


def list_feedback():
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM feedback ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats():
    """汇总统计：对话数、命中率、反馈分布等"""
    with _lock, _conn() as conn:
        sessions = conn.execute("SELECT COUNT(*) c FROM sessions").fetchone()["c"]
        msgs = conn.execute("SELECT COUNT(*) c FROM messages").fetchone()["c"]
        faq_count = conn.execute("SELECT COUNT(*) c FROM faq").fetchone()["c"]
        tickets = conn.execute("SELECT COUNT(*) c FROM tickets").fetchone()["c"]
        handoffs = conn.execute("SELECT COUNT(*) c FROM handoffs").fetchone()["c"]
        unanswered = conn.execute("SELECT COUNT(*) c FROM unanswered").fetchone()["c"]
        like = conn.execute("SELECT COUNT(*) c FROM feedback WHERE rating = 1").fetchone()["c"]
        dislike = conn.execute("SELECT COUNT(*) c FROM feedback WHERE rating = -1").fetchone()["c"]
    return {
        "sessions": sessions, "messages": msgs, "faq_count": faq_count,
        "tickets": tickets, "handoffs": handoffs, "unanswered": unanswered,
        "likes": like, "dislikes": dislike,
    }


# ---------- 问答日志（用于复盘与导出评测集） ----------
def add_qa_log(session_id, message_id, question, answer, hits, elapsed_ms,
               used_faq=False, used_tool=False, no_match=False):
    with _lock, _conn() as conn:
        cur = conn.execute(
            """INSERT INTO qa_log
               (session_id, message_id, question, answer, hits_json, elapsed_ms,
                used_faq, used_tool, no_match, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, message_id, question, answer, json.dumps(hits, ensure_ascii=False),
             elapsed_ms, 1 if used_faq else 0, 1 if used_tool else 0,
             1 if no_match else 0, time.time()),
        )
        return cur.lastrowid


def list_qa_log(session_id="", limit=100, page=1, page_size=None):
    """问答日志列表

    page_size=None 时保持旧行为（limit 条 list）；否则返回 {items, total, page, page_size, pages}
    """
    with _lock, _conn() as conn:
        if page_size is not None:
            count_sql = "SELECT COUNT(*) AS c FROM qa_log"
            sql = "SELECT * FROM qa_log"
            args = []
            if session_id:
                count_sql += " WHERE session_id = ?"
                sql += " WHERE session_id = ?"
                args.append(session_id)
            total = conn.execute(count_sql, args).fetchone()["c"]
            offset = (page - 1) * page_size
            rows = conn.execute(sql + " ORDER BY id DESC LIMIT ? OFFSET ?", args + [page_size, offset]).fetchall()
            items = [dict(r) for r in rows]
            pages = max(1, (total + page_size - 1) // page_size)
            return {"items": items, "total": total, "page": page, "page_size": page_size, "pages": pages}
        else:
            sql = "SELECT * FROM qa_log"
            args = []
            if session_id:
                sql += " WHERE session_id = ?"
                args.append(session_id)
            sql += " ORDER BY id DESC LIMIT ?"
            args.append(limit)
            rows = conn.execute(sql, args).fetchall()
    return [dict(r) for r in rows]


def export_badcases():
    """导出点踩或无命中的问答对（JSONL）"""
    with _lock, _conn() as conn:
        rows = conn.execute("""
            SELECT q.*, f.rating
            FROM qa_log q
            LEFT JOIN feedback f ON f.message_id = q.message_id AND f.session_id = q.session_id
            WHERE q.no_match = 1 OR f.rating = -1
            ORDER BY q.id DESC
        """).fetchall()
    return [dict(r) for r in rows]


# ---------- 看板趋势统计（按天/分布/耗时） ----------
def get_daily_qa(days=14):
    """最近 N 天每日问答量，返回 [{date: '2026-10-16', count: 12}, ...]"""
    cutoff = time.time() - days * 86400
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT date(created_at, 'unixepoch', 'localtime') AS d, COUNT(*) AS c "
            "FROM qa_log WHERE created_at >= ? GROUP BY d ORDER BY d",
            (cutoff,)
        ).fetchall()
    counts = {r["d"]: r["c"] for r in rows}
    # 补全缺失日期为 0
    out = []
    for i in range(days - 1, -1, -1):
        d = time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400))
        out.append({"date": d, "count": counts.get(d, 0)})
    return out


def get_daily_feedback(days=14):
    """最近 N 天每日点赞/点踩数，返回 [{date, likes, dislikes}, ...]"""
    cutoff = time.time() - days * 86400
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT date(created_at, 'unixepoch', 'localtime') AS d, "
            "SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END) AS likes, "
            "SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) AS dislikes "
            "FROM feedback WHERE created_at >= ? GROUP BY d ORDER BY d",
            (cutoff,)
        ).fetchall()
    data = {r["d"]: {"likes": r["likes"], "dislikes": r["dislikes"]} for r in rows}
    out = []
    for i in range(days - 1, -1, -1):
        d = time.strftime("%Y-%m-%d", time.localtime(time.time() - i * 86400))
        out.append({"date": d,
                    "likes": data.get(d, {}).get("likes", 0),
                    "dislikes": data.get(d, {}).get("dislikes", 0)})
    return out


def get_qa_source_distribution():
    """问答来源占比：FAQ 命中 / 知识库命中 / 未命中，返回 [{name, value}, ...]"""
    with _lock, _conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM qa_log").fetchone()[0] or 0
        if total == 0:
            return [
                {"name": "FAQ 命中", "value": 0},
                {"name": "知识库命中", "value": 0},
                {"name": "未命中", "value": 0},
            ]
        faq = conn.execute("SELECT COUNT(*) FROM qa_log WHERE used_faq=1").fetchone()[0] or 0
        no_match = conn.execute("SELECT COUNT(*) FROM qa_log WHERE no_match=1").fetchone()[0] or 0
        kb_hit = total - faq - no_match
        if kb_hit < 0:
            kb_hit = 0
        return [
            {"name": "FAQ 命中", "value": faq},
            {"name": "知识库命中", "value": kb_hit},
            {"name": "未命中", "value": no_match},
        ]


def get_latency_stats(days=14):
    """问答耗时统计：平均、P50、P90、P95（毫秒）"""
    cutoff = time.time() - days * 86400
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT elapsed_ms FROM qa_log WHERE created_at >= ? AND elapsed_ms > 0",
            (cutoff,)
        ).fetchall()
    vals = sorted([r["elapsed_ms"] for r in rows])
    if not vals:
        return {"avg": 0, "p50": 0, "p90": 0, "p95": 0, "count": 0}
    n = len(vals)
    def pct(p):
        idx = min(n - 1, int(n * p / 100))
        return vals[idx]
    return {
        "avg": round(sum(vals) / n),
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "count": n,
    }


# ---------- 评测集（badcase 标注 + 回归测试） ----------
def add_eval_case(question, expected, source="manual", kb_id=""):
    with _lock, _conn() as conn:
        cur = conn.execute(
            "INSERT INTO eval_cases (question, expected, source, kb_id, created_at) VALUES (?, ?, ?, ?, ?)",
            (question, expected, source, kb_id, time.time()),
        )
        return cur.lastrowid


def list_eval_cases(page=1, page_size=None):
    """page_size=None 时保持旧行为返回 list；否则返回分页 dict"""
    with _lock, _conn() as conn:
        sql = "SELECT * FROM eval_cases ORDER BY id DESC"
        if page_size is None:
            rows = conn.execute(sql).fetchall()
            return [dict(r) for r in rows]
        ps = max(1, int(page_size))
        p = max(1, int(page))
        total = conn.execute("SELECT COUNT(1) AS c FROM eval_cases").fetchone()["c"]
        pages = max(1, (total + ps - 1) // ps)
        p = min(p, pages)
        offset = (p - 1) * ps
        rows = conn.execute(sql + " LIMIT ? OFFSET ?", (ps, offset)).fetchall()
        return {"items": [dict(r) for r in rows], "total": total, "page": p, "page_size": ps, "pages": pages}


def update_eval_case(case_id, question, expected):
    with _lock, _conn() as conn:
        conn.execute("UPDATE eval_cases SET question=?, expected=? WHERE id=?",
                     (question, expected, case_id))


def delete_eval_case(case_id):
    with _lock, _conn() as conn:
        conn.execute("DELETE FROM eval_cases WHERE id=?", (case_id,))


# ---------- 知识库健康度 ----------
def get_kb_health():
    """统计每个文档被命中次数、平均分、最近命中时间（来源 qa_log.hits_json）"""
    with _lock, _conn() as conn:
        rows = conn.execute("SELECT hits_json, created_at FROM qa_log").fetchall()
    doc_stats = {}  # doc_name -> {hits, score_sum, last_hit}
    for r in rows:
        try:
            hits = json.loads(r["hits_json"] or "[]")
        except Exception:
            continue
        for h in hits:
            name = h.get("doc", "")
            if not name or name == "FAQ":
                continue
            st = doc_stats.setdefault(name, {"hits": 0, "score_sum": 0.0, "last_hit": 0})
            st["hits"] += 1
            st["score_sum"] += float(h.get("score", 0))
            st["last_hit"] = max(st["last_hit"], r["created_at"])
    return doc_stats


# ---------- 链路追踪 ----------
def add_trace(trace_id, session_id, question, answer, total_ms, stages, error=""):
    """记录一次问答的完整链路信息"""
    with _lock, _conn() as conn:
        _ensure_traces_table(conn)
        conn.execute(
            "INSERT INTO trace_logs (trace_id, session_id, question, answer, total_ms, stages_json, error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (trace_id, session_id, question, answer, total_ms,
             json.dumps(stages, ensure_ascii=False), error, time.time()),
        )


def _ensure_traces_table(conn):
    """老库兼容：无 trace_logs 表时自动创建"""
    exists = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='trace_logs'"
    ).fetchone()
    if not exists:
        conn.executescript("""
            CREATE TABLE trace_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trace_id TEXT NOT NULL,
                session_id TEXT,
                question TEXT NOT NULL,
                answer TEXT DEFAULT '',
                total_ms INTEGER DEFAULT 0,
                stages_json TEXT DEFAULT '[]',
                error TEXT DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trace_logs_created ON trace_logs(created_at DESC);
        """)


def list_traces(limit=50, page=1, page_size=None):
    """获取链路记录列表

    page_size=None 时保持旧行为（limit 条 list）；否则返回 {items, total, page, page_size, pages}
    """
    with _lock, _conn() as conn:
        _ensure_traces_table(conn)
        base_sql = "SELECT trace_id, session_id, question, answer, total_ms, error, created_at FROM trace_logs"
        count_sql = "SELECT COUNT(*) AS c FROM trace_logs"

        if page_size is not None:
            total = conn.execute(count_sql).fetchone()["c"]
            offset = (page - 1) * page_size
            rows = conn.execute(
                base_sql + " ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (page_size, offset),
            ).fetchall()
            items = [
                {
                    "trace_id": r["trace_id"],
                    "session_id": r["session_id"],
                    "question": r["question"],
                    "answer": r["answer"],
                    "total_ms": r["total_ms"],
                    "error": r["error"],
                    "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"])),
                }
                for r in rows
            ]
            pages = max(1, (total + page_size - 1) // page_size)
            return {"items": items, "total": total, "page": page, "page_size": page_size, "pages": pages}
        else:
            rows = conn.execute(
                base_sql + " ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
    return [
        {
            "trace_id": r["trace_id"],
            "session_id": r["session_id"],
            "question": r["question"],
            "answer": r["answer"],
            "total_ms": r["total_ms"],
            "error": r["error"],
            "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"])),
        }
        for r in rows
    ]


def get_trace(trace_id):
    """获取单条链路详情（含完整环节数据）"""
    with _lock, _conn() as conn:
        _ensure_traces_table(conn)
        row = conn.execute(
            "SELECT * FROM trace_logs WHERE trace_id = ?", (trace_id,)
        ).fetchone()
    if not row:
        return None
    return {
        "trace_id": row["trace_id"],
        "session_id": row["session_id"],
        "question": row["question"],
        "answer": row["answer"],
        "total_ms": row["total_ms"],
        "stages": json.loads(row["stages_json"] or "[]"),
        "error": row["error"],
        "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["created_at"])),
    }


# ---------- Agent 管理 ----------
def _ensure_agents_builtin():
    """首次启动时写入 3 个内置 Agent（内置的 tools 会以 enabled=0 展示，用户可编辑）"""
    with _lock, _conn() as conn:
        count = conn.execute("SELECT COUNT(*) c FROM agents WHERE builtin=1").fetchone()["c"]
        if count > 0:
            return
        conn.executemany(
            """INSERT INTO agents (name, description, enabled, endpoint, auth_type, schema_json, builtin, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                ("FAQ 检索", "从 FAQ 表匹配常见问题答案", 1, "builtin://faq", "none", '{"kind":"faq"}', 1, time.time()),
                ("创建工单", "用户投诉/售后/退换时创建工单交由人工处理", 1, "builtin://ticket", "none", '{"kind":"ticket"}', 1, time.time()),
                ("转人工客服", "用户主动要求转接真人客服", 1, "builtin://handoff", "none", '{"kind":"handoff"}', 1, time.time()),
            ],
        )


def list_agents(page=1, page_size=None):
    """page_size=None 时保持旧行为返回 list；否则返回分页 dict"""
    _ensure_agents_builtin()
    sql = "SELECT * FROM agents ORDER BY builtin DESC, id ASC"
    with _lock, _conn() as conn:
        if page_size is None:
            rows = conn.execute(sql).fetchall()
            return [
                {"id": r["id"], "name": r["name"], "description": r["description"],
                 "enabled": bool(r["enabled"]), "endpoint": r["endpoint"],
                 "auth_type": r["auth_type"], "auth_value": r["auth_value"],
                 "schema_json": json.loads(r["schema_json"] or "{}"),
                 "builtin": bool(r["builtin"])}
                for r in rows
            ]
        ps = max(1, int(page_size))
        p = max(1, int(page))
        total = conn.execute("SELECT COUNT(1) AS c FROM agents").fetchone()["c"]
        pages = max(1, (total + ps - 1) // ps)
        p = min(p, pages)
        offset = (p - 1) * ps
        rows = conn.execute(sql + " LIMIT ? OFFSET ?", (ps, offset)).fetchall()
        items = [
            {"id": r["id"], "name": r["name"], "description": r["description"],
             "enabled": bool(r["enabled"]), "endpoint": r["endpoint"],
             "auth_type": r["auth_type"], "auth_value": r["auth_value"],
             "schema_json": json.loads(r["schema_json"] or "{}"),
             "builtin": bool(r["builtin"])}
            for r in rows
        ]
        return {"items": items, "total": total, "page": p, "page_size": ps, "pages": pages}


def create_agent(name, description="", endpoint="", auth_type="none", auth_value="", schema=None):
    _ensure_agents_builtin()
    with _lock, _conn() as conn:
        cur = conn.execute(
            """INSERT INTO agents (name, description, endpoint, auth_type, auth_value, schema_json, builtin, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 0, ?)""",
            (name, description, endpoint, auth_type, auth_value,
             json.dumps(schema or {}, ensure_ascii=False), time.time()),
        )
        return cur.lastrowid


def update_agent(agent_id, name=None, description=None, endpoint=None,
                 auth_type=None, auth_value=None, schema=None):
    fields, args = [], []
    for key, val in [
        ("name", name), ("description", description), ("endpoint", endpoint),
        ("auth_type", auth_type), ("auth_value", auth_value),
    ]:
        if val is not None:
            fields.append(f"{key} = ?")
            args.append(val)
    if schema is not None:
        fields.append("schema_json = ?")
        args.append(json.dumps(schema, ensure_ascii=False))
    if not fields:
        return
    args.append(agent_id)
    with _lock, _conn() as conn:
        conn.execute(f"UPDATE agents SET {', '.join(fields)} WHERE id = ?", args)


def toggle_agent(agent_id, enabled):
    with _lock, _conn() as conn:
        conn.execute("UPDATE agents SET enabled = ? WHERE id = ?", (1 if enabled else 0, agent_id))


def delete_agent(agent_id):
    with _lock, _conn() as conn:
        conn.execute("DELETE FROM agents WHERE id = ? AND builtin = 0", (agent_id,))


# ---------- 用户 & 会话（RBAC 基础） ----------
def _row_to_user(row) -> dict | None:
    if not row:
        return None
    d = dict(row)
    d.pop("password", None)  # 永不返回密码哈希
    d["enabled"] = bool(d.get("enabled", 1))
    return d


def get_user_by_username(username: str) -> dict | None:
    """供 auth.authenticate 内部使用：返回含 password 字段的原始行"""
    with _lock, _conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def get_user_by_id(user_id: int) -> dict | None:
    with _lock, _conn() as conn:
        row = conn.execute(
            "SELECT id, username, role, enabled, created_at FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
    return _row_to_user(row)


def list_users() -> list[dict]:
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT id, username, role, enabled, failed_attempts, lockout_until, org_id, dept_id, created_at FROM users ORDER BY id ASC"
        ).fetchall()
    now = time.time()
    out = []
    for r in rows:
        d = dict(r)
        # 计算友好的时间格式和锁定状态
        d["created_at"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(d["created_at"]))
        d["enabled"] = bool(d.get("enabled", 1))
        d["failed_attempts"] = d.get("failed_attempts", 0) or 0
        d["locked"] = bool(d["lockout_until"] and d["lockout_until"] > now)
        if d["lockout_until"] and d["lockout_until"] > now:
            remaining = int(d["lockout_until"] - now)
            d["lockout_remaining"] = remaining
            d["lockout_until_str"] = time.strftime("%H:%M:%S", time.localtime(d["lockout_until"]))
        else:
            d["lockout_remaining"] = 0
            d["lockout_until_str"] = ""
        d.pop("lockout_until", None)
        out.append(d)
    return out


def create_user(username: str, password: str, role: str = "user") -> int | None:
    import bcrypt as _bcrypt
    try:
        with _lock, _conn() as conn:
            # 唯一约束
            existing = conn.execute(
                "SELECT 1 FROM users WHERE username = ?", (username,)
            ).fetchone()
            if existing:
                return None
            hashed = _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")
            cur = conn.execute(
                """INSERT INTO users (username, password, role, enabled, created_at)
                   VALUES (?, ?, ?, 1, ?)""",
                (username, hashed, role, time.time()),
            )
            return cur.lastrowid
    except Exception:
        return None


def update_user(user_id: int, **fields) -> bool:
    """更新用户字段：username / role / enabled / password"""
    allowed = {"username", "role", "enabled", "password"}
    updates = []
    args = []
    for k, v in fields.items():
        if k not in allowed or v is None:
            continue
        if k == "password":
            import bcrypt as _bcrypt
            v = _bcrypt.hashpw(v.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")
        elif k == "enabled":
            v = 1 if v else 0
        updates.append(f"{k} = ?")
        args.append(v)
    if not updates:
        return False
    args.append(user_id)
    with _lock, _conn() as conn:
        conn.execute(f"UPDATE users SET {', '.join(updates)} WHERE id = ?", args)
    return True


def delete_user(user_id: int) -> bool:
    # 禁止删 superadmin
    with _lock, _conn() as conn:
        row = conn.execute("SELECT role FROM users WHERE id = ?", (user_id,)).fetchone()
        if row and row["role"] == "superadmin":
            return False
        conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
    return True


def toggle_user(user_id: int, enabled: bool) -> bool:
    return update_user(user_id, enabled=enabled)


# ============================================================
# 角色管理
# 角色元数据存 SQLite（本文件 roles 表），权限明细存 PostgreSQL
# role_permissions 表，两者以角色名 name 关联。
# ============================================================

def list_roles() -> list[dict]:
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT name, display_name, description, level, builtin, enabled, created_at "
            "FROM roles ORDER BY level DESC, name ASC"
        ).fetchall()
        counts = {r["role"]: r["c"] for r in
                  conn.execute("SELECT role, COUNT(*) c FROM users GROUP BY role").fetchall()}
    out = []
    for r in rows:
        d = dict(r)
        d["builtin"] = bool(d["builtin"])
        d["enabled"] = bool(d["enabled"])
        d["user_count"] = counts.get(d["name"], 0)
        out.append(d)
    return out


def get_role(name: str) -> dict | None:
    with _lock, _conn() as conn:
        row = conn.execute("SELECT * FROM roles WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None


def create_role(name: str, display_name: str, description: str = "", level: int = 10) -> bool:
    try:
        with _lock, _conn() as conn:
            conn.execute(
                """INSERT INTO roles (name, display_name, description, level, builtin, enabled, created_at)
                   VALUES (?, ?, ?, ?, 0, 1, ?)""",
                (name, display_name, description, level, time.time()),
            )
        return True
    except Exception:
        return False


def update_role(name: str, display_name=None, description=None, level=None, enabled=None) -> bool:
    sets, args = [], []
    for k, v in (("display_name", display_name), ("description", description),
                 ("level", level), ("enabled", enabled)):
        if v is None:
            continue
        if k == "enabled":
            v = 1 if v else 0
        sets.append(f"{k} = ?")
        args.append(v)
    if not sets:
        return False
    args.append(name)
    with _lock, _conn() as conn:
        conn.execute(f"UPDATE roles SET {', '.join(sets)} WHERE name = ?", args)
    return True


def delete_role(name: str) -> tuple[bool, str]:
    """删除角色：内置角色、仍被用户引用的角色不允许删除"""
    r = get_role(name)
    if not r:
        return False, "角色不存在"
    if r.get("builtin"):
        return False, "内置角色不可删除"
    with _lock, _conn() as conn:
        c = conn.execute("SELECT COUNT(*) c FROM users WHERE role = ?", (name,)).fetchone()["c"]
        if c:
            return False, f"仍有 {c} 个用户使用该角色，请先调整这些用户的角色"
        conn.execute("DELETE FROM roles WHERE name = ?", (name,))
    return True, ""


def assign_user_org(user_id: int, org_id, dept_id) -> bool:
    """给用户分配组织与部门（org_id/dept_id 传 None 表示清除）"""
    with _lock, _conn() as conn:
        if not conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone():
            return False
        conn.execute("UPDATE users SET org_id = ?, dept_id = ? WHERE id = ?",
                     (org_id, dept_id, user_id))
    return True


def change_password(user_id: int, new_password: str) -> bool:
    return update_user(user_id, password=new_password)


# ---------- 服务端 session 白名单 ----------
def create_user_session(user_id: int, session_id: str, ttl_seconds: int) -> None:
    with _lock, _conn() as conn:
        conn.execute(
            """INSERT INTO user_sessions (session_id, user_id, revoked, exp_at, created_at)
               VALUES (?, ?, 0, ?, ?)""",
            (session_id, user_id, time.time() + ttl_seconds, time.time()),
        )


def revoke_user_session(session_id: str) -> None:
    with _lock, _conn() as conn:
        conn.execute(
            "UPDATE user_sessions SET revoked = 1 WHERE session_id = ?", (session_id,)
        )


def revoke_all_user_sessions(user_id: int) -> int:
    """强制下线：把该用户所有未 revoke 的 session 置为 revoked=1，返回数量"""
    with _lock, _conn() as conn:
        cur = conn.execute(
            "UPDATE user_sessions SET revoked = 1 WHERE user_id = ? AND revoked = 0",
            (user_id,)
        )
        # 顺便清理过期的
        conn.execute(
            "DELETE FROM user_sessions WHERE exp_at < ?", (time.time(),)
        )
        return cur.rowcount or 0


def is_session_valid(session_id: str) -> bool:
    """JWT 验证时调：存在 + 未 revoke + 未过期"""
    if not session_id:
        return False
    with _lock, _conn() as conn:
        row = conn.execute(
            "SELECT revoked, exp_at FROM user_sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
    if not row:
        return False
    if row["revoked"]:
        return False
    if row["exp_at"] < time.time():
        # 过期顺手清掉
        with _lock, _conn() as conn:
            conn.execute("DELETE FROM user_sessions WHERE session_id = ?", (session_id,))
        return False
    return True


def list_user_sessions(user_id: int) -> list[dict]:
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT session_id, revoked, exp_at, created_at FROM user_sessions "
            "WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
    now = time.time()
    return [
        {
            "session_id": r["session_id"],
            "revoked": bool(r["revoked"]),
            "expired": r["exp_at"] < now,
            "exp_at": r["exp_at"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


# ---------- 安全审计日志 ----------
def write_audit(action: str, resource_type: str = "", resource_id: str = "",
                actor_user_id: int | None = None, actor_username: str | None = None,
                ip: str = "", user_agent: str = "", result: str = "success",
                detail: str = "") -> None:
    """写一条审计日志（轻量同步写，审计是安全关键不能丢）"""
    with _lock, _conn() as conn:
        conn.execute(
            """INSERT INTO audit_logs
               (timestamp, actor_user_id, actor_username, action, resource_type, resource_id,
                ip, user_agent, result, detail)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), actor_user_id, actor_username, action, resource_type, resource_id,
             ip, user_agent, result, detail),
        )
        # 自动清理 90 天前的旧审计日志（保守策略）
        cutoff = time.time() - 90 * 86400
        conn.execute("DELETE FROM audit_logs WHERE timestamp < ?", (cutoff,))


def list_audit_logs(
    limit: int = 200, action: str = "", actor: str = "", keyword: str = "",
    page: int = 1, page_size: int | None = None,
) -> list[dict] | dict:
    """返回审计日志列表。

    兼容两种调用：
    - 只传 limit：保持原行为，直接返回 list
    - 传 page + page_size：返回 { items, total, page, page_size, pages } 分页 dict
    """
    sql = "SELECT * FROM audit_logs WHERE 1=1"
    params: list = []
    if action:
        sql += " AND action LIKE ?"; params.append(f"%{action}%")
    if actor:
        sql += " AND (actor_username LIKE ? OR actor_user_id = ?)"; params.extend([f"%{actor}%", actor if actor.isdigit() else -1])
    if keyword:
        sql += " AND (detail LIKE ? OR action LIKE ? OR resource_type LIKE ?)"; params.extend([f"%{keyword}%"] * 3)

    with _lock, _conn() as conn:
        if page_size is None:
            rows = conn.execute(sql + " ORDER BY timestamp DESC LIMIT ?", params + [limit]).fetchall()
            return _audit_rows_to_dicts(rows)

        count_sql = "SELECT COUNT(*) c FROM audit_logs WHERE 1=1" + sql.split("WHERE 1=1", 1)[1]
        total = conn.execute(count_sql, params).fetchone()["c"]
        ps = max(1, page_size)
        p = max(1, page)
        offset = (p - 1) * ps
        rows = conn.execute(sql + " ORDER BY timestamp DESC LIMIT ? OFFSET ?", params + [ps, offset]).fetchall()

    items = _audit_rows_to_dicts(rows)
    import math as _m
    return {
        "items": items,
        "total": total,
        "page": p,
        "page_size": ps,
        "pages": max(1, _m.ceil(total / ps)),
    }


def _audit_rows_to_dicts(rows):
    return [
        {
            "id": r["id"],
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["timestamp"])),
            "actor_user_id": r["actor_user_id"],
            "actor_username": r["actor_username"] or "(未登录)",
            "action": r["action"],
            "resource_type": r["resource_type"] or "",
            "resource_id": r["resource_id"] or "",
            "ip": r["ip"] or "",
            "user_agent": (r["user_agent"] or "")[:60],
            "result": r["result"],
            "detail": r["detail"] or "",
        }
        for r in rows
    ]


# ---------- 登录失败 & 锁定 ----------
def check_account_locked(username: str) -> bool:
    """返回该账号是否处于锁定状态"""
    with _lock, _conn() as conn:
        row = conn.execute(
            "SELECT lockout_until, failed_attempts FROM users WHERE username = ?",
            (username,)
        ).fetchone()
    if not row:
        return False
    if row["lockout_until"] and row["lockout_until"] > time.time():
        return True
    # 过期自动解锁
    if row["lockout_until"] and row["lockout_until"] <= time.time():
        reset_failed_attempts(username)
    return False


def record_login_failure(username: str, ip: str = "", user_agent: str = "") -> dict:
    """记录一次登录失败 + 累加 failed_attempts，超过阈值自动锁定
    返回 {"locked": bool, "remaining": int, "lockout_seconds": int}
    """
    MAX_ATTEMPTS = 5        # 5 次失败触发锁定
    LOCKOUT_MINUTES = 10    # 锁 10 分钟

    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO auth_failures (username, ip, user_agent, timestamp) VALUES (?, ?, ?, ?)",
            (username, ip, user_agent, time.time()),
        )
        row = conn.execute(
            "SELECT id, failed_attempts FROM users WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return {"locked": False, "remaining": MAX_ATTEMPTS, "lockout_seconds": 0}

        new_count = (row["failed_attempts"] or 0) + 1
        if new_count >= MAX_ATTEMPTS:
            lockout_until = time.time() + LOCKOUT_MINUTES * 60
            conn.execute(
                "UPDATE users SET failed_attempts = ?, lockout_until = ? WHERE id = ?",
                (new_count, lockout_until, row["id"]),
            )
            return {"locked": True, "remaining": 0, "lockout_seconds": LOCKOUT_MINUTES * 60}
        else:
            conn.execute(
                "UPDATE users SET failed_attempts = ? WHERE id = ?",
                (new_count, row["id"]),
            )
            return {"locked": False, "remaining": MAX_ATTEMPTS - new_count, "lockout_seconds": 0}


def reset_failed_attempts(username: str) -> None:
    """登录成功时清零"""
    with _lock, _conn() as conn:
        conn.execute(
            "UPDATE users SET failed_attempts = 0, lockout_until = NULL WHERE username = ?",
            (username,)
        )
        # 顺便清该用户 24 小时前的 auth_failures 记录（防表无限膨胀）
        cutoff = time.time() - 86400
        conn.execute(
            "DELETE FROM auth_failures WHERE username = ? AND timestamp < ?",
            (username, cutoff)
        )


def list_auth_failures(
    username: str = "", ip: str = "", limit: int = 100,
    page: int = 1, page_size: int | None = None,
) -> list[dict] | dict:
    sql = "SELECT * FROM auth_failures WHERE 1=1"
    params: list = []
    if username:
        sql += " AND username = ?"; params.append(username)
    if ip:
        sql += " AND ip = ?"; params.append(ip)

    with _lock, _conn() as conn:
        if page_size is None:
            rows = conn.execute(sql + " ORDER BY timestamp DESC LIMIT ?", params + [limit]).fetchall()
            return _failure_rows_to_dicts(rows)

        count_sql = "SELECT COUNT(*) c FROM auth_failures WHERE 1=1" + sql.split("WHERE 1=1", 1)[1]
        total = conn.execute(count_sql, params).fetchone()["c"]
        ps = max(1, page_size)
        p = max(1, page)
        rows = conn.execute(
            sql + " ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            params + [ps, (p - 1) * ps],
        ).fetchall()

    import math as _m
    return {
        "items": _failure_rows_to_dicts(rows),
        "total": total,
        "page": p, "page_size": ps,
        "pages": max(1, _m.ceil(total / ps)),
    }


def _failure_rows_to_dicts(rows):
    return [
        {
            "id": r["id"],
            "username": r["username"],
            "ip": r["ip"] or "",
            "user_agent": (r["user_agent"] or "")[:60],
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["timestamp"])),
        }
        for r in rows
    ]


# ── 条件覆盖：核心函数走 PostgreSQL ORM（如果 DB_DRIVER=postgresql） ──
# 注意：这块必须放在文件末尾，否则会被上面的 def 定义覆盖
try:
    from core.config import get_settings as _get_settings
    if _get_settings().DB_DRIVER == "postgresql":
        from db_pg import (
            get_user_by_username, get_user_by_id, list_users,
            create_user, update_user, delete_user, toggle_user, change_password, list_roles,
            create_session, list_sessions, rename_session, delete_session,
            add_message, get_messages,
            create_kb, get_kb, list_kb, delete_kb, update_kb,
            write_audit, list_audit_logs,
            create_user_session, revoke_user_session, revoke_all_user_sessions,
            is_session_valid, check_account_locked, record_login_failure,
            reset_failed_attempts, list_auth_failures,
        )
        print("[db] 核心函数已切换到 PostgreSQL ORM 实现")
except Exception as _e:
    import traceback
    print(f"[db] PG 覆盖跳过（{_e}），保持 SQLite 实现")
    traceback.print_exc()
