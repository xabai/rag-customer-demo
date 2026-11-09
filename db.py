# -*- coding: utf-8 -*-
"""SQLite 存储：多会话与消息历史"""
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
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                kind TEXT NOT NULL,
                content TEXT NOT NULL,
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
        """)
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
def create_session(title="新对话"):
    sid = uuid.uuid4().hex[:12]
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
            (sid, title, time.time()),
        )
    return {"id": sid, "title": title}


def list_sessions():
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


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


def list_unanswered():
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT id, question, created_at FROM unanswered ORDER BY id DESC"
        ).fetchall()
    return [
        {"id": r["id"], "question": r["question"],
         "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}
        for r in rows
    ]


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


def list_faq():
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT id, question, answer FROM faq ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_faq(item_id):
    with _lock, _conn() as conn:
        conn.execute("DELETE FROM faq WHERE id = ?", (item_id,))


# ---------- 意图规则 ----------
def list_intent_rules(enabled_only=False):
    sql = "SELECT * FROM intent_rules"
    if enabled_only:
        sql += " WHERE enabled = 1"
    sql += " ORDER BY id"
    with _lock, _conn() as conn:
        rows = conn.execute(sql).fetchall()
    return [dict(r) for r in rows]


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


def list_tickets():
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM tickets ORDER BY id DESC"
        ).fetchall()
    return [
        {"id": r["id"], "question": r["question"], "intent": r["intent"],
         "status": r["status"], "session_id": r["session_id"],
         "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}
        for r in rows
    ]


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


def list_handoffs():
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM handoffs ORDER BY id DESC"
        ).fetchall()
    return [
        {"id": r["id"], "question": r["question"], "status": r["status"],
         "session_id": r["session_id"],
         "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}
        for r in rows
    ]


def update_handoff_status(handoff_id, status):
    with _lock, _conn() as conn:
        conn.execute("UPDATE handoffs SET status = ? WHERE id = ?", (status, handoff_id))


# ---------- 知识库分组 ----------
def create_kb(name, description=""):
    kid = f"kb_{uuid.uuid4().hex[:8]}"
    with _lock, _conn() as conn:
        conn.execute(
            "INSERT INTO knowledge_bases (id, name, description, created_at) VALUES (?, ?, ?, ?)",
            (kid, name, description, time.time()),
        )
    return {"id": kid, "name": name, "description": description}


def list_kb():
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT id, name, description, created_at FROM knowledge_bases ORDER BY created_at ASC"
        ).fetchall()
    return [
        {"id": r["id"], "name": r["name"], "description": r["description"],
         "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}
        for r in rows
    ]


def delete_kb(kid):
    with _lock, _conn() as conn:
        conn.execute("DELETE FROM knowledge_bases WHERE id = ?", (kid,))


# ---------- 记忆（摘要 + 跨会话偏好） ----------
def add_memory(session_id, kind, content):
    with _lock, _conn() as conn:
        cur = conn.execute(
            "INSERT INTO memories (session_id, kind, content, created_at) VALUES (?, ?, ?, ?)",
            (session_id, kind, content, time.time()),
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


def list_cross_memories():
    """跨会话记忆（kind=preference/fact），注入到所有对话"""
    with _lock, _conn() as conn:
        rows = conn.execute(
            "SELECT id, content, created_at FROM memories WHERE kind IN ('preference','fact') ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def list_memories(session_id=None):
    sql = "SELECT id, session_id, kind, content, created_at FROM memories"
    if session_id:
        sql += " WHERE session_id = ?"
    sql += " ORDER BY id DESC"
    params = (session_id,) if session_id else ()
    with _lock, _conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [
        {"id": r["id"], "session_id": r["session_id"], "kind": r["kind"],
         "content": r["content"],
         "created": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["created_at"]))}
        for r in rows
    ]


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


def list_qa_log(session_id="", limit=100):
    with _lock, _conn() as conn:
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


def list_eval_cases():
    with _lock, _conn() as conn:
        rows = conn.execute("SELECT * FROM eval_cases ORDER BY id DESC").fetchall()
    return [dict(r) for r in rows]


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


def list_traces(limit=50):
    """获取最近的链路记录"""
    with _lock, _conn() as conn:
        _ensure_traces_table(conn)
        rows = conn.execute(
            "SELECT trace_id, session_id, question, answer, total_ms, error, created_at "
            "FROM trace_logs ORDER BY created_at DESC LIMIT ?",
            (limit,),
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
