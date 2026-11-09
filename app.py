# -*- coding: utf-8 -*-
"""FastAPI 入口：知识库管理 + 多会话流式聊天"""
import json
import os
import re
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
import db
from core import chunker, intent, memory, ollama_client, parser, rag, settings, tools
from core.vector_store import store

app = FastAPI(title="RAG AI 客服")
db.init_db()

# 挂载知识库图片目录（前端用 /kb_images/xxx.png 直接引用）
from fastapi.staticfiles import StaticFiles as _SF
if os.path.isdir(config.KB_IMG_DIR):
    app.mount("/kb_images", _SF(directory=config.KB_IMG_DIR), name="kb_images")


# ---------- 知识库 ----------
@app.post("/api/upload")
async def upload(file: UploadFile = File(...), kb_id: str = Form("default"),
                 chunk_size: int = Form(0), chunk_overlap: int = Form(-1),
                 chunk_mode: str = Form("flat")):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in config.ALLOWED_EXT:
        raise HTTPException(400, f"不支持的文件类型 {ext}，支持: {', '.join(sorted(config.ALLOWED_EXT))}")

    path = os.path.join(config.UPLOAD_DIR, file.filename)
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 优先走版面感知管线（元素归一化 + 图文配对 + 版面分块）
        use_layout = ext in (".pdf", ".pptx", ".ppt", ".docx", ".xlsx", ".xls")
        if use_layout:
            try:
                elements = parser.parse_file_layout(path)
                chunks = parser.build_layout_chunks(elements)
                chunk_texts, parent_texts, chunk_images, layout_meta = \
                    parser.layout_chunks_to_records(chunks)
                if not chunk_texts:
                    # layout 没产出 → 降级到旧管线
                    raise ValueError("layout 解析为空，降级")
            except Exception as layout_err:
                print(f"[upload] layout 管线失败，降级: {layout_err}")
                use_layout = False

        if not use_layout:
            text = parser.parse_file(path)
            cs = chunk_size if chunk_size > 0 else None
            co = chunk_overlap if chunk_overlap >= 0 else None
            if chunk_mode == "parent_child":
                tree = chunker.split_parent_child(text, chunk_overlap=co)
                chunk_texts, parent_texts = [], []
                for node in tree:
                    for child in node["children"]:
                        chunk_texts.append(child)
                        parent_texts.append(node["parent"])
            else:
                chunk_texts = chunker.split_text(text, chunk_size=cs, chunk_overlap=co)
                parent_texts = None
            chunk_images = None
            layout_meta = None

        if not chunk_texts:
            raise ValueError("未从文件中解析出有效文本")

        # 文本向量（bge-m3）
        vectors = ollama_client.embed_texts(chunk_texts)

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

        doc_id = store.add_document(
            file.filename, chunk_texts, vectors, kb_id=kb_id,
            parent_texts=parent_texts, chunk_images=chunk_images,
            layout_meta=layout_meta, image_vectors=image_vectors,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        os.remove(path)
        raise HTTPException(500, f"文档处理失败: {e}")

    return {"id": doc_id, "name": file.filename, "chunks": len(chunk_texts),
            "kb_id": kb_id, "layout": use_layout}


@app.get("/api/documents")
async def list_documents():
    return store.list_documents()


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    if not store.delete_document(doc_id):
        raise HTTPException(404, "文档不存在")
    return {"ok": True}


@app.get("/api/documents/{doc_id}/chunks")
async def get_chunks(doc_id: str):
    """查看文档的所有分块内容"""
    chunks = store.get_document_chunks(doc_id)
    if not chunks:
        raise HTTPException(404, "文档不存在或无分块")
    return chunks


class ChunkUpdate(BaseModel):
    text: str


@app.put("/api/documents/{doc_id}/chunks/{chunk_index}")
async def update_chunk(doc_id: str, chunk_index: int, body: ChunkUpdate):
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
async def upload_url(body: UrlImportBody):
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
async def retrieval_test(body: RetrievalTestBody):
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
async def create_kb(body: KbBody):
    return db.create_kb(body.name, body.description)


@app.get("/api/kb")
async def list_kb():
    return db.list_kb()


@app.delete("/api/kb/{kb_id}")
async def delete_kb(kb_id: str):
    db.delete_kb(kb_id)
    return {"ok": True}


# ---------- 会话 ----------
@app.get("/api/sessions")
async def list_sessions():
    return db.list_sessions()


@app.post("/api/sessions")
async def create_session():
    return db.create_session()


@app.get("/api/sessions/{session_id}/messages")
async def get_messages(session_id: str):
    return db.get_messages(session_id)


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    db.delete_session(session_id)
    return {"ok": True}


# ---------- 设置（提示词 + 敏感词 + 检索开关） ----------
class SettingsBody(BaseModel):
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
async def get_settings():
    return settings.load_settings()


@app.put("/api/settings")
async def update_settings(body: SettingsBody):
    return settings.save_settings(body.model_dump())


# ---------- 数据存储目录 ----------
class DataDirBody(BaseModel):
    path: str


@app.get("/api/settings/data_dir")
async def get_data_dir():
    return {"data_dir": config.DATA_DIR,
            "default_dir": os.path.join(config.BASE_DIR, "data")}


@app.post("/api/settings/migrate_data")
async def migrate_data(body: DataDirBody):
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
async def list_faq():
    return db.list_faq()


@app.post("/api/faq")
async def add_faq(body: FaqBody):
    db.add_faq(body.question, body.answer)
    return {"ok": True}


@app.delete("/api/faq/{item_id}")
async def delete_faq(item_id: int):
    db.delete_faq(item_id)
    return {"ok": True}


# ---------- 意图规则管理 ----------
class IntentRuleBody(BaseModel):
    name: str
    keywords: str
    action: str
    reply: str = ""


@app.get("/api/intent-rules")
async def list_intent_rules():
    return db.list_intent_rules()


@app.post("/api/intent-rules")
async def add_intent_rule(body: IntentRuleBody):
    db.add_intent_rule(body.name, body.keywords, body.action, body.reply)
    return {"ok": True}


@app.delete("/api/intent-rules/{rule_id}")
async def delete_intent_rule(rule_id: int):
    db.delete_intent_rule(rule_id)
    return {"ok": True}


@app.put("/api/intent-rules/{rule_id}/toggle")
async def toggle_intent_rule(rule_id: int, body: dict):
    db.toggle_intent_rule(rule_id, 1 if body.get("enabled") else 0)
    return {"ok": True}


# ---------- 工单管理 ----------
@app.get("/api/tickets")
async def list_tickets():
    return db.list_tickets()


@app.put("/api/tickets/{ticket_id}/status")
async def update_ticket_status(ticket_id: int, body: dict):
    db.update_ticket_status(ticket_id, body.get("status", "open"))
    return {"ok": True}


# ---------- 转人工管理 ----------
@app.get("/api/handoffs")
async def list_handoffs():
    return db.list_handoffs()


@app.put("/api/handoffs/{handoff_id}/status")
async def update_handoff_status(handoff_id: int, body: dict):
    db.update_handoff_status(handoff_id, body.get("status", "pending"))
    return {"ok": True}


# ---------- 未命中问题沉淀 ----------
@app.get("/api/unanswered")
async def list_unanswered():
    return db.list_unanswered()


@app.delete("/api/unanswered/{item_id}")
async def delete_unanswered(item_id: int):
    db.delete_unanswered(item_id)
    return {"ok": True}


# ---------- 用户反馈 ----------
class FeedbackBody(BaseModel):
    session_id: str
    message_id: int
    rating: int  # 1=赞, -1=踩
    comment: str = ""


@app.post("/api/feedback")
async def add_feedback(body: FeedbackBody):
    db.add_feedback(body.session_id, body.message_id, body.rating, body.comment)
    return {"ok": True}


@app.get("/api/feedback")
async def list_feedback():
    return db.list_feedback()


# ---------- 统计看板 ----------
@app.get("/api/stats")
async def get_stats():
    return db.get_stats()


@app.get("/api/stats/trend")
async def get_trend(days: int = 14):
    """看板图表数据：每日问答量、每日反馈、问答来源分布、耗时统计"""
    return {
        "daily_qa": db.get_daily_qa(days),
        "daily_feedback": db.get_daily_feedback(days),
        "qa_source": db.get_qa_source_distribution(),
        "latency": db.get_latency_stats(days),
    }


# ---------- 问答日志 ----------
@app.get("/api/qa_log")
async def get_qa_log(session_id: str = "", limit: int = 100):
    return db.list_qa_log(session_id, limit)


@app.get("/api/qa_log/export")
async def export_badcases():
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
async def export_session(session_id: str):
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
async def list_eval_cases():
    return db.list_eval_cases()


@app.post("/api/eval/cases")
async def add_eval_case(body: EvalCaseBody):
    return {"id": db.add_eval_case(body.question, body.expected, "manual", body.kb_id)}


@app.post("/api/eval/cases/from_log")
async def add_eval_case_from_log(body: dict):
    """从问答日志/badcase 一键加入评测集"""
    return {"id": db.add_eval_case(body["question"], body.get("expected", body.get("answer", "")),
                                   "badcase", body.get("kb_id", ""))}


@app.put("/api/eval/cases/{case_id}")
async def update_eval_case(case_id: int, body: EvalCaseBody):
    db.update_eval_case(case_id, body.question, body.expected)
    return {"ok": True}


@app.delete("/api/eval/cases/{case_id}")
async def delete_eval_case(case_id: int):
    db.delete_eval_case(case_id)
    return {"ok": True}


@app.post("/api/eval/run")
async def run_eval():
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
async def kb_health():
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


# ---------- 图片知识库 ----------
@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...), kb_id: str = Form("default")):
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
async def get_kb_image(img_name: str):
    """获取知识库图片（前端预览用）"""
    # 防路径穿越
    safe = os.path.basename(img_name)
    path = os.path.join(config.KB_IMG_DIR, safe)
    if not os.path.exists(path):
        raise HTTPException(404, "图片不存在")
    return FileResponse(path)


# ---------- 记忆管理 ----------
@app.get("/api/memories")
async def list_memories(session_id: str = ""):
    return db.list_memories(session_id or None)


@app.delete("/api/memories/{mem_id}")
async def delete_memory(mem_id: int):
    db.delete_memory(mem_id)
    return {"ok": True}


# ---------- 聊天（SSE 流式） ----------
class ChatRequest(BaseModel):
    session_id: str
    message: str
    model: str = ""      # 为空则用默认模型
    use_kb: bool = True  # 是否检索知识库（关闭则直接与大模型对话）
    kb_id: str | None = ""  # 指定知识库，空则全部
    use_tools: bool = True  # 是否启用 Agent 工具调用
    enabled_tools: list[str] | None = None  # 允许使用的工具名列表，None 表示全部可用
    images: list[str] = []  # base64 编码图片（多模态），有图时自动切换视觉模型


@app.get("/api/tools")
async def list_tools():
    """返回可调用的 Agent 工具清单（供前端工具选择面板展示）"""
    return [{"name": name,
             "label": spec.get("label", name),
             "description": spec["description"]}
            for name, spec in tools.TOOLS.items()]


@app.get("/api/models")
async def list_models():
    """读取 Ollama 已安装模型（过滤掉 embedding 模型）"""
    import requests as http

    try:
        r = http.get(f"{config.OLLAMA_BASE}/api/tags", timeout=10)
        r.raise_for_status()
    except Exception:
        raise HTTPException(502, "无法连接 Ollama，请确认服务已启动")
    models = [
        m["name"] for m in r.json().get("models", [])
        if "embedding" not in m.get("capabilities", [])
    ]
    return models


# 敏感词命中时的固定回复
SENSITIVE_REPLY = "抱歉，您的问题包含敏感内容，无法回答。请换个问题试试。"


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
async def chat(req: ChatRequest):
    question = req.message.strip()
    if not question:
        raise HTTPException(400, "消息不能为空")

    import uuid as _uuid
    trace_id = _uuid.uuid4().hex[:8]
    trace_start = time.time()
    stages = []  # 各环节记录

    def _stage(name, status, input_data="", output_data="", error=""):
        """记录一个环节的执行情况"""
        stages.append({
            "name": name,
            "elapsed_ms": int((time.time() - trace_start) * 1000),
            "status": status,  # pass | hit | skip | error
            "input": str(input_data)[:200] if input_data else "",
            "output": str(output_data)[:200] if output_data else "",
            "error": str(error)[:500] if error else "",
        })

    history = db.get_messages(req.session_id, limit=30)
    if not history:  # 首条消息：用问题做会话标题
        db.rename_session(req.session_id, question[:20])
    # 图片持久化：user 消息同时存文字和图片，重启后可还原显示
    db.add_message(req.session_id, "user", question, images=req.images or None)
    # 发给模型的历史需去掉 images 字段（文本模型不接受该字段）
    history = [{"role": m["role"], "content": m["content"]} for m in history]

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
            hist = memory.maybe_summarize(req.session_id, history)
            memory_ctx = memory.build_memory_context()
            _stage("记忆处理", "pass", "", f"注入{len(memory_ctx)}字" if memory_ctx else "无")

            # ⑤ 知识库检索 + 大模型生成
            messages, sources, matched, kb_images = rag.build_messages(
                question, hist, req.use_kb, req.model or None,
                kb_id=req.kb_id or None,
            )
            _stage("知识库检索", "pass", question,
                   f"命中{len(sources)}条" if sources else "未命中")

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

            for token in ollama_client.chat_stream(messages, req.model or None, images=all_images or None):
                answer += token
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
            # ⑧ 输出敏感词检测：命中则整段替换为安全回复
            if settings.contains_sensitive(answer):
                answer = SENSITIVE_REPLY
                yield f"data: {json.dumps({'type': 'replace', 'content': answer}, ensure_ascii=False)}\n\n"

            _stage("生成回答", "pass", question, answer[:50])

            msg_id = db.add_message(req.session_id, "assistant", answer, sources=sources if sources else [])
            # ⑨ 长期记忆：从本轮问答提取可记忆信息
            try:
                memory.extract_and_save(req.session_id, question, answer)
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
            yield f"data: {json.dumps({'type': 'done', 'sources': sources, 'total_ms': total_ms, 'trace_id': trace_id}, ensure_ascii=False)}\n\n"
        except Exception as e:
            total_ms = int((_time.time() - chat_start) * 1000) if chat_start else 0
            db.add_trace(trace_id, req.session_id, question, answer, total_ms, stages, error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ---------- 链路追踪 ----------
@app.get("/api/traces")
async def list_traces_api(limit: int = 50):
    """获取链路追踪列表"""
    return db.list_traces(limit)


@app.get("/api/traces/{trace_id}")
async def get_trace_api(trace_id: str):
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8008)
