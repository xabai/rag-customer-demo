# -*- coding: utf-8 -*-
"""RAG 组装：查询改写 + 检索 + 轻量重排序 + 构造 Prompt

检索管线（按优先级触发）：
  1. 文本向量（bge-m3） ← 每次必走
  2. BM25（纯 Python）   ← settings.hybrid_enabled=True 时双路融合
  3. 图文向量（qwen3-vl-embed） ← 用户上传图片 或 问题含"图/像/图片/流程图"等关键词时触发

融合权重：
  hybrid_vector_weight  控制 文本向量 vs BM25
  image_weight          控制 图文向量 vs 其他通道（默认 0.3）
"""
import re

import config
from core import ollama_client, settings
from core.bm25 import bm25
from core.vector_store import store

# 查询改写提示词：把依赖上下文的追问改写成独立完整的检索问题
REWRITE_PROMPT = (
    "你是查询改写助手。根据对话历史，把用户最新问题改写成一个独立、完整、适合检索的查询。"
    "要求：补全“它/这个/那”等指代，保留关键实体词；只输出改写后的问题本身，"
    "不要解释、不要引号、不要换行；若原问题已独立完整，原样输出。"
)

# 触发图文检索的关键词（用户问题里出现这些词，就算没上传图片也走图文双路）
IMAGE_TRIGGER_WORDS = {"图", "图片", "图像", "照片", "相片", "流程图", "示意图",
                       "架构图", "表格", "图表", "截图", "界面", "页面"}


def rewrite_query(question, history, model=None):
    """多轮对话时，结合历史把当前问题改写成独立查询；失败则降级用原问题"""
    if not history:
        return question
    messages = [{"role": "system", "content": REWRITE_PROMPT}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": question})
    try:
        rewritten = ollama_client.chat_once(messages, model, timeout=60)
        rewritten = rewritten.strip().strip('"“”').splitlines()[0].strip()
        if rewritten and rewritten != question:
            print(f"[RAG] 查询改写: {question} -> {rewritten}")
        return rewritten or question
    except Exception as e:
        print(f"[RAG] 查询改写失败，使用原问题: {e}")
        return question


def _bigrams(text):
    text = text.replace(" ", "")
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def rerank(question, hits, vector_weight=0.7):
    qset = _bigrams(question)
    if not qset:
        return hits
    overlap_weight = 1.0 - vector_weight
    fused = []
    for score, chunk in hits:
        cset = _bigrams(chunk["text"])
        overlap = len(qset & cset) / len(qset) if qset else 0.0
        fused.append((vector_weight * score + overlap_weight * overlap, chunk))
    fused.sort(key=lambda x: x[0], reverse=True)
    return fused


def _is_image_query(question, user_images=None):
    """判断是否需要触发图文双路检索"""
    if user_images:
        return True
    for w in IMAGE_TRIGGER_WORDS:
        if w in question:
            return True
    return False


def _chunk_key(chunk):
    """chunk 唯一键（用于多通道融合去重）"""
    return (chunk.get("doc_id", ""), chunk.get("text", ""))


def retrieve(question, rerank_enabled=True, vector_weight=0.7, candidates=20,
             hybrid_enabled=False, hybrid_vector_weight=0.6, kb_id=None,
             user_images=None):
    """检索相关知识块

    user_images: 用户上传的图片 base64 列表（多模态对话）
    """
    if not store.list_documents():
        return "", [], []
    candidate_k = max(candidates, config.TOP_K) if rerank_enabled else config.TOP_K

    # ---- 通道 1：文本向量（bge-m3，每次必走）----
    vec_hits = []
    try:
        query_vec = ollama_client.embed_query(question)
        vec_hits = store.search_text(query_vec, top_k=candidate_k, kb_id=kb_id)
    except Exception as e:
        print(f"[RAG] 文本向量嵌入失败: {e}")

    # ---- 通道 2：BM25 ----
    bm25_hits = []
    if hybrid_enabled:
        bm25_hits = bm25.search(question, top_k=candidate_k)

    # ---- 通道 3：图文向量（qwen3-vl-embed）----
    image_hits = []
    if store._image_vectors is not None and _is_image_query(question, user_images):
        try:
            if user_images:
                # 用户上传了图片 → 直接把第一张图编向量当 query
                import tempfile, os as _os
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    import base64 as _b64
                    img_bytes = _b64.b64decode(user_images[0])
                    tmp.write(img_bytes)
                    tmp_path = tmp.name
                img_vec = ollama_client.embed_images([tmp_path])
                _os.remove(tmp_path)
                if len(img_vec) > 0:
                    image_hits = store.search_image(img_vec[0], top_k=candidate_k, kb_id=kb_id)
            else:
                # 纯文本问题但含图关键词 → 用图文模型编文本向量做检索
                vl_query = ollama_client.embed_vl_query(question)
                # 复用 search_image 通道但改用图文向量空间（所有 chunk 的图文向量都在 image_vectors.npy 里）
                scores = store._image_vectors @ vl_query
                if kb_id:
                    import numpy as np
                    mask = np.array([c.get("kb_id", "default") == kb_id for c in store._chunks])
                    scores = np.where(mask, scores, -1.0)
                idx = scores.argsort()[::-1][:candidate_k]
                for i in idx:
                    if scores[i] > config.SCORE_THRESHOLD:
                        chunk = dict(store._chunks[i])
                        chunk["chunk_index"] = sum(
                            1 for c in store._chunks[:i] if c["doc_id"] == chunk["doc_id"]
                        )
                        image_hits.append((float(scores[i]), chunk, i))
        except Exception as e:
            print(f"[RAG] 图文向量检索失败: {e}")

    # ---- 多通道融合 ----
    all_hits = []  # [(score, chunk)]
    if vec_hits:
        all_hits.extend([(s, c) for s, c, _ in vec_hits])
    if bm25_hits:
        all_hits.extend(bm25_hits)
    if image_hits:
        all_hits.extend([(s, c) for s, c, _ in image_hits])

    if not all_hits:
        return "", [], []

    # 融合：同一 chunk 多条目求和（文本 + BM25 + 图文）
    image_weight = 0.3  # 图文通道权重
    text_weight = 0.7
    fused = {}
    meta = {}
    for score, chunk in all_hits:
        key = _chunk_key(chunk)
        # 判断来源做权重（粗糙但够用：有 layout 字段大概率来自图文通道）
        if chunk.get("layout") is not None and chunk.get("image"):
            w = image_weight
        else:
            w = text_weight
        fused[key] = fused.get(key, 0.0) + w * score
        meta.setdefault(key, chunk)

    hits = [(s, meta[k]) for k, s in fused.items()]

    if rerank_enabled:
        hits = rerank(question, hits, vector_weight)
    hits = hits[:config.TOP_K]

    # 父子分块：命中子块后用其父块作为上下文，并对同一父块去重
    context_parts, sources, kb_images = [], [], []
    seen_parents = set()
    for score, chunk in hits:
        parent = chunk.get("parent_text") or chunk["text"]
        key = (chunk["doc_id"], parent[:64])
        if key in seen_parents:
            continue
        seen_parents.add(key)
        context_parts.append(f"[资料{len(context_parts) + 1}] {parent}")
        src = {
            "doc": chunk["doc_name"],
            "score": round(score, 3),
            "doc_id": chunk.get("doc_id"),
            "chunk_index": chunk.get("chunk_index", 0),
            "has_image": bool(chunk.get("image")),
            "has_layout": bool(chunk.get("layout")),
            "image": chunk.get("image") or None,  # 给前端渲染用，已在 vector_store 转好 URL
        }
        sources.append(src)
        if chunk.get("image"):
            kb_images.append(chunk["image"])
    return "\n\n".join(context_parts), sources, kb_images


def retrieve_debug(question, top_k=5, kb_id=None):
    """召回测试专用：走与线上一致的召回+重排管线"""
    import time as _t
    t0 = _t.time()
    cfg = settings.load_settings()
    rerank_enabled = cfg.get("rerank_enabled", True)
    hybrid_enabled = cfg.get("hybrid_enabled", False)
    candidates = max(cfg.get("rerank_candidates", 20), top_k)

    vec_hits, bm25_hits, image_hits = [], [], []
    try:
        qv = ollama_client.embed_query(question)
        vec_hits = store.search_text(qv, top_k=candidates, kb_id=kb_id, threshold=0.0)
    except Exception as e:
        print(f"[召回测试] 文本向量检索失败: {e}")
    if hybrid_enabled:
        bm25_hits = bm25.search(question, top_k=candidates)
    if store._image_vectors is not None and _is_image_query(question):
        try:
            vl_q = ollama_client.embed_vl_query(question)
            scores = store._image_vectors @ vl_q
            if kb_id:
                import numpy as np
                mask = np.array([c.get("kb_id", "default") == kb_id for c in store._chunks])
                scores = np.where(mask, scores, -1.0)
            idx = scores.argsort()[::-1][:candidates]
            for i in idx:
                chunk = dict(store._chunks[i])
                image_hits.append((float(scores[i]), chunk))
        except Exception as e:
            print(f"[召回测试] 图文检索失败: {e}")

    all_hits = ([(s, c) for s, c, _ in vec_hits] +
                bm25_hits + image_hits)
    if not all_hits:
        return [], int((_t.time() - t0) * 1000)

    # 融合
    fused = {}
    meta = {}
    for score, chunk in all_hits:
        key = _chunk_key(chunk)
        w = 0.3 if (chunk.get("layout") and chunk.get("image")) else 0.7
        fused[key] = fused.get(key, 0.0) + w * score
        meta.setdefault(key, chunk)
    hits = [(s, meta[k]) for k, s in fused.items()]

    if rerank_enabled:
        hits = rerank(question, hits, cfg.get("rerank_vector_weight", 0.7))
    hits.sort(key=lambda x: x[0], reverse=True)
    hits = hits[:top_k]
    elapsed = int((_t.time() - t0) * 1000)
    results = [
        {"doc": c["doc_name"], "score": round(s, 4), "text": c["text"],
         "kb_id": c.get("kb_id", "default"), "doc_id": c.get("doc_id"),
         "chunk_index": c.get("chunk_index", 0),
         "has_image": bool(c.get("image")),
         "has_layout": bool(c.get("layout"))}
        for s, c in hits
    ]
    return results, elapsed


def build_messages(question, history, use_kb=True, model=None, kb_id=None, user_images=None):
    """组装发给聊天模型的 messages；支持多模态图片输入"""
    cfg = settings.load_settings()
    matched = False
    kb_images = []
    if use_kb:
        search_q = (
            rewrite_query(question, history, model)
            if cfg.get("rewrite_enabled", True) else question
        )
        context, sources, kb_images = retrieve(
            search_q,
            rerank_enabled=cfg.get("rerank_enabled", True),
            vector_weight=cfg.get("rerank_vector_weight", 0.7),
            candidates=cfg.get("rerank_candidates", 20),
            hybrid_enabled=cfg.get("hybrid_enabled", False),
            hybrid_vector_weight=cfg.get("hybrid_vector_weight", 0.6),
            kb_id=kb_id,
            user_images=user_images,
        )
        matched = bool(context)
        system = cfg["kb_prompt"].format(context=context) if context else cfg["no_match_prompt"]
    else:
        system = cfg["direct_prompt"]
        sources = []
    messages = [{"role": "system", "content": system}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages, sources, matched, kb_images
