# -*- coding: utf-8 -*-
"""Ollama 客户端：向量嵌入 + 流式聊天 + 图文统一向量

显存调度策略：
- 向量模型（bge-m3 文本、qwen3-vl-embed 图文）keep_alive=-1，常驻显存
- 聊天/视觉/探测模型 keep_alive=0，用完立即释放
"""
import base64
import io
import json

import numpy as np
import requests

import config


def _options(keep_alive):
    """构造 Ollama API 的 options 参数（keep_alive 秒，-1 常驻，0 立即释放）"""
    return {"keep_alive": keep_alive}


# ============================================================
# 文本向量（bge-m3，常驻）
# ============================================================

def embed_texts(texts):
    """调用 bge-m3 批量生成文本向量，返回 (n, dim) 的 float32 数组（已归一化）"""
    resp = requests.post(
        f"{config.OLLAMA_BASE}/api/embed",
        json={
            "model": config.EMBED_MODEL,
            "input": texts,
            "options": _options(config.KEEP_ALIVE_EMBED),
        },
        timeout=300,
    )
    resp.raise_for_status()
    embeddings = np.array(resp.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def embed_query(text):
    """单条查询向量化（纯文本问题）"""
    return embed_texts([text])[0]


# ============================================================
# 图文统一向量（qwen3-vl-embed-2b，常驻）
# ============================================================

def embed_images(image_paths):
    """调用 qwen3-vl-embed 批量生成图片向量，返回 (n, VL_EMBED_DIM) float32 数组（已归一化）
    支持传入文件路径列表，每张图单独编码"""
    if not image_paths:
        return np.zeros((0, config.VL_EMBED_DIM), dtype=np.float32)
    embeddings = []
    for path in image_paths:
        try:
            with open(path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception:
            # 读不到就跳过，补零向量占位
            embeddings.append(np.zeros(config.VL_EMBED_DIM, dtype=np.float32))
            continue
        resp = requests.post(
            f"{config.OLLAMA_BASE}/api/embed",
            json={
                "model": config.VL_EMBED_MODEL,
                "input": [f"data:image/png;base64,{img_b64}"],
                "options": _options(config.KEEP_ALIVE_VL_EMBED),
            },
            timeout=120,
        )
        try:
            emb = np.array(resp.json()["embeddings"][0], dtype=np.float32)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            embeddings.append(emb)
        except Exception:
            embeddings.append(np.zeros(config.VL_EMBED_DIM, dtype=np.float32))
    return np.array(embeddings, dtype=np.float32)


def embed_vl_texts(texts):
    """用 qwen3-vl-embed 编文本向量（需要与图片向量在同一空间时使用）"""
    if not texts:
        return np.zeros((0, config.VL_EMBED_DIM), dtype=np.float32)
    resp = requests.post(
        f"{config.OLLAMA_BASE}/api/embed",
        json={
            "model": config.VL_EMBED_MODEL,
            "input": texts,
            "options": _options(config.KEEP_ALIVE_VL_EMBED),
        },
        timeout=120,
    )
    resp.raise_for_status()
    embeddings = np.array(resp.json()["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def embed_vl_query(text):
    """查询文本在图文空间的向量"""
    return embed_vl_texts([text])[0]


# ============================================================
# 聊天 / 视觉 / 探测（冷启，用完释放）
# ============================================================

def _inject_images(messages, images):
    """把图片 base64 列表注入到最后一条 user 消息里（Ollama 多模态协议）"""
    if not images:
        return messages
    msgs = [m.copy() for m in messages]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i]["role"] == "user":
            msgs[i]["images"] = images
            break
    return msgs


def chat_once(messages, model=None, timeout=60, images=None):
    """非流式调用聊天模型，一次性返回完整文本（用于查询改写等内部任务）"""
    use_model = model or (config.VISION_MODEL if images else config.CHAT_MODEL)
    keep = config.KEEP_ALIVE_VISION if images else config.KEEP_ALIVE_CHAT
    resp = requests.post(
        f"{config.OLLAMA_BASE}/api/chat",
        json={
            "model": use_model,
            "messages": _inject_images(messages, images),
            "stream": False,
            "think": False,
            "options": _options(keep),
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


def chat_stream(messages, model=None, images=None):
    """调用聊天模型流式输出，逐 token yield 文本
    images: list[str] base64 编码的图片，含图片时自动切换到 VISION_MODEL"""
    use_model = model or (config.VISION_MODEL if images else config.CHAT_MODEL)
    keep = config.KEEP_ALIVE_VISION if images else config.KEEP_ALIVE_CHAT
    resp = requests.post(
        f"{config.OLLAMA_BASE}/api/chat",
        json={
            "model": use_model,
            "messages": _inject_images(messages, images),
            "stream": True,
            "think": False,
            "options": _options(keep),
        },
        stream=True,
        timeout=600,
    )
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        content = data.get("message", {}).get("content", "")
        if content:
            yield content
        if data.get("done"):
            break
