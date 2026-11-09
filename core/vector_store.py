# -*- coding: utf-8 -*-
"""纯 NumPy 向量库：余弦相似度检索，本地 .npy + .json 持久化

双路向量索引：
  vectors.npy       — bge-m3 文本向量 (n, 1024)
  image_vectors.npy — qwen3-vl-embed 图文向量 (n, 2048)，无图片的 chunk 用零向量占位
chunks.json 每 chunk 可含: image / layout_meta 字段（图文关联元数据）
"""
import json
import os
import threading
import time

import numpy as np

import config
from core.bm25 import bm25

IMAGE_VECTOR_PATH = os.path.join(config.DATA_DIR, "image_vectors.npy")


class VectorStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._vectors = None       # 文本向量 (n, 1024)
        self._image_vectors = None  # 图片向量 (n, VL_EMBED_DIM)，可 None
        self._chunks = []           # [{"doc_id", "doc_name", "text", ...}, ...]
        self._docs = {}
        self._load()

    # ---------- 持久化 ----------
    def _load(self):
        if os.path.exists(config.CHUNKS_PATH):
            with open(config.CHUNKS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._chunks = data.get("chunks", [])
            self._docs = data.get("docs", {})
        if os.path.exists(config.VECTOR_PATH):
            self._vectors = np.load(config.VECTOR_PATH)
        if os.path.exists(IMAGE_VECTOR_PATH):
            self._image_vectors = np.load(IMAGE_VECTOR_PATH)

    def _save(self):
        with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump({"chunks": self._chunks, "docs": self._docs}, f, ensure_ascii=False)
        if self._vectors is not None:
            np.save(config.VECTOR_PATH, self._vectors)
        elif os.path.exists(config.VECTOR_PATH):
            os.remove(config.VECTOR_PATH)
        if self._image_vectors is not None:
            np.save(IMAGE_VECTOR_PATH, self._image_vectors)
        elif os.path.exists(IMAGE_VECTOR_PATH):
            os.remove(IMAGE_VECTOR_PATH)

    # ---------- 文档操作 ----------
    def add_document(self, doc_name, chunk_texts, vectors, kb_id="default",
                     parent_texts=None, chunk_images=None, layout_meta=None,
                     image_vectors=None):
        """新增文档

        layout_meta: list[list[dict]] — 每个 chunk 的 elements 原始数据（图文关联元数据）
        image_vectors: np.ndarray (n, VL_EMBED_DIM) — 图文向量，无图片的 chunk 用零向量占位
        """
        with self._lock:
            doc_id = f"doc_{int(time.time() * 1000)}"
            for i, text in enumerate(chunk_texts):
                rec = {"doc_id": doc_id, "doc_name": doc_name, "text": text, "kb_id": kb_id}
                if parent_texts and i < len(parent_texts) and parent_texts[i]:
                    rec["parent_text"] = parent_texts[i]
                if chunk_images and i < len(chunk_images) and chunk_images[i]:
                    rec["image"] = chunk_images[i]
                if layout_meta and i < len(layout_meta) and layout_meta[i]:
                    rec["layout"] = layout_meta[i]
                self._chunks.append(rec)

            self._vectors = (
                vectors if self._vectors is None
                else np.vstack([self._vectors, vectors])
            )

            # 图片向量：对齐 chunks 数量，无图片的 chunk 用零向量
            if image_vectors is not None:
                self._image_vectors = (
                    image_vectors if self._image_vectors is None
                    else np.vstack([self._image_vectors, image_vectors])
                )
            elif self._image_vectors is not None:
                zeros = np.zeros((len(chunk_texts), config.VL_EMBED_DIM), dtype=np.float32)
                self._image_vectors = np.vstack([self._image_vectors, zeros])

            self._docs[doc_id] = {
                "id": doc_id,
                "name": doc_name,
                "kb_id": kb_id,
                "chunks": len(chunk_texts),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._save()
            bm25.mark_dirty()
            return doc_id

    def delete_document(self, doc_id):
        with self._lock:
            if doc_id not in self._docs:
                return False
            keep = [i for i, c in enumerate(self._chunks) if c["doc_id"] != doc_id]
            self._chunks = [self._chunks[i] for i in keep]
            self._vectors = self._vectors[keep] if keep and self._vectors is not None else None
            if self._image_vectors is not None:
                self._image_vectors = (
                    self._image_vectors[keep] if keep else None
                )
            del self._docs[doc_id]
            self._save()
            bm25.mark_dirty()
            return True

    def list_documents(self):
        return sorted(self._docs.values(), key=lambda d: d["created"], reverse=True)

    def get_document_chunks(self, doc_id):
        with self._lock:
            result = []
            for i, c in enumerate(self._chunks):
                if c["doc_id"] != doc_id:
                    continue
                image_url = None
                raw_img = c.get("image")
                if raw_img:
                    # 绝对路径 → HTTP URL（支持多图：分号分隔）
                    # "F:\rag-BOT\data\kb_images\xxx.png" → "/kb_images/xxx.png"
                    parts = []
                    for p in raw_img.split(";"):
                        fname = os.path.basename(p.strip())
                        if fname:
                            parts.append(f"/kb_images/{fname}")
                    image_url = ";".join(parts) if parts else None
                # layout 元数据里的图片也转成 URL
                layout = c.get("layout")
                if layout:
                    for el in layout:
                        if el.get("type") == "image" and el.get("image_path"):
                            fname = os.path.basename(el["image_path"])
                            el["image_url"] = f"/kb_images/{fname}"
                result.append({
                    "index": i, "text": c["text"],
                    "image": image_url, "layout": layout
                })
            return result

    def update_chunk(self, doc_id, chunk_index, new_text):
        from core import ollama_client
        with self._lock:
            doc_start = None
            doc_count = 0
            for i, c in enumerate(self._chunks):
                if c["doc_id"] == doc_id:
                    if doc_start is None:
                        doc_start = i
                    doc_count += 1
            if doc_start is None:
                return False
            if chunk_index < 0 or chunk_index >= doc_count:
                return False
            global_idx = doc_start + chunk_index
            self._chunks[global_idx]["text"] = new_text
            new_vec = ollama_client.embed_texts([new_text])
            self._vectors[global_idx] = new_vec[0]
            self._save()
            bm25.mark_dirty()
            return True

    # ---------- 检索 ----------
    @staticmethod
    def _abs_to_url(abs_path):
        """绝对路径 → HTTP URL（支持分号分隔多图）"""
        if not abs_path:
            return None
        import os as _os
        parts = []
        for p in str(abs_path).split(";"):
            p = p.strip()
            if not p:
                continue
            try:
                rel = _os.path.relpath(p, config.KB_IMG_DIR).replace("\\", "/")
                parts.append(f"/kb_images/{rel}")
            except ValueError:
                # relpath 在不同盘符时报 ValueError，用 basename 兜底
                parts.append(f"/kb_images/{_os.path.basename(p)}")
        return ";".join(parts) if parts else None

    def search_text(self, query_vec, top_k=None, threshold=None, kb_id=None):
        """纯文本向量检索（bge-m3）"""
        if self._vectors is None or len(self._chunks) == 0:
            return []
        top_k = top_k or config.TOP_K
        threshold = config.SCORE_THRESHOLD if threshold is None else threshold
        scores = self._vectors @ query_vec
        if kb_id:
            mask = np.array([c.get("kb_id", "default") == kb_id for c in self._chunks])
            scores = np.where(mask, scores, -1.0)
        idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in idx:
            if scores[i] < threshold:
                continue
            chunk = dict(self._chunks[i])
            chunk["chunk_index"] = sum(
                1 for c in self._chunks[:i] if c["doc_id"] == chunk["doc_id"]
            )
            # 绝对路径 → HTTP URL（让前端直接渲染）
            chunk["image"] = self._abs_to_url(chunk.get("image"))
            results.append((float(scores[i]), chunk, i))
        return results

    def search_image(self, query_img_vec, top_k=None, threshold=None, kb_id=None):
        """图片向量检索（qwen3-vl-embed）——只检索有图片的 chunk"""
        if self._image_vectors is None or len(self._chunks) == 0:
            return []
        top_k = top_k or config.TOP_K
        threshold = config.SCORE_THRESHOLD if threshold is None else threshold
        scores = self._image_vectors @ query_img_vec
        if kb_id:
            mask = np.array([c.get("kb_id", "default") == kb_id for c in self._chunks])
            scores = np.where(mask, scores, -1.0)
        idx = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in idx:
            if scores[i] < threshold or not self._chunks[i].get("image"):
                continue
            chunk = dict(self._chunks[i])
            chunk["chunk_index"] = sum(
                1 for c in self._chunks[:i] if c["doc_id"] == chunk["doc_id"]
            )
            chunk["image"] = self._abs_to_url(chunk.get("image"))
            results.append((float(scores[i]), chunk, i))
        return results

    def get_all_vectors(self):
        return self._vectors, self._image_vectors

    def get_chunk_image_paths(self):
        """返回所有含图片的 chunk 图片路径列表（用于批量向量化）"""
        return [c.get("image") for c in self._chunks]


store = VectorStore()
