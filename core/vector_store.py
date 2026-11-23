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
                     image_vectors=None, parent_ids=None,
                     metadata=None):
        """新增文档

        metadata: dict 或 list[dict] —— 可选的元数据，可传：
            - dict: 所有 chunk 共用同一套 metadata（推荐）
            - list: 每个 chunk 一套，长度需等于 len(chunk_texts)
          支持的字段：doc_type / heading_path / page_no / valid_from / valid_to / source_path / uploader_id
        """
        with self._lock:
            doc_id = f"doc_{int(time.time() * 1000)}"
            for i, text in enumerate(chunk_texts):
                rec = {"doc_id": doc_id, "doc_name": doc_name, "text": text, "kb_id": kb_id}
                if parent_texts and i < len(parent_texts) and parent_texts[i]:
                    rec["parent_text"] = parent_texts[i]
                if parent_ids and i < len(parent_ids) and parent_ids[i]:
                    rec["parent_id"] = parent_ids[i]
                if chunk_images and i < len(chunk_images) and chunk_images[i]:
                    rec["image"] = chunk_images[i]
                if layout_meta and i < len(layout_meta) and layout_meta[i]:
                    rec["layout"] = layout_meta[i]
                # ★ 新增：metadata 注入
                if metadata:
                    if isinstance(metadata, dict):
                        rec.update(metadata)
                    elif isinstance(metadata, list) and i < len(metadata) and metadata[i]:
                        rec.update(metadata[i])
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

    def update_document(self, old_doc_id, doc_name, chunk_texts, vectors, **add_kwargs):
        """原子替换：先删除旧文档，再添加新 chunks。

        加新失败时旧文档保持不变（不会"删了加不回"导致该文档索引为空）。
        """
        with self._lock:
            if old_doc_id not in self._docs:
                # 旧文档不存在 → 退化为直接 add
                return self.add_document(doc_name, chunk_texts, vectors, **add_kwargs)

            # 先算好新的所有参数
            doc_id = old_doc_id  # 保持同一个 doc_id，对外部透明

            # 从 add_kwargs 中提取可能不存在的可选参数
            parent_texts = add_kwargs.get("parent_texts")
            chunk_images = add_kwargs.get("chunk_images")
            layout_meta = add_kwargs.get("layout_meta")
            image_vectors = add_kwargs.get("image_vectors")
            parent_ids = add_kwargs.get("parent_ids")
            metadata = add_kwargs.get("metadata")
            kb_id = add_kwargs.get("kb_id", "default")

            # ① 先构造新的 chunks 列表（在内存中，还没碰持久化）
            new_chunks = []
            for i, text in enumerate(chunk_texts):
                rec = {"doc_id": doc_id, "doc_name": doc_name, "text": text, "kb_id": kb_id}
                if parent_texts and i < len(parent_texts) and parent_texts[i]:
                    rec["parent_text"] = parent_texts[i]
                if parent_ids and i < len(parent_ids) and parent_ids[i]:
                    rec["parent_id"] = parent_ids[i]
                if chunk_images and i < len(chunk_images) and chunk_images[i]:
                    rec["image"] = chunk_images[i]
                if layout_meta and i < len(layout_meta) and layout_meta[i]:
                    rec["layout"] = layout_meta[i]
                if metadata:
                    if isinstance(metadata, dict):
                        rec.update(metadata)
                    elif isinstance(metadata, list) and i < len(metadata) and metadata[i]:
                        rec.update(metadata[i])
                new_chunks.append(rec)

            # ② 计算要删除的索引
            remove_idx = {i for i, c in enumerate(self._chunks) if c["doc_id"] == doc_id}
            keep_idx = [i for i in range(len(self._chunks)) if i not in remove_idx]

            # ③ 替换：旧索引被删，新 chunks append，向量对齐
            self._chunks = [self._chunks[i] for i in keep_idx] + new_chunks
            if self._vectors is not None:
                kept = self._vectors[keep_idx] if keep_idx else None
                self._vectors = (
                    np.vstack([kept, vectors]) if kept is not None
                    else vectors.copy()
                )
            if self._image_vectors is not None:
                kept_iv = self._image_vectors[keep_idx] if keep_idx else None
                # 新 image_vectors 对齐：没传就用零向量占位
                if image_vectors is not None:
                    new_iv = image_vectors
                else:
                    new_iv = np.zeros((len(chunk_texts), config.VL_EMBED_DIM), dtype=np.float32)
                self._image_vectors = (
                    np.vstack([kept_iv, new_iv]) if kept_iv is not None
                    else new_iv.copy()
                )

            # ④ 更新 docs 元信息 + 持久化
            self._docs[doc_id] = {
                "id": doc_id, "name": doc_name, "kb_id": kb_id,
                "chunks": len(chunk_texts),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._save()
            bm25.mark_dirty()
            return doc_id

    # 允许在前端编辑的元数据字段
    EDITABLE_META_KEYS = ("doc_type", "title", "valid_from", "valid_to", "tags")

    def get_document_metadata(self, doc_id):
        """读取文档级元数据（取任一分块上的字段）"""
        with self._lock:
            for c in self._chunks:
                if c["doc_id"] == doc_id:
                    return {k: c.get(k) for k in self.EDITABLE_META_KEYS if c.get(k) not in (None, "")}
        return {}

    def update_metadata(self, doc_id, patch):
        """只更新元数据，不重新向量化；返回更新的分块数（0 表示文档不存在）"""
        if not isinstance(patch, dict):
            return 0
        clean = {}
        for k, v in patch.items():
            if k not in self.EDITABLE_META_KEYS:
                continue
            clean[k] = v
        with self._lock:
            n = 0
            for c in self._chunks:
                if c["doc_id"] != doc_id:
                    continue
                n += 1
                for k, v in clean.items():
                    if v in (None, ""):
                        c.pop(k, None)
                    else:
                        c[k] = v
            if n:
                self._save()
            return n

    def list_documents(self, page=1, page_size=None):
        """page_size=None 时保持旧行为返回 list；否则返回 {items, total, page, page_size, pages}"""
        items = sorted(self._docs.values(), key=lambda d: d["created"], reverse=True)
        if page_size is None:
            return items
        ps = max(1, int(page_size))
        total = len(items)
        pages = max(1, (total + ps - 1) // ps)
        p = max(1, min(int(page), pages))
        start = (p - 1) * ps
        return {"items": items[start:start + ps], "total": total, "page": p, "page_size": ps, "pages": pages}

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

    def search_text(self, query_vec, top_k=None, threshold=None, kb_id=None,
                    filters=None, skip_expired=True):
        """纯文本向量检索（bge-m3）

        filters: dict 可选元数据过滤，支持字段：
            - doc_type    str  仅匹配文件类型（"pdf"/"xlsx"/"docx" 等）
            - source_path str  仅匹配源文件路径
            - uploader_id int  仅匹配上传者
            - doc_id      str  仅匹配指定文档
            - heading_path str 包含 heading 路径的 chunk

        skip_expired: bool 自动过滤已过期 chunk（valid_to < 当前时间）
        """
        if self._vectors is None or len(self._chunks) == 0:
            return []
        top_k = top_k or config.TOP_K
        threshold = config.SCORE_THRESHOLD if threshold is None else threshold
        scores = self._vectors @ query_vec

        # ① kb_id 过滤
        if kb_id:
            mask = np.array([c.get("kb_id", "default") == kb_id for c in self._chunks])
            scores = np.where(mask, scores, -1.0)

        # ② metadata filters
        if filters:
            for key, val in filters.items():
                if val is None:
                    continue
                # 支持 "包含" 匹配（如 heading_path 包含关键词）
                if key == "heading_path":
                    mask = np.array([val in str(c.get("heading_path", "")) for c in self._chunks])
                else:
                    mask = np.array([str(c.get(key, "")) == str(val) for c in self._chunks])
                scores = np.where(mask, scores, -1.0)

        # ③ 自动跳过已过期内容
        if skip_expired:
            import time as _t
            now = _t.time()
            mask = np.array([
                not c.get("valid_to") or float(c["valid_to"]) > now
                for c in self._chunks
            ])
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
