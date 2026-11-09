# -*- coding: utf-8 -*-
"""轻量 BM25 检索器：纯 Python 实现，基于字符二元组（中文无需分词）"""
import math
from collections import Counter


def _get_store():
    """延迟导入，避免与 vector_store 循环依赖"""
    from core.vector_store import store
    return store


def _tokenize(text):
    """中文按字符二元组切分，英文按单词+二元组切分"""
    text = text.lower().replace(" ", "")
    if len(text) < 2:
        return [text] if text else []
    return [text[i:i + 2] for i in range(len(text) - 1)]


class BM25:
    """对当前向量库中的 chunks 构建 BM25 索引（惰性重建）"""

    def __init__(self):
        self._docs = []          # 每个文档的 term 频率 Counter
        self._doc_len = []       # 每个文档长度
        self._avgdl = 0.0
        self._df = Counter()     # 每个 term 出现在多少个文档中
        self._n = 0
        self._dirty = True
        self._k1 = 1.5
        self._b = 0.75

    def _rebuild_if_dirty(self):
        store = _get_store()
        chunks = store._chunks
        if not self._dirty and self._n == len(chunks):
            return
        self._docs = []
        self._doc_len = []
        self._df = Counter()
        for c in chunks:
            tokens = _tokenize(c["text"])
            tf = Counter(tokens)
            self._docs.append(tf)
            self._doc_len.append(len(tokens))
            for t in tf:
                self._df[t] += 1
        self._n = len(chunks)
        self._avgdl = sum(self._doc_len) / self._n if self._n else 0.0
        self._dirty = False

    def mark_dirty(self):
        self._dirty = True

    def search(self, query, top_k=10):
        """返回 [(score, chunk), ...]，score 已归一化到 0~1"""
        self._rebuild_if_dirty()
        if self._n == 0:
            return []
        store = _get_store()
        q_tokens = _tokenize(query)
        scores = []
        for idx in range(self._n):
            score = 0.0
            doc_tf = self._docs[idx]
            dl = self._doc_len[idx]
            for t in q_tokens:
                if t not in doc_tf:
                    continue
                df = self._df[t]
                idf = math.log(1 + (self._n - df + 0.5) / (df + 0.5))
                tf = doc_tf[t]
                numerator = tf * (self._k1 + 1)
                denominator = tf + self._k1 * (1 - self._b + self._b * dl / (self._avgdl or 1))
                score += idf * numerator / denominator
            if score > 0:
                scores.append((score, store._chunks[idx]))
        scores.sort(key=lambda x: x[0], reverse=True)
        top = scores[:top_k]
        max_s = top[0][0] if top else 1.0
        return [(s / max_s, c) for s, c in top]  # 归一化到 0~1


bm25 = BM25()
