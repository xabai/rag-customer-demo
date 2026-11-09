# -*- coding: utf-8 -*-
"""文本切分：按段落聚合到目标长度，带重叠"""
import re

import config


def split_text(text, chunk_size=None, chunk_overlap=None):
    """先按段落/换行切，再聚合成约 chunk_size 字符的块，块间重叠 chunk_overlap 字符"""
    size = chunk_size or config.CHUNK_SIZE
    overlap = chunk_overlap if chunk_overlap is not None else config.CHUNK_OVERLAP
    text = re.sub(r"\r\n?", "\n", text)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}|(?<=[。！？!?])\n", text) if p.strip()]

    chunks = []
    buf = ""
    for para in paragraphs:
        # 单段超过切块大小：先按句号等硬切
        if len(para) > size:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_hard_split(para, size, overlap))
            continue
        if len(buf) + len(para) + 1 > size and buf:
            chunks.append(buf)
            buf = _tail(buf, overlap) + para
        else:
            buf = f"{buf}\n{para}" if buf else para
    if buf:
        chunks.append(buf)
    return chunks


def split_parent_child(text, parent_size=1500, child_size=400, chunk_overlap=None):
    """父子分块：先切大父块（保留完整上下文），再把父块切成小检索子块（提升命中精度）。
    返回 [{"parent": 父块文本, "children": [子块文本, ...]}, ...]"""
    overlap = config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    parents = split_text(text, chunk_size=parent_size, chunk_overlap=overlap)
    result = []
    for p in parents:
        if len(p) <= child_size:
            result.append({"parent": p, "children": [p]})
        else:
            children = split_text(p, chunk_size=child_size, chunk_overlap=overlap)
            result.append({"parent": p, "children": children or [p]})
    return result


def _hard_split(text, size, overlap):
    """长段落按句子边界硬切"""
    sentences = re.split(r"(?<=[。！？!?；;])", text)
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) > size and buf:
            chunks.append(buf)
            buf = _tail(buf, overlap) + s
        else:
            buf += s
    if buf:
        chunks.append(buf)
    return chunks


def _tail(text, overlap):
    """取上一块结尾作为重叠部分"""
    return text[-overlap:] if overlap > 0 else ""
