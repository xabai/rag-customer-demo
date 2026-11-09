# -*- coding: utf-8 -*-
"""系统设置：提示词 + 敏感词 + 检索开关，持久化到 data/settings.json"""
import json
import os

import config

SETTINGS_FILE = os.path.join(config.DATA_DIR, "settings.json")

DEFAULT_SETTINGS = {
    # 知识库问答：有匹配资料时使用（严格按资料回答）
    "kb_prompt": (
        "你是一名严格遵循资料的 AI 客服。\n"
        "必须严格遵守以下规则：\n"
        "1. 只能基于【知识库资料】中的内容回答用户问题，绝不使用你自己的知识；\n"
        "2. 答案必须逐条引用资料原文，不允许概括、推测或补充资料之外的信息；\n"
        "3. 如果资料中没有相关信息，只回答“抱歉，当前资料中未提及该问题，请联系人工客服或补充资料”；\n"
        "4. 不要在回答中提及“知识库资料”“参考资料”“根据资料”等字眼，直接给出答案；\n"
        "5. 回答简洁，不要有多余开场白或结尾语。\n\n"
        "【知识库资料】\n{context}"
    ),
    # 知识库开启但无匹配资料时使用
    "no_match_prompt": (
        "你是一名 AI 客服。当前知识库中没有与用户问题相关的资料。\n"
        "你只能回复：“抱歉，当前资料中未提及该问题，请联系人工客服或补充资料。”\n"
        "禁止使用你自身的知识回答，不要编造任何信息。"
    ),
    # 直接对话：完全不用知识库
    "direct_prompt": (
        "你是一名专业、热情的 AI 客服助手。请直接、准确地回答用户的问题，"
        "回答简洁清晰，可分点说明。\n"
        "注意：用户已关闭知识库检索，请完全基于自身知识回答，"
        "不要引用历史对话中出现的任何知识库资料或参考来源内容。"
    ),
    # 敏感词库：一行一个词，输入或输出命中即拦截
    "sensitive_words": "暴力\n赌博\n诈骗\n色情\n枪支\n毒品",
    # 检索增强开关
    "rerank_enabled": True,    # 向量初检 + 词面重叠重排序
    "rewrite_enabled": True,   # 多轮对话时自动改写问题（补全指代）
    # 检索增强参数
    "rerank_vector_weight": 0.7,  # 重排序时向量分数权重（0~1），剩余权重给词面重叠
    "rerank_candidates": 20,      # 开启重排序时初检候选数
    # 混合检索
    "hybrid_enabled": False,        # 向量 + BM25 双路召回
    "hybrid_vector_weight": 0.6,    # 混合检索中向量分数权重
    # FAQ / 意图
    "faq_enabled": True,
    "faq_threshold": 0.6,           # FAQ 匹配阈值（0~1）
    "intent_enabled": True,
}


def load_settings():
    """读取全部设置，不存在则写入默认值；与默认值合并保证新键生效"""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    if not os.path.exists(SETTINGS_FILE):
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS.copy()
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    merged = DEFAULT_SETTINGS.copy()
    for key, default in DEFAULT_SETTINGS.items():
        val = data.get(key)
        if isinstance(default, bool):
            if isinstance(val, bool):
                merged[key] = val
        elif isinstance(default, (int, float)):
            if isinstance(val, (int, float)):
                merged[key] = type(default)(val)
        elif isinstance(val, str) and val.strip():
            merged[key] = val
    return merged


def save_settings(settings):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    data = DEFAULT_SETTINGS.copy()
    for key, default in DEFAULT_SETTINGS.items():
        val = settings.get(key)
        if isinstance(default, bool):
            if isinstance(val, bool):
                data[key] = val
        elif isinstance(default, (int, float)):
            if isinstance(val, (int, float)):
                data[key] = type(default)(val)
        elif isinstance(val, str) and val.strip():
            data[key] = val
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


# 兼容旧接口名
def load_prompts():
    return load_settings()


def save_prompts(prompts):
    return save_settings(prompts)


# ---------- 敏感词 ----------
def load_sensitive_words():
    """从设置中解析敏感词列表（一行一个）"""
    raw = load_settings().get("sensitive_words", "")
    return [w.strip() for w in raw.splitlines() if w.strip()]


def contains_sensitive(text):
    """文本是否命中任一敏感词"""
    return any(w in text for w in load_sensitive_words())
