# -*- coding: utf-8 -*-
"""系统设置：提示词 + 敏感词 + 检索开关 + LLM 配置，持久化到 data/settings.json"""
import json
import os

import config

SETTINGS_FILE = os.path.join(config.DATA_DIR, "settings.json")

DEFAULT_SETTINGS = {
    # ---------- LLM Provider（统一入口）----------
    "llm_provider": "ollama",           # ollama | openai_compat | baidu_ernie | aliyun_dashscope
    "llm_base": "",                      # 云端 API Base URL（如 https://api.deepseek.com/v1）
    "llm_api_key": "",                   # 云端 API Key（百度用 Secret Key）
    "llm_temperature": 0.7,
    # ---------- LLM / Ollama（本地默认）----------
    "ollama_base": config.OLLAMA_BASE,
    "chat_model": config.CHAT_MODEL,
    "vision_model": config.VISION_MODEL,
    "embed_model": config.EMBED_MODEL,
    "vl_embed_model": config.VL_EMBED_MODEL,
    "keep_alive_embed": config.KEEP_ALIVE_EMBED,
    "keep_alive_vl_embed": config.KEEP_ALIVE_VL_EMBED,
    "keep_alive_chat": config.KEEP_ALIVE_CHAT,
    "keep_alive_vision": config.KEEP_ALIVE_VISION,
    # 可选：OpenAI 兼容（老字段保留）
    "openai_base": "",
    "openai_api_key": "",
    "openai_model": "",
    # ---------- 提示词 ----------
    "kb_prompt": (
        "你是一名严格遵循资料的 AI 客服。\n"
        "必须严格遵守以下规则：\n"
        "1. 只能基于【知识库资料】中的内容回答用户问题，绝不使用你自己的知识；\n"
        "2. 答案必须逐条引用资料原文，不允许概括、推测或补充资料之外的信息；\n"
        "3. 如果资料中没有相关信息，只回答‘抱歉，当前资料中未提及该问题，请联系人工客服或补充资料’；\n"
        "4. 不要在回答中提及‘知识库资料’‘参考资料’‘根据资料’等字眼，直接给出答案；\n"
        "5. 回答简洁，不要有多余开场白或结尾语；\n"
        "6. 必须使用简体中文回答，禁止输出英文或其它语言；\n"
        "7. 严禁重复同一句话或同一个词，若资料不足以回答，直接说资料中未提及；\n"
        "8. 资料中出现的机构名、条款名必须原样引用，不得翻译或改写成其他名称。\n\n"
        "【知识库资料】\n{context}"
    ),
    "no_match_prompt": (
        "你是一名 AI 客服。当前知识库中没有与用户问题相关的资料。\n"
        "你只能回复：‘抱歉，当前资料中未提及该问题，请联系人工客服或补充资料。’\n"
        "禁止使用你自身的知识回答，不要编造任何信息。"
    ),
    "direct_prompt": (
        "你是一名专业、热情的 AI 客服助手。请直接、准确地回答用户的问题，"
        "回答简洁清晰，可分点说明。\n"
        "注意：用户已关闭知识库检索，请完全基于自身知识回答，"
        "不要引用历史对话中出现的任何知识库资料或参考来源内容。"
    ),
    "sensitive_words": "暴力\n赌博\n诈骗\n色情\n枪支\n毒品",
    "rerank_enabled": True,
    "rewrite_enabled": True,
    "rerank_vector_weight": 0.7,
    "rerank_candidates": 20,
    "hybrid_enabled": False,
    "hybrid_vector_weight": 0.6,
    "faq_enabled": True,
    "faq_threshold": 0.6,
    "intent_enabled": True,
}

# 各字段的类型描述，用于 load_settings 做类型安全的合并
_FIELD_TYPES = {
    str: ["llm_provider", "llm_base", "llm_api_key",
          "ollama_base", "chat_model", "vision_model", "embed_model", "vl_embed_model",
          "openai_base", "openai_api_key", "openai_model",
          "kb_prompt", "no_match_prompt", "direct_prompt", "sensitive_words"],
    int: ["keep_alive_embed", "keep_alive_vl_embed", "keep_alive_chat", "keep_alive_vision",
          "rerank_candidates"],
    float: ["llm_temperature", "rerank_vector_weight", "hybrid_vector_weight", "faq_threshold"],
    bool: ["rerank_enabled", "rewrite_enabled", "hybrid_enabled", "faq_enabled", "intent_enabled"],
}
_KEY_TYPE = {}
for _t, _keys in _FIELD_TYPES.items():
    for _k in _keys:
        _KEY_TYPE[_k] = _t


def load_settings():
    """读取全部设置，不存在则写入默认值；与默认值合并保证新键生效"""
    os.makedirs(config.DATA_DIR, exist_ok=True)
    if not os.path.exists(SETTINGS_FILE):
        save_settings(DEFAULT_SETTINGS)
        return _clone(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = {}
    merged = _clone(DEFAULT_SETTINGS)
    for key, default in DEFAULT_SETTINGS.items():
        val = data.get(key)
        t = _KEY_TYPE.get(key) or type(default)
        if isinstance(val, t):
            merged[key] = val
        elif isinstance(val, str) and t is str and val.strip():
            merged[key] = val
        elif isinstance(val, (int, float)) and t is int:
            merged[key] = int(val)
        elif isinstance(val, (int, float)) and t is float:
            merged[key] = float(val)
    return merged


def save_settings(settings):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    data = _clone(DEFAULT_SETTINGS)
    for key in DEFAULT_SETTINGS:
        val = settings.get(key)
        t = _KEY_TYPE.get(key) or type(DEFAULT_SETTINGS[key])
        if key in _FIELD_TYPES[str]:
            if isinstance(val, str) and val.strip():
                data[key] = val
        elif isinstance(val, t):
            data[key] = val
        elif isinstance(val, (int, float)) and t is int:
            data[key] = int(val)
        elif isinstance(val, (int, float)) and t is float:
            data[key] = float(val)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return data


def get_llm_config():
    """返回 LLM 相关配置的便捷 dict（provider/ollama/openai/模型名/keep_alive/温度）"""
    s = load_settings()
    return {
        "llm_provider": s["llm_provider"],
        "llm_base": s["llm_base"],
        "llm_api_key": s["llm_api_key"],
        "llm_temperature": s["llm_temperature"],
        "ollama_base": s["ollama_base"],
        "chat_model": s["chat_model"],
        "vision_model": s["vision_model"],
        "embed_model": s["embed_model"],
        "vl_embed_model": s["vl_embed_model"],
        "keep_alive_embed": s["keep_alive_embed"],
        "keep_alive_vl_embed": s["keep_alive_vl_embed"],
        "keep_alive_chat": s["keep_alive_chat"],
        "keep_alive_vision": s["keep_alive_vision"],
        "openai_base": s["openai_base"],
        "openai_api_key": s["openai_api_key"],
        "openai_model": s["openai_model"],
    }


def _clone(d):
    return {k: (v.copy() if isinstance(v, dict) else v) for k, v in d.items()}


# ---------- 兼容旧接口 ----------
def load_prompts(): return load_settings()
def save_prompts(prompts): return save_settings(prompts)


# ---------- 敏感词 ----------
def load_sensitive_words():
    raw = load_settings().get("sensitive_words", "")
    return [w.strip() for w in raw.splitlines() if w.strip()]


def contains_sensitive(text):
    return any(w in text for w in load_sensitive_words())
