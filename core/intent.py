# -*- coding: utf-8 -*-
"""意图识别 + FAQ 快捷回复：规则优先，向量 exemplar 兜底

三层防线（按顺序）：
  ① 敏感词拦截 → 外部 app.py 做
  ② FAQ 快捷回复 → bigram Jaccard 相似度
  ③ 关键词意图规则 → 词边界正则 + 最长匹配 + 停用词过滤
  ④ BGE-M3 exemplar 向量路由 → 规则没命中时，用意图 exemplar 向量做兜底分类（毫秒级）

思路：
  规则匹配快（纯子串）但可能漏；向量路由慢 10-20ms 但语义覆盖全。
  两者配合：规则负责明确触发词，向量路由负责模糊意图。

停用词表避免单字关键词误命中：配了 `好` → 会命中 "这个问题不好办"；
加停用词后，"好" 只有跟其他关键词一起出现才生效。
"""
import re

import db
from core import settings

# 停用词：匹配时会过滤掉这些常见字，防止单字关键词误命中
STOP_WORDS = set("的了吗呢啊嘛吧也都就会能要去来在是我你他她它这个那一下一些有些")

# 意图 exemplar（向量路由用）：给每种意图写几条典型问题，bge-m3 编向量存 settings
# 用户问题 → 编向量 → 跟所有 exemplar 算余弦 → 取最高分那组意图标签
# 设置页可编辑 / 追加 exemplar；留空则跳过向量路由
DEFAULT_INTENT_EXEMPLARS = {
    "chat": [
        "你好", "您好", "在吗", "hi", "hello", "早上好", "下午好",
        "你是谁", "你能做什么", "介绍一下自己",
    ],
    "transfer": [
        "转人工", "接人工客服", "我要真人", "请转人工", "人工服务",
    ],
    "complain": [
        "我要投诉", "这个不好", "太差了", "不满意", "我要差评",
    ],
}


def _keyword_regex(kw):
    """把关键词变成带词边界的正则，防止单字/短词误命中
    中文没有空格边界，用：不能被中文/英文/数字紧邻包围"""
    escaped = re.escape(kw)
    # (?<![\w\u4e00-\u9fff\d]) 后面同理 —— 左右两边不能紧挨着汉字/字母/数字
    return re.compile(
        r'(?<![\w\u4e00-\u9fff\d])' + escaped + r'(?![\w\u4e00-\u9fff\d])'
    )


def _filter_stopwords(text):
    """把文本里的停用字移除，方便关键词匹配"""
    return "".join(c for c in text if c not in STOP_WORDS)


def match_intent(text):
    """匹配意图规则：规则在前，规则没命中才走 exemplar 向量路由兜底

    推荐在 chat 主链路里用 match_intent_rule_only() 先判断——
    规则精确命中才截断，规则没命中先让 RAG 跑一遍，
    RAG 也没召回到再用 match_exemplar() 兜底。
    """
    # 第 1 层：规则匹配（词边界 + 最长匹配优先）
    rule = match_intent_rule_only(text)
    if rule:
        return rule

    # 第 2 层：exemplar 向量路由兜底
    return _match_exemplar(text)


def match_intent_rule_only(text):
    """仅跑关键词规则匹配（不跑 exemplar 向量路由）

    用于 chat 主链路前置：规则精确命中才直接截断执行；
    规则没命中 → 让 RAG 先跑 → RAG 没召回再用 match_exemplar() 兜底。
    """
    if not settings.load_settings().get("intent_enabled", True):
        return None

    rules = db.list_intent_rules(enabled_only=True)
    if not rules:
        return None

    candidates = []
    for rule in rules:
        keywords = [k.strip() for k in rule["keywords"].split(",") if k.strip()]
        for kw in keywords:
            kw_clean = _filter_stopwords(kw)
            if not kw_clean:
                continue
            regex = _keyword_regex(kw_clean)
            if regex.search(text):
                candidates.append((len(kw_clean), rule))
    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    return None


def _match_exemplar(text):
    """BGE-M3 exemplar 向量路由

    返回: 一个合成的意图规则 dict（name/keywords/action/reply）或 None
    """
    # 读 exemplar 配置（settings.json 里 intent_exemplars 字段）
    cfg = settings.load_settings()
    exemplars = cfg.get("intent_exemplars", DEFAULT_INTENT_EXEMPLARS)
    if not exemplars:
        return None

    # 检查向量缓存（settings.json 里存 intent_vec_cache）
    cache = cfg.get("intent_vec_cache", {})

    from core import ollama_client
    import numpy as np
    try:
        # 检查缓存是否可用（所有 exemplar 的向量都在缓存里）
        all_exemplars = []
        label_map = {}  # exemplar index → 标签名
        for label, examples in exemplars.items():
            for ex in examples:
                all_exemplars.append(ex)
                label_map[len(all_exemplars) - 1] = label

        # 如果缓存的 exemplar 列表和当前一致，直接用；否则重新编
        cache_key = "|".join(all_exemplars)
        if cache.get("key") == cache_key and cache.get("vectors"):
            import numpy as np
            ex_vecs = np.array(cache["vectors"], dtype=np.float32)
        else:
            ex_vecs = ollama_client.embed_texts(all_exemplars)
            import numpy as np
            cfg["intent_vec_cache"] = {"key": cache_key, "vectors": ex_vecs.tolist()}
            settings.save_settings(cfg)

        # 查询向量
        q_vec = ollama_client.embed_query(text)
        scores = ex_vecs @ q_vec
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])

        # 阈值 0.62 以上才触发（在 bge-m3 语义空间里，0.5 附近是大量噪声）
        if best_score < 0.62:
            return None

        label = label_map[best_idx]
        # 根据标签合成一条规则
        action_map = {
            "chat": "chat",
            "transfer": "handoff",
            "complain": "complain",
        }
        reply_map = {
            "chat": "你好！我是客服助手，有什么可以帮您的？",
            "transfer": "好的，正在为您转接人工客服，请稍候。",
            "complain": "非常抱歉给您带来不好的体验，我们会认真处理您的反馈。",
        }
        return {
            "name": f"[向量路由] {label}",
            "keywords": exemplars.get(label, []),
            "action": action_map.get(label, "chat"),
            "reply": reply_map.get(label, ""),
            "source": "exemplar_vector",
            "score": round(best_score, 3),
        }
    except Exception as e:
        print(f"[intent] exemplar 向量路由失败，跳过: {e}")
        return None


def _bigrams(text):
    text = text.replace(" ", "")
    if len(text) < 2:
        return {text} if text else set()
    return {text[i:i + 2] for i in range(len(text) - 1)}


def match_faq(question):
    """FAQ 快捷回复：词面重叠率超过阈值即命中，返回 (answer, score) 或 None"""
    cfg = settings.load_settings()
    if not cfg.get("faq_enabled", True):
        return None
    threshold = cfg.get("faq_threshold", 0.6)
    qset = _bigrams(question)
    if not qset:
        return None
    best = None
    best_score = 0.0
    for item in db.list_faq():
        fset = _bigrams(item["question"])
        if not fset:
            continue
        score = len(qset & fset) / len(qset | fset)
        if score > best_score:
            best_score = score
            best = item
    if best and best_score >= threshold:
        return best["answer"], round(best_score, 3)
    return None


# 问候/闲聊短句 —— 包含以下关键词且不含产品实体词 → 直接回复预设
GREETING_KEYWORDS = (
    "你好", "您好", "嗨", "hi", "hello", "在吗", "在么", "忙吗", "空吗",
    "早上好", "下午好", "晚上好", "大家好", "哈喽",
    "你是谁", "你能做什么", "你是干什么的", "介绍一下自己", "你是",
    "再见", "拜拜", "bye", "谢谢", "谢谢你", "感谢",
)

# 如果用户问题里包含这些实体词（产品/服务相关），说明不是纯问候，应该走 RAG
ENTITY_HINTS = (
    "退货", "退款", "订单", "价格", "发票", "保修", "维修", "售后",
    "产品", "商品", "购买", "下载", "安装", "使用", "怎么用", "如何",
    "联系", "客服", "人工", "投诉", "问题", "故障", "报错",
    "多少钱", "优惠", "活动", "包邮", "配送",
)


def is_greeting(question):
    """判断用户问题是否是纯问候/闲聊短句（不该走 RAG，直接回复预设）

    规则（按顺序短路）：
    1. 问题里**包含任何实体/产品相关词** → 不是问候（哪怕开头有"你好"）
    2. 否则，**包含问候关键词** → 是问候
    3. 关键词匹配后再看字长：如果超过 50 字，大概率是用户在描述一个长问题，排除
    """
    q = (question or "").strip().lower()
    if not q:
        return False
    # 实体词优先：有任何产品/服务相关词 → 不是纯问候
    if any(ent in q for ent in ENTITY_HINTS):
        return False
    # 没有实体词 + 包含问候关键词 → 是问候
    has_greet = any(kw in q for kw in GREETING_KEYWORDS)
    if not has_greet:
        return False
    # 超长排除：超过 50 字大概率是用户在讲具体事情
    if len(q) > 50:
        return False
    return True


GREETING_REPLY = (
    "你好！我是智能客服助手，有什么可以帮您的？\n"
    "您可以直接问我产品相关问题，比如：退货政策、如何下单、产品功能等。"
)
