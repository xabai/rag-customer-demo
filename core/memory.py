# -*- coding: utf-8 -*-
"""长期记忆：对话摘要压缩 + 跨会话偏好记忆"""
import re
import db
from core import ollama_client

# 超过此轮数触发摘要压缩
SUMMARY_THRESHOLD = 12

SUMMARY_PROMPT = (
    "你是对话摘要助手。请把以下对话历史压缩成一段简洁摘要，保留关键事实、用户需求、已达成的结论。"
    "只输出摘要本身，不要解释。\n\n对话历史：\n{history}"
)

# 约束 LLM 输出：每行严格 "类型: 内容" 格式，标签行和内容行不分开
EXTRACT_PROMPT = (
    "你是信息提取助手。从以下对话中提取关于用户的稳定长期信息。\n"
    "识别的信息类别只有 4 种：\n"
    "  偏好 —— 用户喜欢/习惯/偏好什么（如：喜欢简洁回答、习惯用表格）\n"
    "  身份 —— 用户角色/身份（如：产品经理、学生、创业者）\n"
    "  需求 —— 用户表达过的长期需求（如：想学习 AI、需要客户管理）\n"
    "  事实 —— 关于用户的客观事实（如：公司在上海、有两个孩子）\n"
    "输出格式：每条一行，严格用英文冒号分隔，例如：\n"
    "  偏好: 喜欢用简洁的语言\n"
    "  身份: 产品经理\n"
    "  需求: 想了解 AI 客服系统\n"
    "  事实: 公司做 SaaS 产品\n"
    "如果没有值得长期记忆的稳定信息，只回复“无”。不要输出空行，不要输出“类型:”单独一行，不要输出编号。\n\n对话：\n{text}"
)

# 类型词 → kind 映射
_KIND_MAP = {
    "偏好": "preference", "喜欢": "preference", "习惯": "preference",
    "身份": "preference",        # 身份也算偏好类（跨会话注入）
    "需求": "fact", "约束": "fact", "事实": "fact",
}


def _parse_extracted(text):
    """解析 LLM 输出，兼容两种格式：
    ① 标准格式：每行 '类型: 内容'  → 直接解析
    ② 旧 bug 格式：'类型:X' 占一行，'内容:Y' 占下一行 → 配对后解析
    返回 [(kind, content), ...]
    """
    lines = [l.strip().lstrip("- ").strip() for l in text.splitlines() if l.strip()]
    results = []
    i = 0
    pending_kind_label = None  # 临时存 "类型:偏好" 这种标签行

    while i < len(lines):
        line = lines[i]

        # 格式 A：完整的 "类型: 内容" —— 直接解析
        # 匹配中英文冒号后的类型词
        m = re.match(r"^([^:：]+?)\s*[:：]\s*(.+)$", line)
        if m:
            label, rest = m.group(1).strip(), m.group(2).strip()

            if label == "类型":
                # 遇到单独一行的 "类型: 偏好" —— 存起来，等下一行的 "内容: xxx"
                pending_kind_label = rest
                i += 1
                continue

            if label in _KIND_MAP:
                # 标准格式 "偏好: 喜欢简洁"
                kind = _KIND_MAP[label]
                results.append((kind, rest))
                pending_kind_label = None
                i += 1
                continue

            # 兜底：带冒号但 label 不认识，检查是不是值里包含了冒号
            # 比如 "事实: 喜欢用 Python:3.12" → 把第一段当类型
            if pending_kind_label and label == "内容":
                # 配对成功：pending_kind_label 是类型，rest 是内容
                kind = _KIND_MAP.get(pending_kind_label, "fact")
                results.append((kind, rest))
                pending_kind_label = None
                i += 1
                continue

            # 真的识别不了——跳过
            i += 1
            continue

        # 没有冒号的行——跳过（说明是噪声或纯内容行）
        i += 1

    return results


def maybe_summarize(session_id, history):
    """对话轮数超过阈值时，把较早的消息压缩成摘要记忆，返回用于拼装的 history"""
    if len(history) <= SUMMARY_THRESHOLD:
        return history
    keep = 6
    old_msgs = history[:-keep]
    recent = history[-keep:]
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_msgs)
    try:
        summary = ollama_client.chat_once([
            {"role": "system", "content": SUMMARY_PROMPT.format(history=history_text)}
        ], timeout=60)
        if summary and summary != "无":
            db.add_memory(session_id, "summary", summary)
            return [{"role": "system", "content": f"【对话摘要】{summary}"}] + recent
    except Exception as e:
        print(f"[memory] 摘要失败，保留原历史: {e}")
    return history


def extract_and_save(session_id, question, answer):
    """从一轮问答中提取可长期记忆的信息"""
    try:
        text = f"用户：{question}\n助手：{answer}"
        extracted = ollama_client.chat_once([
            {"role": "system", "content": EXTRACT_PROMPT.format(text=text)}
        ], timeout=60)
        if not extracted or extracted.strip() == "无":
            return
        parsed = _parse_extracted(extracted)
        for kind, content in parsed:
            db.add_memory(session_id, kind, content)
            print(f"[memory] 提取长期记忆 [{kind}]: {content}")
    except Exception as e:
        print(f"[memory] 提取失败: {e}")


def build_memory_context():
    """把跨会话记忆拼成 system prompt 片段"""
    mems = db.list_cross_memories()
    if not mems:
        return ""
    lines = "\n".join(f"- {m['content']}" for m in mems)
    return f"\n【用户长期记忆】\n{lines}"
