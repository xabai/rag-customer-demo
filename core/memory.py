# -*- coding: utf-8 -*-
"""长期记忆：对话摘要压缩 + 跨会话偏好记忆"""
import db
from core import ollama_client

# 超过此轮数触发摘要压缩
SUMMARY_THRESHOLD = 12

SUMMARY_PROMPT = (
    "你是对话摘要助手。请把以下对话历史压缩成一段简洁摘要，保留关键事实、用户需求、已达成的结论。"
    "只输出摘要本身，不要解释。\n\n对话历史：\n{history}"
)

EXTRACT_PROMPT = (
    "你是信息提取助手。从以下对话中提取关于用户的稳定信息（偏好、身份、需求、约束），"
    "每条一行，格式：「类型:内容」。如果没有值得长期记忆的信息，只回复“无”。\n\n对话：\n{text}"
)


def maybe_summarize(session_id, history):
    """对话轮数超过阈值时，把较早的消息压缩成摘要记忆，返回用于拼装的 history"""
    if len(history) <= SUMMARY_THRESHOLD:
        return history
    # 取较早的一半做摘要，保留最近 6 轮
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
            # 用摘要替换早期历史
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
        for line in extracted.strip().splitlines():
            line = line.strip().strip("- ")
            if not line:
                continue
            kind = "preference" if any(k in line for k in ["偏好", "喜欢", "习惯", "身份"]) else "fact"
            db.add_memory(session_id, kind, line)
            print(f"[memory] 提取长期记忆: {line}")
    except Exception as e:
        print(f"[memory] 提取失败: {e}")


def build_memory_context():
    """把跨会话记忆拼成 system prompt 片段"""
    mems = db.list_cross_memories()
    if not mems:
        return ""
    lines = "\n".join(f"- {m['content']}" for m in mems)
    return f"\n【用户长期记忆】\n{lines}"
