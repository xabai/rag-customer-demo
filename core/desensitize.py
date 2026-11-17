"""
敏感信息脱敏 —— 统一在审计日志/日志里脱敏敏感字段

设计：
- 提供 desensitize() 函数，对字符串中常见敏感数据模式替换
- 审计中间件写日志前调用
"""
from __future__ import annotations

import re


# ---- 敏感模式（按优先级匹配） ----
_PATTERNS: list[tuple[re.Pattern, str]] = [
    # JWT
    (re.compile(r"eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"), "[JWT_REDACTED]"),
    # Bearer Token
    (re.compile(r"(Bearer\s+)[A-Za-z0-9._-]+", re.IGNORECASE), r"\1[TOKEN_REDACTED]"),
    # API Key
    (re.compile(r"(api[_-]?key\s*[=:]\s*)[A-Za-z0-9_\-]{16,}", re.IGNORECASE), r"\1[API_KEY_REDACTED]"),
    # 密码字段（JSON body 中的 "password": "xxx"）
    (re.compile(r'("password"\s*:\s*")[^"]*(")', re.IGNORECASE), r'\1***REDACTED***\2'),
    (re.compile(r'("old_password"\s*:\s*")[^"]*(")', re.IGNORECASE), r'\1***REDACTED***\2'),
    (re.compile(r'("new_password"\s*:\s*")[^"]*(")', re.IGNORECASE), r'\1***REDACTED***\2'),
    # 手机号（中国 11 位）
    (re.compile(r"1[3-9]\d{9}"), "1**********"),
    # 邮箱
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "***@***.***"),
    # 身份证号（15/18 位）
    (re.compile(r"\d{17}[\dXx]|\d{15}"), "****************"),
    # 银行卡（16-19 位数字）
    (re.compile(r"\b\d{16,19}\b"), "****-****-****-****"),
    # 内网 IP（审计日志里可选脱敏）
    # (re.compile(r'\b(10\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])|192\.168)\.\d{1,3}\.\d{1,3}\b'), '[INTERNAL_IP]'),
]


def desensitize(text: str) -> str:
    """对任意字符串做脱敏，返回脱敏后的副本"""
    if not text:
        return text
    for pattern, repl in _PATTERNS:
        text = pattern.sub(repl, text)
    return text


def desensitize_dict(data: dict) -> dict:
    """递归脱敏 dict（对所有 str 值）"""
    result = {}
    for k, v in data.items():
        if isinstance(v, str):
            result[k] = desensitize(v)
        elif isinstance(v, dict):
            result[k] = desensitize_dict(v)
        elif isinstance(v, list):
            result[k] = [desensitize_dict(i) if isinstance(i, dict)
                         else desensitize(i) if isinstance(i, str)
                         else i for i in v]
        else:
            result[k] = v
    return result
