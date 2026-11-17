"""
密码策略 —— 复杂度校验 + 历史检查

配置走 pydantic-settings PASSWORD_* 字段
"""
from __future__ import annotations

import re

from .config import get_settings

_settings = get_settings()


class PasswordPolicy:
    """密码策略校验器"""

    def __init__(self):
        self.min_len = _settings.PASSWORD_MIN_LEN
        self.require_upper = _settings.PASSWORD_REQUIRE_UPPER
        self.require_lower = _settings.PASSWORD_REQUIRE_LOWER
        self.require_digit = _settings.PASSWORD_REQUIRE_DIGIT
        self.require_special = _settings.PASSWORD_REQUIRE_SPECIAL
        self.history_check = _settings.PASSWORD_HISTORY_CHECK

    def validate(self, password: str) -> tuple[bool, str | None]:
        """返回 (is_valid, error_message)"""
        if not password:
            return False, "密码不能为空"
        if len(password) < self.min_len:
            return False, f"密码长度至少 {self.min_len} 位"
        if self.require_upper and not re.search(r"[A-Z]", password):
            return False, "密码需包含大写字母"
        if self.require_lower and not re.search(r"[a-z]", password):
            return False, "密码需包含小写字母"
        if self.require_digit and not re.search(r"\d", password):
            return False, "密码需包含数字"
        if self.require_special and not re.search(r"[^A-Za-z0-9]", password):
            return False, "密码需包含特殊字符"
        # 常见弱密码黑名单
        blacklist = {"password", "password123", "123456", "12345678", "qwerty", "admin", "letmein"}
        if password.lower() in blacklist:
            return False, "密码过于简单，请勿使用常见弱密码"
        return True, None

    def check_history(self, password: str, user_id: int) -> tuple[bool, str | None]:
        """检查密码历史（防止重用过近 N 次的密码）

        注意：历史检查需要 DB 查询，这里只做接口定义
        实际调用方传 hashes: list[str] 进来
        """
        import bcrypt
        from .models import PasswordHistory
        from .db_engine import db_session

        if self.history_check <= 0:
            return True, None

        with db_session() as db:
            recent = (
                db.query(PasswordHistory.password_hash)
                .filter(PasswordHistory.user_id == user_id)
                .order_by(PasswordHistory.created_at.desc())
                .limit(self.history_check)
                .all()
            )
        for (stored_hash,) in recent:
            try:
                if bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8")):
                    return False, f"密码不能与最近 {self.history_check} 次密码相同"
            except Exception:
                pass
        return True, None

    def record_password(self, user_id: int, password_hash: str) -> None:
        """保存新密码到历史表"""
        from .models import PasswordHistory
        from .db_engine import db_session

        with db_session() as db:
            db.add(PasswordHistory(user_id=user_id, password_hash=password_hash))
            # 清理超过 history_check * 2 条的老记录
            to_delete = (
                db.query(PasswordHistory)
                .filter(PasswordHistory.user_id == user_id)
                .order_by(PasswordHistory.created_at.desc())
                .offset(self.history_check * 2)
                .all()
            )
            for item in to_delete:
                db.delete(item)


policy = PasswordPolicy()


def check_password_strength(password: str) -> dict:
    """返回密码强度评分 0-4 和提示"""
    score = 0
    tips: list[str] = []
    if len(password) >= 8: score += 1
    else: tips.append("密码太短")
    if re.search(r"[A-Z]", password): score += 1
    else: tips.append("缺少大写字母")
    if re.search(r"[a-z]", password): score += 1
    else: tips.append("缺少小写字母")
    if re.search(r"\d", password): score += 1
    else: tips.append("缺少数字")
    if re.search(r"[^A-Za-z0-9]", password): score += 1
    # 基础分归一化到 0-4
    normalized = min(4, int(score / 5 * 4))
    levels = ["极弱", "弱", "一般", "强", "极强"]
    return {"score": normalized, "level": levels[normalized], "tips": tips}
