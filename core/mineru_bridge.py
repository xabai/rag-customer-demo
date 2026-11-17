# -*- coding: utf-8 -*-
"""MinerU bridge: 用 mineru conda 环境解析 PDF，JSON 输出 markdown 到 stdout

用法: conda run -n mineru python mineru_bridge.py <pdf_path>
输出: {"ok": true, "markdown": "...", "error": null} 或 {"ok": false, "markdown": null, "error": "..."}
"""
import sys
import json
import traceback


def _clear_mineru_locks():
    """清 MinerU Windows filelock 残留——这是 Windows 上 PermissionError 的元凶"""
    import os, glob, stat
    locks_dir = os.path.join(os.path.expanduser("~"), ".mineru", "models", ".locks")
    try:
        os.makedirs(locks_dir, exist_ok=True)
        for f in glob.glob(os.path.join(locks_dir, "*.lock")):
            try:
                os.chmod(f, stat.S_IWRITE | stat.S_IREAD)
                os.remove(f)
            except OSError:
                pass
    except OSError:
        pass


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"ok": False, "markdown": None, "error": "缺少 PDF 路径参数"}))
        return

    _clear_mineru_locks()
    pdf_path = sys.argv[1]
    try:
        from mineru import parse as mineru_parse
        result = mineru_parse(pdf_path)
        md = result.markdown() if callable(result.markdown) else result.markdown

        # 过滤掉 base64 图片嵌入（mineru 会把页面里的图片内嵌为 data URI）
        import re
        # 移除 ![...](data:image/...) 形式
        md = re.sub(r'!\[[^\]]*\]\(data:image/[^)]+\)', '', md)
        # 也清理可能遗留的空行
        md = re.sub(r'\n{3,}', '\n\n', md).strip()

        print(json.dumps({"ok": True, "markdown": md, "error": None}, ensure_ascii=False))
    except Exception as e:
        print(json.dumps({
            "ok": False, "markdown": None,
            "error": f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-500:]}"
        }, ensure_ascii=False))


if __name__ == '__main__':
    main()
