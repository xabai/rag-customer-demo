# -*- coding: utf-8 -*-
"""文件解析：多格式统一为版面感知的结构化元素 + 图文配对 + 版面感知分块

元素类型: text / image / table / heading / chart
每个元素带 bbox{x0,y0,x1,y1}（像素坐标，相对 page/slide）和 parent_id（页码/幻灯片号）
配对规则: 同 parent + 空间距离最近 → image.anchor_to = text.id
分块规则: 按 heading 边界切，单块上限 800 字 / 3 张图（config.LAYOUT_MAX_WORDS / LAYOUT_MAX_IMAGES）
"""
import os
import re
import uuid
import io
import shutil
import tempfile

import config

# =====================================================================
# OCR 懒加载单例（避免每张图都加载一次模型）
# =====================================================================
_OCR_INSTANCE = None


def _get_ocr():
    """返回 RapidOCR 单例；不可用时返回 None（整个图片 OCR 降级为空操作）"""
    global _OCR_INSTANCE
    if _OCR_INSTANCE is not None:
        return _OCR_INSTANCE
    try:
        from rapidocr_onnxruntime import RapidOCR
        _OCR_INSTANCE = RapidOCR()
        return _OCR_INSTANCE
    except Exception as e:
        print(f"[parser] RapidOCR 不可用，图片 OCR 跳过: {e}")
        _OCR_INSTANCE = False  # False = 不可用，以后不用再试
        return None


# =====================================================================
# 图片入库三阶段：判断 → 懒 OCR → 结构化
# =====================================================================

def _should_ocr(img_el, chunk_text_all):
    """判断一张图片要不要 OCR —— 核心策略：附近文字能覆盖图片主题就不 OCR

    Args:
        img_el: LayoutElement(type="image")
        chunk_text_all: 同 chunk 内所有 heading + text 聚合后的字符串

    Returns:
        bool: True = 需要 OCR；False = 附近文字已覆盖，只做图文关联
    """
    # 1) anchor_to 为空 → 没配对上任何文字 → 必须 OCR
    if img_el.anchor_to is None:
        return True

    text = (chunk_text_all or "").strip()
    if not text:
        return True  # 同 chunk 完全没文字 → 必须 OCR

    # 2) chunk 聚合文字很短（< 15 字）→ 可能只是一个笼统 heading，覆盖不了 → OCR
    if len(text) < 15:
        return True

    # 3) 文字里有产品/功能/规格关键词 → 图片主题大概率已覆盖 → 不 OCR
    coverage_keywords = [
        "产品", "规格", "参数", "功能", "特点", "特性", "外观", "界面", "操作",
        "安装", "使用", "配置", "架构", "流程", "步骤", "效果", "性能",
        "价格", "价格表", "对比", "对比表",
    ]
    hit = sum(1 for kw in coverage_keywords if kw in text)
    if hit >= 1:
        return False  # 被覆盖 → 不 OCR

    # 4) 兜底：文字不短但也没关键词 → OCR（可能图片是独立内容）
    return True


def _ocr_image(img_path):
    """对单张图片跑 RapidOCR，返回合并后的文本

    Returns:
        str: OCR 合并文本；失败或无结果返回 ""
    """
    if not img_path or not os.path.exists(img_path):
        return ""
    ocr = _get_ocr()
    if ocr is None:
        return ""
    try:
        result, _ = ocr(img_path)
        if not result:
            return ""
        # 按 y 坐标排序（从上到下），合并
        lines = []
        for quad, text, conf in result:
            if conf < 0.5:
                continue
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            top_y = min(ys)
            lines.append((top_y, text.strip()))
        lines.sort(key=lambda x: x[0])
        return "\n".join(t for _, t in lines if t)
    except Exception as e:
        print(f"[parser] OCR 失败 {os.path.basename(img_path)}: {e}")
        return ""


def _ocr_to_structured(ocr_text):
    """判断 OCR 文本是否是表格/键值对，尝试结构化

    Returns:
        dict: {"kind": "table"|"kv"|"text"|"none", ...}
        - table: {"kind": "table", "headers": [...], "rows": [[...], ...]}
        - kv:    {"kind": "kv", "pairs": [{"key": "...", "value": "..."}, ...]}
        - text:  {"kind": "text", "raw": "..."}
        - none:  {"kind": "none"}  → 没有有效识别结果
    """
    text = (ocr_text or "").strip()
    if not text or len(text) < 5:
        return {"kind": "none"}

    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # ---- 键值对检测（行内出现冒号/分号 "Key: Value" 模式）----
    kv_pattern = re.compile(r"^(.+?)[：:;]\s*(.+)$")
    kv_pairs = []
    for ln in lines:
        m = kv_pattern.match(ln)
        if m and len(m.group(1)) < 20 and len(m.group(2)) < 30:
            kv_pairs.append({"key": m.group(1).strip(), "value": m.group(2).strip()})

    if len(kv_pairs) >= 2:
        return {"kind": "kv", "pairs": kv_pairs}

    # ---- 简易表格检测（至少 2 行、每行 2+ 个"列"用空格/制表符分隔）----
    if len(lines) >= 2:
        splitter = re.compile(r"[\t\s]{2,}")
        cols_per_line = [len(splitter.split(l)) for l in lines]
        if min(cols_per_line) >= 2 and len(set(cols_per_line)) == 1:
            # 每行列数一致 → 判定为表格
            cells = [splitter.split(l) for l in lines]
            headers = cells[0]
            rows = cells[1:]
            return {"kind": "table", "headers": headers, "rows": rows}

    # ---- 普通文字（OCR 结果不是表格也不是键值对）----
    return {"kind": "text", "raw": text}


def _build_image_content(img_el, ocr_struct, chunk_text_all):
    """根据 OCR 结构化结果 + 附近文字，给图片的 content 填值（喂向量检索用）

    Returns:
        str: 填到 LayoutElement.content 的文本；空串 = 这张图不参与向量检索
    """
    text = (chunk_text_all or "").strip()

    # OCR 成功 → 优先用 OCR 结果
    if ocr_struct["kind"] == "table":
        headers = " / ".join(ocr_struct["headers"])
        return f"[图片表格: {headers}]"
    if ocr_struct["kind"] == "kv":
        pairs = " / ".join(f"{p['key']}={p['value']}" for p in ocr_struct["pairs"])
        return f"[图片参数: {pairs}]"
    if ocr_struct["kind"] == "text":
        # OCR 有普通文字 → 直接用
        cleaned = re.sub(r"\s+", " ", ocr_struct["raw"])[:200]
        return f"[图片文字: {cleaned}]"

    # OCR 没用 → 用附近文字填一个占位（让图能和 chunk 关联上）
    if text:
        # 从 heading 里抽个短语就行
        snippet = re.split(r"[。！？；;\n]", text)[0][:40]
        return f"[配图说明: {snippet}]"

    return ""  # 彻底没用，content 为空


# =====================================================================
# 通用分隔符工具（parent_sep / child_sep 共用）
# =====================================================================

def _compile_separator(sep_str, hint=""):
    """编译分隔符为 regex。先尝试原字符串，匹配不到则 re.escape 兜底。"""
    if not sep_str:
        return None
    try:
        sep_re = re.compile(sep_str, re.MULTILINE)
        if hint and not sep_re.search(hint):
            sep_re = re.compile(re.escape(sep_str))
        elif not hint and not sep_re.search(sep_str):
            sep_re = re.compile(re.escape(sep_str))
    except re.error:
        sep_re = re.compile(re.escape(sep_str))
    return sep_re


def _split_elements_by_separator(elements, sep_re):
    """把 text/heading element 的 content 按 separator 拆成多个独立 element"""
    split_elements = []
    for el in elements:
        if el.type in ("text", "heading") and el.content:
            parts = sep_re.split(el.content)
            parts = [p.strip() for p in parts if p and p.strip()]
            if len(parts) >= 2:
                for pi, part in enumerate(parts):
                    h = el.bbox[3] - el.bbox[1]
                    chunk_h = h / len(parts) if h > 0 else 0
                    new_bbox = (
                        el.bbox[0],
                        el.bbox[1] + chunk_h * pi,
                        el.bbox[2],
                        el.bbox[1] + chunk_h * (pi + 1) if chunk_h > 0 else el.bbox[3],
                    )
                    new_el = LayoutElement(
                        type_=el.type,
                        content=part,
                        bbox=new_bbox,
                        parent_id=el.parent_id,
                        doc_name=el.doc_name,
                    )
                    split_elements.append(new_el)
            else:
                split_elements.append(el)
        else:
            split_elements.append(el)
    return split_elements


def preprocess_text(text, rules=None):
    """Dify 风格文本预处理规则

    默认规则集（rules=None 时全部开启）：
    - clean_whitespace:   清理连续空白行（最多 2 个 \n）、行首尾空白
    - normalize_spaces:   清理连续空格/制表符 → 单空格（中文语境跳过）
    - strip_html:         去除 HTML 标签 <xxx>
    - strip_urls:         可选：清理 http(s):// 开头的长 URL
    - strip_nbsp:         &nbsp; / 零宽字符 / 不可见 unicode

    Args:
        text: 原始文本
        rules: dict[str, bool] — 关闭某些规则，如 {"strip_urls": False}

    Returns:
        预处理后的文本
    """
    if not text or not text.strip():
        return text or ""

    rules = rules or {}

    # 1) 不可见 unicode / BOM / &nbsp;
    text = text.replace("\uFEFF", "").replace("\u00A0", " ")
    text = re.sub(r"[\u200B-\u200D\uFE0F]", "", text)

    # 2) HTML 标签
    if rules.get("strip_html", True):
        text = re.sub(r"(?is)<(script|style|noscript|pre|code)[^>]*>.*?</\1>", " ", text)
        text = re.sub(r"(?is)<[^>]+>", " ", text)

    # 3) 连续空白行 → 最多 2 个 \n
    if rules.get("clean_whitespace", True):
        text = re.sub(r"\n{3,}", "\n\n", text)

    # 4) 行首尾空白
    if rules.get("clean_whitespace", True):
        lines = text.split("\n")
        lines = [ln.rstrip() for ln in lines]
        text = "\n".join(lines)

    # 5) 连续空格 / 制表符 → 单空格（中文内容跳过：判断有无 ASCII 字母/数字）
    if rules.get("normalize_spaces", True):
        if re.search(r"[A-Za-z0-9]", text):
            text = re.sub(r"[ \t]{2,}", " ", text)

    # 6) URL（可选，默认保留）
    if rules.get("strip_urls", False):
        text = re.sub(r"https?://\S+", "", text)

    return text.strip()


# =====================================================================
# 元素数据结构
# =====================================================================

class LayoutElement:
    """统一的版面元素"""
    __slots__ = ("id", "type", "content", "bbox", "parent_id", "doc_name",
                 "anchor_to", "image_path", "image_b64",
                 "_ocr_struct", "_should_ocr", "_ocr_text")

    def __init__(self, type_, content, bbox=None, parent_id=0, doc_name=""):
        self.id = f"el_{uuid.uuid4().hex[:8]}"
        self.type = type_                # text / image / table / heading / chart
        self.content = content           # 文本内容 或 图片描述 或 表格markdown
        self.bbox = bbox or (0, 0, 0, 0) # (x0, y0, x1, y1) 像素
        self.parent_id = parent_id       # 页码 或 幻灯片号
        self.doc_name = doc_name
        self.anchor_to = None            # image 专用：配对的 text element id
        self.image_path = None           # image 专用：落盘路径
        self.image_b64 = None            # image 专用：base64（可选，避免重复读文件）
        # 图片 OCR 后处理填充（构建完 chunks 后）
        self._ocr_struct = None          # {"kind": "kv"/"table"/"text"/"none", ...}
        self._should_ocr = False         # 是否实际跑了 OCR
        self._ocr_text = ""              # OCR 原始文本（留空 = 没跑）


# =====================================================================
# 主入口
# =====================================================================

def parse_file(path):
    """兼容旧接口：返回纯文本（把 layout elements 展平）→ 预处理"""
    elements = parse_file_layout(path)
    parts = []
    # 按 parent_id + bbox 排序，保证页码顺序和页内阅读顺序
    elements.sort(key=lambda e: (e.parent_id, e.bbox[1]))
    for el in elements:
        if el.type in ("text", "heading"):
            parts.append(el.content)
        elif el.type == "table":
            parts.append(f"[表格]\n{el.content}")
        elif el.type in ("image", "chart") and el.content:
            parts.append(f"[{el.type}说明: {el.content}]")
    raw = "\n\n".join(parts)
    return preprocess_text(raw)


def parse_file_layout(path):
    """按扩展名解析为结构化元素列表"""
    ext = os.path.splitext(path)[1].lower()
    basename = os.path.basename(path)

    # .ppt → 先转 .pptx（用 LibreOffice soffice）
    if ext == ".ppt":
        path, ext = _convert_ppt_to_pptx(path)

    if ext == ".pptx":
        return _parse_pptx(path, basename)
    if ext == ".pdf":
        return _parse_pdf(path, basename)
    if ext in (".txt", ".md", ".markdown"):
        return _parse_text_layout(path, basename)
    if ext == ".docx":
        return _parse_docx_layout(path, basename)
    if ext == ".doc":
        return _parse_docx_layout(path, basename, _use_olefallback=True)
    if ext == ".xlsx":
        return _parse_xlsx_layout(path, basename)
    if ext == ".xls":
        return _parse_xlsx_layout(path, basename, _use_xls=True)
    raise ValueError(f"不支持的文件类型: {ext}")


# =====================================================================
# PPT / PPTX
# =====================================================================

def _convert_ppt_to_pptx(path):
    """老版 .ppt → .pptx（调 LibreOffice）。没装 soffice 就直接原样返回，上层会走 ole 兜底"""
    try:
        # 检查 LibreOffice
        soffice = shutil.which("soffice") or shutil.which("libreoffice")
        if not soffice:
            # Windows 常见路径
            for candidate in [r"C:\Program Files\LibreOffice\program\soffice.exe",
                              r"C:\Program Files (x86)\LibreOffice\program\soffice.exe"]:
                if os.path.exists(candidate):
                    soffice = candidate
                    break
        if not soffice:
            print("[parser] 未找到 LibreOffice，.ppt 文件将直接尝试 ole 读取（效果有限）")
            return path, ".ppt"

        outdir = tempfile.mkdtemp(prefix="pptconv_")
        import subprocess
        subprocess.run(
            [soffice, "--headless", "--convert-to", "pptx", "--outdir", outdir, path],
            capture_output=True, timeout=120
        )
        converted = os.path.join(outdir, os.path.splitext(os.path.basename(path))[0] + ".pptx")
        if os.path.exists(converted):
            print(f"[parser] .ppt → .pptx 转换成功: {converted}")
            return converted, ".pptx"
        return path, ".ppt"
    except Exception as e:
        print(f"[parser] .ppt 转换失败，直接走 ole: {e}")
        return path, ".ppt"


def _parse_pptx(path, doc_name):
    """解析 .pptx：python-pptx 原生坐标，按 slide → shape 遍历
    元素类型: heading（标题） / text（正文） / image / table / chart"""
    from pptx import Presentation
    from pptx.util import Emu

    prs = Presentation(path)
    elements = []

    # 像素换算：1 EMU = 1/914400 英寸；96 DPI → 1 英寸 = 96 px
    EMU_TO_PX = 96.0 / 914400.0

    for slide_idx, slide in enumerate(prs.slides):
        parent_id = slide_idx + 1  # 幻灯片号从 1 开始

        for shape in slide.shapes:
            # shape 类型太多（SmartArt/图表/视频/动作按钮…），单个 shape 崩了不影响其他
            try:
                # 计算 bbox（像素）
                left = shape.left * EMU_TO_PX if shape.left is not None else 0
                top = shape.top * EMU_TO_PX if shape.top is not None else 0
                width = shape.width * EMU_TO_PX if shape.width is not None else 0
                height = shape.height * EMU_TO_PX if shape.height is not None else 0
                bbox = (left, top, left + width, top + height)

                # 标题/正文文本
                if getattr(shape, "has_text_frame", False) and shape.text_frame.text.strip():
                    try:
                        is_heading = bool(getattr(shape, "is_placeholder", False)) and \
                                     "Title" in (shape.placeholder_format.type.name if
                                                 getattr(shape, "placeholder_format", None) else "")
                    except Exception:
                        is_heading = False
                    for para in shape.text_frame.paragraphs:
                        text = para.text.strip()
                        if text:
                            el = LayoutElement(
                                type_="heading" if is_heading else "text",
                                content=text, bbox=bbox,
                                parent_id=parent_id, doc_name=doc_name,
                            )
                            elements.append(el)

                # 图片
                elif shape.shape_type == 13:  # MSO_SHAPE_TYPE.PICTURE = 13
                    try:
                        img = shape.image
                        img_bytes = img.blob
                        ext = (img.content_type or "image/png").split("/")[-1]
                        img_path = _save_temp_image(img_bytes, ext)
                        el = LayoutElement(
                            type_="image", content="", bbox=bbox,
                            parent_id=parent_id, doc_name=doc_name,
                        )
                        el.image_path = img_path
                        elements.append(el)
                    except Exception:
                        pass  # 图片提取失败静默跳过（SmartArt 里的图抓不到很常见）

                # 表格
                elif getattr(shape, "has_table", False):
                    try:
                        tbl = shape.table
                        lines = []
                        for row in tbl.rows:
                            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                            lines.append("| " + " | ".join(cells) + " |")
                        if lines:
                            if len(lines) > 1:
                                cols = lines[0].count("|") - 1
                                lines.insert(1, "|" + "|".join(["---"] * cols) + "|")
                            el = LayoutElement(
                                type_="table", content="\n".join(lines), bbox=bbox,
                                parent_id=parent_id, doc_name=doc_name,
                            )
                            elements.append(el)
                    except Exception:
                        pass
            except Exception as shape_err:
                print(f"[parser] pptx shape 跳过（type={getattr(shape, 'shape_type', '?')}）: {shape_err}")
                continue

    # 图文配对（同一 slide 内按 top 排序）
    _pair_images_to_text(elements, by_parent=True)
    return elements


# =====================================================================
# PDF（电子版 + 扫描件 OCR 兜底）
# =====================================================================

def _parse_pdf(path, doc_name):
    """解析 PDF：优先 MinerU（conda 环境），失败则 pdfplumber + RapidOCR 兜底

    MinerU 走 subprocess 调独立 conda 环境，返回结构化 markdown，
    从中解析出 heading / text / table 三种 LayoutElement。
    """
    # ---- 优先 MinerU ----
    mineru_elements = _try_parse_pdf_mineru(path, doc_name)
    if mineru_elements is not None:
        return mineru_elements

    # ---- 兜底：pdfplumber（电子版）+ RapidOCR（扫描件）----
    print("[parser] MinerU 未启用/失败，走 pdfplumber + RapidOCR 兜底...")
    return _parse_pdf_fallback(path, doc_name)


def _try_parse_pdf_mineru(path, doc_name):
    """尝试用 MinerU 解析 PDF。成功返回 elements，失败返回 None"""
    try:
        import subprocess
        import json

        bridge_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mineru_bridge.py")
        if not os.path.exists(bridge_script):
            print("[parser] mineru_bridge.py 不存在，跳过 MinerU")
            return None

        # subprocess 调 conda 环境的 mineru bridge
        # timeout 180s：首次加载模型 + GPU 推理，3 页 PDF 约 30-60s
        result = subprocess.run(
            ["conda", "run", "-n", "mineru", "python", bridge_script, path],
            capture_output=True, text=True, timeout=180, encoding="utf-8",
        )

        if result.returncode != 0:
            print(f"[parser] MinerU 进程退出码 {result.returncode}: {result.stderr[:300]}")
            return None

        # mineru bridge 输出格式：JSON 到 stdout
        # 但 conda run 可能在 stdout 混入 loguru 日志 → 取最后一行 JSON
        output = result.stdout.strip()
        json_line = output
        for line in reversed(output.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                json_line = line
                break

        data = json.loads(json_line)
        if not data.get("ok"):
            print(f"[parser] MinerU 解析失败: {data.get('error', '未知')}")
            return None

        md = data.get("markdown", "")
        if not md.strip():
            print("[parser] MinerU 返回空 markdown")
            return None

        # 把 MinerU markdown 解析成 LayoutElement
        elements = _mineru_md_to_elements(md, doc_name)
        print(f"[parser] MinerU 解析成功: {len(elements)} 个元素")
        return elements

    except subprocess.TimeoutExpired:
        print("[parser] MinerU 超时（180s）")
        return None
    except FileNotFoundError:
        print("[parser] conda 命令未找到")
        return None
    except Exception as e:
        print(f"[parser] MinerU 异常: {type(e).__name__}: {e}")
        return None


def _mineru_md_to_elements(md, doc_name):
    """把 MinerU 输出的 markdown 解析成 LayoutElement 列表

    MinerU markdown 特点:
    - ## / ### 开头 = 标题
    - 空行分段
    - | 开头 = 表格
    - 没有页码信息（mineru bridge 过滤了图片，也没返回 page mapping）
    """
    elements = []
    parent_id = 1  # mineru 不返回页码，统一设 1
    lines = md.splitlines()

    i = 0
    while i < len(lines):
        line = lines[i].rstrip()

        # 跳过空行
        if not line.strip():
            i += 1
            continue

        # 标题行：## / ### / ####
        heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading_match:
            content = heading_match.group(2).strip()
            if content:
                elements.append(LayoutElement(
                    type_="heading", content=content,
                    bbox=(0, 0, 0, 0), parent_id=parent_id, doc_name=doc_name,
                ))
            i += 1
            continue

        # 表格：检测表格行（含 | ... | 结构）
        if "|" in line and line.strip().startswith("|"):
            table_lines = []
            while i < len(lines):
                tl = lines[i].rstrip()
                if "|" in tl and tl.strip().startswith("|"):
                    table_lines.append(tl)
                    i += 1
                else:
                    break
            # 跳过 markdown 表格分隔行（|---|---|）
            table_content_lines = [tl for tl in table_lines if not re.match(r'^\|[\s\-:|]+\|$', tl)]
            if table_content_lines:
                # 简单转成 markdown table
                elements.append(LayoutElement(
                    type_="table", content="\n".join(table_content_lines),
                    bbox=(0, 0, 0, 0), parent_id=parent_id, doc_name=doc_name,
                ))
            continue

        # 普通文本行：收集连续非空行（段落）
        text_lines = [line]
        i += 1
        while i < len(lines):
            nl = lines[i].rstrip()
            if not nl.strip():
                break
            if re.match(r'^#{1,6}\s+', nl):
                break
            if nl.strip().startswith("|") and "|" in nl:
                break
            text_lines.append(nl)
            i += 1

        text_content = "\n".join(text_lines).strip()
        if text_content:
            elements.append(LayoutElement(
                type_="text", content=text_content,
                bbox=(0, 0, 0, 0), parent_id=parent_id, doc_name=doc_name,
            ))

    return elements


def _parse_pdf_fallback(path, doc_name):
    """原有 pdfplumber + RapidOCR 方案（不变）"""
    try:
        import pdfplumber
    except Exception as exc:
        raise ImportError("解析 PDF 需要 pdfplumber，请执行：pip install pdfplumber") from exc

    elements = []
    scan_fallback = False

    with pdfplumber.open(path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            parent_id = page_idx + 1

            # 1. 尝试抽 word-level bbox（电子版才有）
            words = page.extract_words(x_tolerance=2, y_tolerance=3)
            page_text = page.extract_text() or ""

            if words:
                # 能抽到 bbox → 电子版，按 y 坐标聚类成行
                lines_by_y = {}
                for w in words:
                    y_key = round(w["top"], 0)
                    if y_key not in lines_by_y:
                        lines_by_y[y_key] = {"text": "", "bbox": [99999, 99999, 0, 0]}
                    ln = lines_by_y[y_key]
                    ln["text"] += (" " if ln["text"] else "") + w["text"]
                    ln["bbox"][0] = min(ln["bbox"][0], w["x0"])
                    ln["bbox"][1] = min(ln["bbox"][1], w["top"])
                    ln["bbox"][2] = max(ln["bbox"][2], w["x1"])
                    ln["bbox"][3] = max(ln["bbox"][3], w["bottom"])

                for y in sorted(lines_by_y.keys()):
                    ln = lines_by_y[y]
                    text = ln["text"].strip()
                    if not text:
                        continue
                    is_heading = _looks_like_heading(text)
                    el = LayoutElement(
                        type_="heading" if is_heading else "text",
                        content=text, bbox=tuple(ln["bbox"]),
                        parent_id=parent_id, doc_name=doc_name,
                    )
                    elements.append(el)
            elif page_text.strip():
                for line in page_text.splitlines():
                    line = line.strip()
                    if line:
                        is_heading = _looks_like_heading(line)
                        el = LayoutElement(
                            type_="heading" if is_heading else "text",
                            content=line, bbox=(0, 0, 0, 0),
                            parent_id=parent_id, doc_name=doc_name,
                        )
                        elements.append(el)
            else:
                scan_fallback = True
                break

            # 2. 表格
            tables = page.extract_tables()
            for tbl in tables:
                if not tbl:
                    continue
                lines = []
                for row in tbl:
                    cells = [(c or "").strip().replace("\n", " ") for c in row]
                    lines.append("| " + " | ".join(cells) + " |")
                if lines:
                    cols = lines[0].count("|") - 1
                    lines.insert(1, "|" + "|".join(["---"] * cols) + "|")
                    el = LayoutElement(
                        type_="table", content="\n".join(lines),
                        bbox=(0, 0, 0, 0), parent_id=parent_id, doc_name=doc_name,
                    )
                    elements.append(el)

            # 3. 图片
            for img_info in page.images:
                try:
                    img_bytes = img_info.get("stream", b"") or b""
                    if not img_bytes:
                        continue
                    bbox = (
                        img_info.get("x0", 0), img_info.get("top", 0),
                        img_info.get("x1", 0), img_info.get("bottom", 0),
                    )
                    img_path = _save_temp_image(img_bytes, "png")
                    el = LayoutElement(
                        type_="image", content="", bbox=bbox,
                        parent_id=parent_id, doc_name=doc_name,
                    )
                    el.image_path = img_path
                    elements.append(el)
                except Exception as e:
                    print(f"[parser] pdf 图片提取失败: {e}")

        if not scan_fallback:
            _pair_images_to_text(elements, by_parent=True)
            return elements

    # ---- 扫描件兜底：RapidOCR ----
    print("[parser] PDF 无文本层，启动 RapidOCR 扫描件兜底...")
    return _parse_pdf_scanned(path, doc_name)


def _parse_pdf_scanned(path, doc_name):
    """RapidOCR 兜底：先把 PDF 每页渲染成 PNG → OCR → 抽文字 bbox → 配对"""
    try:
        from rapidocr_onnxruntime import RapidOCR
        import fitz  # PyMuPDF
    except Exception as e:
        print(f"[parser] 扫描件依赖缺失，返回空：{e}")
        return []

    ocr = RapidOCR()
    pdf = fitz.open(path)
    elements = []

    for page_idx in range(len(pdf)):
        parent_id = page_idx + 1
        page = pdf[page_idx]
        # 渲染成 PNG（2x 缩放让 OCR 更准）
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        png_bytes = pix.tobytes("png")
        img_path = _save_temp_image(png_bytes, "png")

        # OCR：返回 [bbox_quad, text, confidence]
        result, _ = ocr(img_path)
        if not result:
            continue
        for item in result:
            quad, text, conf = item
            # quad 是四点四边形 [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
            xs = [p[0] for p in quad]
            ys = [p[1] for p in quad]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            text = text.strip()
            if not text:
                continue
            is_heading = _looks_like_heading(text)
            el = LayoutElement(
                type_="heading" if is_heading else "text",
                content=text, bbox=bbox,
                parent_id=parent_id, doc_name=doc_name,
            )
            elements.append(el)

        # 渲染出的整页图：也存一张，后续视觉模型分析
        page_el = LayoutElement(
            type_="image", content="", bbox=(0, 0, pix.width, pix.height),
            parent_id=parent_id, doc_name=doc_name,
        )
        page_el.image_path = img_path
        elements.append(page_el)

    pdf.close()
    _pair_images_to_text(elements, by_parent=True)
    return elements


# =====================================================================
# 简单格式（txt / md / docx / xlsx）
# =====================================================================

def _parse_text_layout(path, doc_name):
    """纯文本/Markdown：按段落切成 heading / text"""
    raw = open(path, "rb").read()
    text = None
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="ignore")

    elements = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if not para:
            continue
        # Markdown heading: # / ## / ### ...
        if re.match(r"^#{1,6}\s+", para):
            content = re.sub(r"^#{1,6}\s+", "", para)
            elements.append(LayoutElement("heading", content, parent_id=1, doc_name=doc_name))
        else:
            elements.append(LayoutElement("text", para, parent_id=1, doc_name=doc_name))
    return elements


def _parse_docx_layout(path, doc_name, _use_olefallback=False):
    """Word：段落 + 表格 + 图片"""
    try:
        from docx import Document
    except Exception:
        if _use_olefallback:
            return _parse_doc_via_ole(path, doc_name)
        raise

    doc = Document(path)
    elements = []

    for p in doc.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        style = (p.style.name or "") if p.style else ""
        is_heading = "Heading" in style
        elements.append(LayoutElement(
            type_="heading" if is_heading else "text",
            content=text, parent_id=1, doc_name=doc_name,
        ))

    for tbl in doc.tables:
        lines = []
        for row in tbl.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            lines.append("| " + " | ".join(cells) + " |")
        if lines:
            cols = lines[0].count("|") - 1
            lines.insert(1, "|" + "|".join(["---"] * cols) + "|")
            elements.append(LayoutElement(
                type_="table", content="\n".join(lines),
                parent_id=1, doc_name=doc_name,
            ))

    # 内嵌图片
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            try:
                img_bytes = rel.target_part.blob
                ext = os.path.splitext(rel.target_part.content_type or "image/png")[-1]
                if not ext or "/" in ext:
                    ext = "png"
                img_path = _save_temp_image(img_bytes, ext)
                el = LayoutElement("image", "", parent_id=1, doc_name=doc_name)
                el.image_path = img_path
                elements.append(el)
            except Exception:
                pass

    return elements


def _parse_doc_via_ole(path, doc_name):
    """老版 .doc 尽力而为的 olefile 兜底"""
    import olefile
    elements = []
    if not olefile.OleFileIO(path).exists("WordDocument"):
        raise ValueError("不是有效的 .doc 文件")
    with olefile.OleFileIO(path) as ole:
        raw = ole.openstream("WordDocument").read()
    text = raw.decode("gb18030", errors="ignore")
    text = re.sub(r"[\x00-\x08\x0b-\x1f\uf000-\uffff&&[^\u4e00-\u9fff]]", " ", text)
    text = re.sub(r"\s{2,}", "\n", text)
    for ln in text.splitlines():
        ln = ln.strip()
        if len(ln) >= 2:
            elements.append(LayoutElement("text", ln, parent_id=1, doc_name=doc_name))
    return elements


def _parse_xlsx_layout(path, doc_name, _use_xls=False):
    """Excel：每个 sheet 的表头 + 数据行"""
    if _use_xls:
        import xlrd
        wb = xlrd.open_workbook(path)
        sheets = [(ws.name, [[str(ws.cell_value(r, c)) for c in range(ws.ncols)]
                              for r in range(ws.nrows)]) for ws in wb.sheets()]
    else:
        from openpyxl import load_workbook
        wb = load_workbook(path, read_only=True, data_only=True)
        sheets = [(ws.title, list(ws.iter_rows(values_only=True))) for ws in wb.worksheets]
        wb.close()

    elements = []
    for sname, rows in sheets:
        if not rows:
            continue
        lines = []
        for row in rows:
            cells = [str(c).strip().replace("\n", " ") for c in row if c is not None]
            if not cells:
                continue
            lines.append("| " + " | ".join(cells) + " |")
        if lines:
            cols = lines[0].count("|") - 1
            lines.insert(1, "|" + "|".join(["---"] * cols) + "|")
            elements.append(LayoutElement(
                type_="heading", content=f"工作表: {sname}",
                parent_id=1, doc_name=doc_name,
            ))
            elements.append(LayoutElement(
                type_="table", content="\n".join(lines),
                parent_id=1, doc_name=doc_name,
            ))
    return elements


# =====================================================================
# 图文配对
# =====================================================================

def _pair_images_to_text(elements, by_parent=True, max_dist_px=500):
    """对所有 image 元素找 anchor_to：同 parent 空间距离最近的 text/heading

    max_dist_px: 图像和配对文字的最大允许像素距离
      - PPT 原生坐标：图片和说明文字一般 < 100px，宽松阈值足够
      - PDF 电子版：pdfplumber words bbox 很准
      - 扫描件 OCR：bbox 有轻微偏移，用更宽松阈值
    """
    images = [e for e in elements if e.type == "image"]
    texts = [e for e in elements if e.type in ("text", "heading")]

    for img in images:
        if by_parent:
            candidates = [t for t in texts if t.parent_id == img.parent_id]
        else:
            candidates = texts
        if not candidates:
            continue

        def dist(t):
            t0, t1 = t.bbox[0], t.bbox[3]   # 顶部 x 和 底部 y
            img0, img1 = img.bbox[1], img.bbox[3]   # 图的 top 和 bottom y
            if t0 >= img1:
                return t0 - img1
            if img0 >= t1:
                return img0 - t1
            return 0  # 有重叠

        best = min(candidates, key=dist)
        d = dist(best)
        if d < max_dist_px:
            img.anchor_to = best.id


# =====================================================================
# 版面感知分块
# =====================================================================

class LayoutChunk:
    """一个分块：按 heading 边界切，聚合 text/image/table"""
    __slots__ = ("heading", "elements", "parent_id", "doc_name")

    def __init__(self, doc_name="", parent_id=0):
        self.heading = ""
        self.elements = []  # LayoutElement 列表（按 top 排序）
        self.parent_id = parent_id
        self.doc_name = doc_name

    @property
    def word_count(self):
        return sum(len(e.content) for e in self.elements if e.type in ("text", "heading"))

    @property
    def image_count(self):
        return sum(1 for e in self.elements if e.type == "image")

    @property
    def image_paths(self):
        return [e.image_path for e in self.elements if e.type == "image" and e.image_path]

    def to_text(self):
        """把块内所有元素展平成喂模型的纯文本"""
        parts = []
        if self.heading:
            parts.append(f"## {self.heading}")
        # 按 parent_id + bbox 排序
        sorted_els = sorted(self.elements, key=lambda e: (e.parent_id, e.bbox[1]))
        for el in sorted_els:
            if el.type == "heading" and el.content != self.heading:
                parts.append(f"### {el.content}")
            elif el.type == "text":
                parts.append(el.content)
            elif el.type == "table":
                parts.append(el.content)
            elif el.type in ("image", "chart") and el.content:
                parts.append(f"[图片说明: {el.content}]")
        return "\n".join(parts)


def build_layout_chunks(elements, parent_size=None, max_images=None,
                         parent_separator=None, child_separator=None):
    """按 parent_separator + heading 边界 + 跨 parent_id + 硬上限切父块

    触发新父块的条件（优先级）:
    1. parent_separator —— 用户自定义父块分隔符（regex / 字符串）
       ① 先把所有 text/heading element 的 content 按 parent_separator **预先拆成多个 element**
       ② 拆完后每个新 element 独立处理，命中就切新父块
    2. heading 边界 —— 当前块已有 ≥100 字正文 + 遇到新 heading
    3. 跨 parent_id（PDF 跨页 / PPT 跨 slide）
    4. 硬上限：parent_size / max_images
    """
    if not elements:
        return []

    parent_size = parent_size or config.LAYOUT_MAX_WORDS
    max_images = max_images or config.LAYOUT_MAX_IMAGES

    # 兼容旧名 custom_separator（被调用方还在传）
    if not parent_separator and child_separator is None:
        # 旧调用没传任何分隔符 → 保持原样
        pass

    # ============ 预处理：先清理 element.content 里的脏字符 ============
    for el in elements:
        if el.type in ("text", "heading") and el.content:
            el.content = preprocess_text(el.content)

    # ============ 过滤：纯 separator / 太短的独立 element 跳过 ============
    # 防止 separator 作为独立段落被切出空 chunk
    if parent_separator:
        sep_re = _compile_separator(parent_separator)
        if sep_re:
            elements = [
                el for el in elements
                if not (el.type in ("text", "heading")
                        and len(el.content.strip()) < len(parent_separator) + 5
                        and sep_re.fullmatch(el.content.strip()))
            ]

    # ============ parent_separator 预先拆分 text element ============
    if parent_separator:
        sep_re = _compile_separator(parent_separator)
        if sep_re:
            elements = _split_elements_by_separator(elements, sep_re)

    elements_sorted = sorted(elements, key=lambda e: (e.parent_id, e.bbox[1]))
    chunks = []
    current = None

    def new_chunk(el):
        nonlocal current
        current = LayoutChunk(doc_name=el.doc_name, parent_id=el.parent_id)
        chunks.append(current)

    for el in elements_sorted:
        if current is None:
            new_chunk(el)

        force_new = False

        # 1) parent_separator（预先已经把 element 拆好了，这里简化：只要 parent_sep 存在且当前块非空就切）
        if parent_separator and current.elements:
            force_new = True

        # 2) heading + 当前块已有足够正文
        if not force_new and el.type == "heading" and current.elements and current.word_count > 100:
            force_new = True

        # 3) 跨 parent_id
        if not force_new and current.parent_id != el.parent_id and current.elements:
            force_new = True

        # 4) 硬上限
        if not force_new and (current.word_count >= parent_size or current.image_count >= max_images):
            force_new = True

        if force_new:
            new_chunk(el)

        # heading 元素 → 作为块标题（如果当前块还没标题）
        if el.type == "heading":
            if not current.heading:
                current.heading = el.content
            else:
                # 已有标题，把这个 heading 降级为普通 text 元素
                current.elements.append(el)
        else:
            current.elements.append(el)

    # 过滤掉空块
    chunks = [c for c in chunks if c.word_count > 0 or c.image_count > 0]

    # 后处理：对每个 chunk 的图片做"判断 → 懒 OCR → 结构化 → 填 content"
    for ch in chunks:
        # 聚合该 chunk 内所有 heading + text 的文字（用于"附近文字覆盖度"判断）
        chunk_text_all = " ".join(
            el.content for el in ch.elements
            if el.type in ("heading", "text") and el.content
        )
        if ch.heading:
            chunk_text_all = (ch.heading + " " + chunk_text_all).strip()

        for img_el in ch.elements:
            if img_el.type != "image":
                continue
            try:
                should = _should_ocr(img_el, chunk_text_all)
                ocr_text = ""
                ocr_struct = {"kind": "none"}
                if should:
                    ocr_text = _ocr_image(img_el.image_path)
                    if ocr_text:
                        ocr_struct = _ocr_to_structured(ocr_text)
                # 把 ocr_struct 挂到 element 上（layout_chunks_to_records 要写进 metadata）
                img_el._ocr_struct = ocr_struct
                img_el._should_ocr = should
                img_el._ocr_text = ocr_text
                # 填 content（喂向量检索用）
                img_el.content = _build_image_content(img_el, ocr_struct, chunk_text_all)
            except Exception as e:
                print(f"[parser] 图片后处理跳过: {e}")
                img_el.content = ""
                img_el._ocr_struct = {"kind": "none"}
                img_el._should_ocr = False

    return chunks


# =====================================================================
# 父子分块核心：父块（大窗口给 LLM）→ 子块（小窗口 + overlap 给向量检索）
# =====================================================================

def _split_parent_into_children(parent_text, child_size=500, overlap=50,
                                 child_sep=None):
    """把父块文本切成多个子块，相邻子块重叠 overlap 字

    切分优先级:
    1. child_sep —— 用户自定义子块分隔符（regex / 字面量）
    2. 句子边界（中文句号/问号/感叹号 + 换行）—— 默认
    3. 强制字符数切（child_size 硬上限兜底）

    Args:
        parent_text: 父块完整文本
        child_size: 子块目标最大字数
        overlap: 相邻子块重叠字数（父子模式传 0，通用模式传 50）
        child_sep: 用户自定义子块分隔符

    Returns:
        list[str]: 子块文本列表，至少 1 个
    """
    if not parent_text.strip():
        return []

    # 智能 child_size：如果父块本身 ≤ child_size，强制缩到父块 50% 尝试再切
    child_size = max(50, child_size)
    if len(parent_text) <= child_size:
        child_size = max(50, len(parent_text) // 2)

    overlap = max(0, min(overlap, child_size // 2))

    # 第一步：确定切分边界
    sep_re = _compile_separator(child_sep, hint=parent_text[:50]) if child_sep else None
    if sep_re:
        parts = sep_re.split(parent_text)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            sentences = parts
        else:
            # 分隔符没起到作用 → 降级到句子边界
            sentences = re.split(r'(?<=[。！？!?\n])', parent_text)
            sentences = [s.strip() for s in sentences if s.strip()]
    else:
        sentences = re.split(r'(?<=[。！？!?\n])', parent_text)
        sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return [parent_text]

    # 第二步：累计进子块
    children = []
    buf = ""
    for sent in sentences:
        if len(buf) + len(sent) <= child_size:
            buf += sent
        else:
            if buf:
                children.append(buf)
            if len(sent) > child_size:
                # 单句超限 → 硬切
                for i in range(0, len(sent), child_size):
                    children.append(sent[i:i + child_size])
                buf = ""
            else:
                buf = sent
    if buf:
        children.append(buf)

    # 兜底：如果还是只有 1 个子块，强制按 child_size 硬切
    # （极短父块 + 无句号/换行的场景 —— 比如只有 heading 和数字）
    if len(children) == 1 and len(children[0]) > child_size:
        only = children[0]
        children = [only[i:i + child_size] for i in range(0, len(only), child_size)]

    # 最终兜底：还是只有 1 个子块，强制从中间劈两半（确保父子分块至少有 2 层）
    if len(children) == 1 and len(children[0]) >= 20:
        mid = len(children[0]) // 2
        children = [children[0][:mid], children[0][mid:]]

    # 第三步：overlap
    if overlap > 0 and len(children) > 1:
        overlapped = [children[0]]
        for i in range(1, len(children)):
            prev_tail = children[i - 1][-overlap:] if len(children[i - 1]) > overlap else children[i - 1]
            overlapped.append(prev_tail + children[i])
        children = overlapped

    return children


# =====================================================================
# 持久化：LayoutChunk 列表 → chunks.json 记录格式
# =====================================================================

def _normalize_chunk_config(cfg):
    """把任意前端传来的 chunk_config 归一化成内部统一结构（兼容新旧字段）

    内部统一结构：
        {
            "mode": "general" | "parent_child",
            "separator": ...,        # 通用模式分隔符
            "max_length": ...,        # 通用模式最大长度
            "overlap": ...,           # 通用模式 overlap
            "parent_strategy": "paragraph" | "document",  # 父子模式策略
            "parent_separator": ...,  # 父子·段落模式父块分隔符
            "parent_max_length": ..., # 父子·段落模式父块最大长度
            "child_separator": ...,   # 父子模式子块分隔符
            "child_max_length": ...,  # 父子模式子块最大长度
        }
    """
    cfg = cfg or {}

    # ---- 判断 mode ----
    # 新字段 mode 或旧字段 split_mode
    mode = cfg.get("mode") or cfg.get("split_mode") or "general"
    if mode == "flat":
        mode = "general"
    if mode not in ("general", "parent_child"):
        mode = "general"

    if mode == "parent_child":
        # ---- 父子模式 ----
        parent_strategy = cfg.get("parent_strategy") or "paragraph"
        if parent_strategy not in ("paragraph", "document"):
            parent_strategy = "paragraph"

        parent_sep = cfg.get("parent_separator") or cfg.get("custom_separator") or None
        parent_max = cfg.get("parent_max_length") or cfg.get("parent_size") or 1024
        child_sep = cfg.get("child_separator") or None
        child_max = cfg.get("child_max_length") or cfg.get("child_size") or 300

        return {
            "mode": "parent_child",
            "parent_strategy": parent_strategy,
            "parent_separator": parent_sep,
            "parent_max_length": parent_max,
            "child_separator": child_sep,
            "child_max_length": child_max,
            # 父子模式 overlap 强制 0
            "overlap": 0,
        }
    else:
        # ---- 通用模式 ----
        sep = cfg.get("separator") or cfg.get("custom_separator") or None
        max_len = cfg.get("max_length") or cfg.get("child_size") or cfg.get("parent_size") or 500
        overlap = cfg.get("overlap", 50)
        return {
            "mode": "general",
            "separator": sep,
            "max_length": max_len,
            "overlap": overlap,
        }


def layout_chunks_to_records(chunks, chunk_config=None):
    """把 LayoutChunk 列表转换成 chunks.json 的记录格式 + 图片路径列表

    两种模式**严格二选一**，互不干扰：
    ┌──────────────────────────────────────────────────────────────┐
    │ 通用模式 general：单层 flat                                    │
    │   - chunk_texts = 单层切片（max_length + overlap + separator）│
    │   - parent_texts = None   （完全没父块）                        │
    │   - parent_ids   = None                                       │
    ├──────────────────────────────────────────────────────────────┤
    │ 父子模式 parent_child：两层                                   │
    │   - chunk_texts  = 子块（给向量检索）                            │
    │   - parent_texts = 父块（检索命中后喂 LLM）                     │
    │   - parent_ids   = 父块唯一 ID（去重用）                         │
    │   - overlap 强制 0（Dify 逻辑：父块兜底了）                      │
    └──────────────────────────────────────────────────────────────┘

    Returns:
        chunk_texts: list[str]        # 文本给向量编码
        parent_texts: list[str]|None  # 通用模式=None；父子模式=每个子块的父块文本
        chunk_images: list[str|None]
        layout_elements: list[list]
        parent_ids: list[str]|None    # 通用模式=None；父子模式=父块 ID
    """
    cfg = _normalize_chunk_config(chunk_config)
    mode = cfg["mode"]

    # ========== 通用模式：单层 flat，完全没父子 ==========
    if mode == "general":
        sep = cfg.get("separator") or None
        max_len = cfg["max_length"]
        overlap = cfg.get("overlap", 50)

        chunk_texts = []
        chunk_images = []
        layout_elements = []

        for ch in chunks:
            # 图片持久化（flat 里图片直接跟着切片走）
            persisted_images = []
            src_to_dst = {}
            if ch.image_paths:
                for src in ch.image_paths:
                    if not src or not os.path.exists(src):
                        continue
                    import uuid as _uuid
                    dst = os.path.join(config.KB_IMG_DIR, f"{_uuid.uuid4().hex}{os.path.splitext(src)[1]}")
                    try:
                        shutil.copy2(src, dst)
                        persisted_images.append(dst)
                        src_to_dst[src] = dst
                    except Exception:
                        pass
            images_str = ";".join(persisted_images) if persisted_images else None

            # layout elements 元数据
            el_data = []
            for el in ch.elements:
                rec = {"type": el.type, "content": el.content,
                       "parent_id": el.parent_id, "anchor_to": el.anchor_to}
                if el.type == "image":
                    new_path = src_to_dst.get(el.image_path, el.image_path)
                    rec["image_path"] = new_path
                    ocr_struct = getattr(el, "_ocr_struct", {"kind": "none"})
                    rec["ocr_kind"] = ocr_struct.get("kind", "none")
                    if ocr_struct.get("kind") == "kv":
                        rec["ocr_specs"] = ocr_struct.get("pairs", [])
                    elif ocr_struct.get("kind") == "table":
                        rec["ocr_table"] = {"headers": ocr_struct.get("headers", []),
                                             "rows": ocr_struct.get("rows", [])}
                el_data.append(rec)

            # 用 _split_parent_into_children 做 flat 切分（传 overlap 生效）
            flat_pieces = _split_parent_into_children(ch.to_text(), max_len, overlap, child_sep=sep)
            for piece in flat_pieces:
                chunk_texts.append(piece)
                chunk_images.append(images_str)
                layout_elements.append(el_data)

        # ← 关键：通用模式 parent_texts 和 parent_ids 直接返回 None
        return chunk_texts, None, chunk_images, layout_elements, None

    # ========== 父子模式 ==========
    child_size = cfg["child_max_length"]
    child_overlap = 0  # Dify 逻辑：父子模式 overlap 强制 0
    child_sep = cfg.get("child_separator") or None
    parent_strategy = cfg.get("parent_strategy", "paragraph")

    chunk_texts = []
    parent_texts = []
    chunk_images = []
    layout_elements = []
    parent_ids = []

    # ---- 父子·全文模式：整篇文档 = 1 个父块 ----
    if parent_strategy == "document":
        if chunks:
            all_text = "\n\n".join(ch.to_text() for ch in chunks if ch.to_text().strip())
            all_images, all_el_data, src_to_dst = [], [], {}
            for ch in chunks:
                if ch.image_paths:
                    for src in ch.image_paths:
                        if not src or not os.path.exists(src):
                            continue
                        import uuid as _uuid
                        dst = os.path.join(config.KB_IMG_DIR, f"{_uuid.uuid4().hex}{os.path.splitext(src)[1]}")
                        try:
                            shutil.copy2(src, dst)
                            all_images.append(dst)
                            src_to_dst[src] = dst
                        except Exception:
                            pass
                for el in ch.elements:
                    rec = {"type": el.type, "content": el.content,
                           "parent_id": el.parent_id, "anchor_to": el.anchor_to}
                    if el.type == "image":
                        new_path = src_to_dst.get(el.image_path, el.image_path)
                        rec["image_path"] = new_path
                        ocr_struct = getattr(el, "_ocr_struct", {"kind": "none"})
                        rec["ocr_kind"] = ocr_struct.get("kind", "none")
                        if ocr_struct.get("kind") == "kv":
                            rec["ocr_specs"] = ocr_struct.get("pairs", [])
                        elif ocr_struct.get("kind") == "table":
                            rec["ocr_table"] = {"headers": ocr_struct.get("headers", []),
                                                 "rows": ocr_struct.get("rows", [])}
                    all_el_data.append(rec)

            parent_text = all_text.strip()
            parent_id = f"parent_document_{chunks[0].doc_name}"
            images_str = ";".join(all_images) if all_images else None

            children = _split_parent_into_children(parent_text, child_size, child_overlap,
                                                    child_sep=child_sep)
            for child in children:
                chunk_texts.append(child)
                parent_texts.append(parent_text)
                chunk_images.append(images_str)
                layout_elements.append(all_el_data)
                parent_ids.append(parent_id)
            return chunk_texts, parent_texts, chunk_images, layout_elements, parent_ids
        else:
            return [], None, [], [], None

    # ---- 父子·段落模式：每个 layout chunk = 一个父块 ----
    for ch in chunks:
        parent_text = ch.to_text()
        parent_id = f"parent_{ch.doc_name}_{ch.heading or ''}_{len(ch.elements)}"

        children = _split_parent_into_children(parent_text, child_size, child_overlap,
                                                child_sep=child_sep)
        if not children:
            continue

        persisted_images, src_to_dst = [], {}
        if ch.image_paths:
            for src in ch.image_paths:
                if not src or not os.path.exists(src):
                    continue
                import uuid as _uuid
                dst = os.path.join(config.KB_IMG_DIR, f"{_uuid.uuid4().hex}{os.path.splitext(src)[1]}")
                try:
                    shutil.copy2(src, dst)
                    persisted_images.append(dst)
                    src_to_dst[src] = dst
                except Exception:
                    pass
        images_str = ";".join(persisted_images) if persisted_images else None

        el_data = []
        for el in ch.elements:
            rec = {"type": el.type, "content": el.content,
                   "parent_id": el.parent_id, "anchor_to": el.anchor_to}
            if el.type == "image":
                new_path = src_to_dst.get(el.image_path, el.image_path)
                rec["image_path"] = new_path
                ocr_struct = getattr(el, "_ocr_struct", {"kind": "none"})
                rec["ocr_kind"] = ocr_struct.get("kind", "none")
                if ocr_struct.get("kind") == "kv":
                    rec["ocr_specs"] = ocr_struct.get("pairs", [])
                elif ocr_struct.get("kind") == "table":
                    rec["ocr_table"] = {"headers": ocr_struct.get("headers", []),
                                         "rows": ocr_struct.get("rows", [])}
            el_data.append(rec)

        for child in children:
            chunk_texts.append(child)
            parent_texts.append(parent_text)
            chunk_images.append(images_str)
            layout_elements.append(el_data)
            parent_ids.append(parent_id)

    return chunk_texts, parent_texts, chunk_images, layout_elements, parent_ids


# =====================================================================
# 实时预览（前端上传弹窗用，不写文件不持久化，纯内存模拟切分）
# =====================================================================

def preview_chunks(text, chunk_config=None, preprocess_rules=None, max_preview_blocks=80):
    """接收原始文本 + chunk_config → 返回前端预览需要的结构

    Args:
        text: 原始文本（已从文件读取）
        chunk_config: 归一化前或后的 chunk_config dict
        preprocess_rules: dict[str, bool] 预处理规则开关
        max_preview_blocks: 最多预览多少个 block（防止超大文件卡前端）

    Returns:
        {
            mode: "general" | "parent_child",
            stats: { total_blocks, avg_length, total_chars },
            blocks: [...],                    # 通用模式：flat 列表
            parents: [...],                   # 父子模式：父块 + 子块层级
        }
    """
    if not text or not text.strip():
        return {"mode": "general", "stats": {"total_blocks": 0, "avg_length": 0, "total_chars": 0},
                "blocks": [], "parents": [], "error": "文件为空"}

    # 1) 预处理
    rules = preprocess_rules or {}
    if any(rules.values()):
        text = preprocess_text(text, rules if rules else None)
    text = text.strip()

    cfg = _normalize_chunk_config(chunk_config or {})
    mode = cfg["mode"]

    if mode == "general":
        sep = cfg.get("separator") or None
        max_len = cfg["max_length"]
        overlap = cfg.get("overlap", 50)

        # 用 separator 先拆大段
        sep_re = _compile_separator(sep) if sep else None
        if sep_re:
            raw_parts = [p.strip() for p in sep_re.split(text) if p.strip()]
        else:
            raw_parts = [text]

        blocks = []
        for part in raw_parts:
            if len(part) <= max_len:
                blocks.append(part)
            else:
                sub = _split_parent_into_children(part, max_len, overlap, child_sep=sep)
                blocks.extend(sub)

        # 截断
        total = len(blocks)
        preview_blocks = blocks[:max_preview_blocks]
        if len(blocks) > max_preview_blocks:
            tail_omitted = len(blocks) - max_preview_blocks
        else:
            tail_omitted = 0

        char_total = sum(len(b) for b in blocks)
        avg = round(char_total / total) if total else 0

        return {
            "mode": "general",
            "stats": {"total_blocks": total, "avg_length": avg, "total_chars": char_total},
            "blocks": [{"index": i + 1, "text": b, "length": len(b)} for i, b in enumerate(preview_blocks)],
            "parents": [],
            "tail_omitted": tail_omitted,
        }

    # ========== 父子模式 ==========
    child_size = cfg["child_max_length"]
    child_sep = cfg.get("child_separator") or None
    parent_strategy = cfg.get("parent_strategy", "paragraph")
    parent_sep = cfg.get("parent_separator") or None
    parent_max_len = cfg.get("parent_max_length", 1024)

    parents_out = []

    if parent_strategy == "document":
        # 全文模式：整篇 = 1 个父块
        parent_text = text
        children = _split_parent_into_children(parent_text, child_size, 0, child_sep=child_sep)
        preview_children = children[:max_preview_blocks]
        parents_out.append({
            "index": 1,
            "parent_text": parent_text[:2000] + ("..." if len(parent_text) > 2000 else ""),
            "length": len(parent_text),
            "children": [{"child_index": i + 1, "text": c, "length": len(c)}
                         for i, c in enumerate(preview_children)],
            "total_children": len(children),
        })
    else:
        # 段落模式：parent_separator 先拆父块
        if parent_sep:
            sep_re = _compile_separator(parent_sep)
            raw_parents = [p.strip() for p in sep_re.split(text) if p.strip()]
        else:
            # 没填 parent_separator → 按空行 + heading 启发式拆
            raw_parents = _heuristic_split_parents(text, parent_max_len)

        for pi, parent_text in enumerate(raw_parents):
            if len(parents_out) >= max_preview_blocks:
                break
            if not parent_text.strip():
                continue
            children = _split_parent_into_children(parent_text, child_size, 0, child_sep=child_sep)
            preview_children = children[:max_preview_blocks]
            parents_out.append({
                "index": pi + 1,
                "parent_text": parent_text[:2000] + ("..." if len(parent_text) > 2000 else ""),
                "length": len(parent_text),
                "children": [{"child_index": i + 1, "text": c, "length": len(c)}
                             for i, c in enumerate(preview_children)],
                "total_children": len(children),
            })

    total_parents = len(parents_out)
    total_children = sum(p["total_children"] for p in parents_out)
    parent_avg = round(sum(p["length"] for p in parents_out) / total_parents) if total_parents else 0

    return {
        "mode": "parent_child",
        "stats": {
            "total_parents": total_parents,
            "total_children": total_children,
            "avg_parent_length": parent_avg,
            "avg_child_length": round(
                sum(len(c["text"]) for p in parents_out for c in p["children"]) / max(total_children, 1)
            ) if total_children else 0,
            "total_chars": sum(p["length"] for p in parents_out),
        },
        "blocks": [],
        "parents": parents_out,
        "tail_omitted": 0,
    }


def _heuristic_split_parents(text, target_len=1024):
    """没填 parent_separator 时的启发式父块切分：按 \n\n 和 heading 边界"""
    if len(text) <= target_len:
        return [text]

    # 先按 \n\n 拆段落
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    if not paragraphs:
        return [text]

    parents = []
    cur = []
    cur_len = 0
    for para in paragraphs:
        # 遇到 heading 且当前块已有内容 → 切新父块
        if cur and _looks_like_heading(para.split('\n')[0].strip()) and cur_len >= target_len * 0.5:
            parents.append("\n\n".join(cur))
            cur = [para]
            cur_len = len(para)
        elif cur_len + len(para) > target_len and cur:
            parents.append("\n\n".join(cur))
            cur = [para]
            cur_len = len(para)
        else:
            cur.append(para)
            cur_len += len(para)

    if cur:
        parents.append("\n\n".join(cur))
    return parents


# =====================================================================
# 工具函数
# =====================================================================

def _looks_like_heading(text):
    """启发式判断一行文本是否是标题"""
    if not text:
        return False
    # 数字编号 + 标题词："3. 退款政策" / "一、产品简介" / "1.1 背景"
    if re.match(r"^[\d一二三四五六七八九十]+[\.\、．\-\s]", text[:12]):
        return True
    # 全大写或短句（< 20 字）且末尾不是句号/问号/感叹号
    if len(text) < 20 and text[-1] not in "。？！：；.,!?;:":
        # 排除显然是正文的短句（纯数字、纯标点）
        if re.search(r"[\u4e00-\u9fa5A-Za-z]", text):
            return True
    return False


def _save_temp_image(img_bytes, ext):
    """图片字节落盘到临时目录，返回路径"""
    if not ext or "." not in ext:
        ext = "png"
    if not ext.startswith("."):
        ext = "." + ext
    fname = f"img_{uuid.uuid4().hex}{ext}"
    # 优先存到 uploads 里的 temp 子目录（避免散落系统 temp）
    save_dir = os.path.join(config.UPLOAD_DIR, "_img_tmp")
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, fname)
    with open(path, "wb") as f:
        f.write(img_bytes)
    return path
