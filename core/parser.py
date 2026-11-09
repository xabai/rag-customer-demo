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
# 元素数据结构
# =====================================================================

class LayoutElement:
    """统一的版面元素"""
    __slots__ = ("id", "type", "content", "bbox", "parent_id", "doc_name",
                 "anchor_to", "image_path", "image_b64")

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


# =====================================================================
# 主入口
# =====================================================================

def parse_file(path):
    """兼容旧接口：返回纯文本（把 layout elements 展平）"""
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
    return "\n\n".join(parts)


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
    """解析 PDF：优先电子版（pdfplumber 抽 words+bbox），失败则 RapidOCR 兜底
    抽 words 用 extract_words() 拿每个词的 bbox，extract_text() 拿纯文本，两者结合"""
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
                    # 行 y 归一化：取 y0 四舍五入到整数
                    y_key = round(w["top"], 0)
                    if y_key not in lines_by_y:
                        lines_by_y[y_key] = {"text": "", "bbox": [99999, 99999, 0, 0]}
                    ln = lines_by_y[y_key]
                    ln["text"] += (" " if ln["text"] else "") + w["text"]
                    ln["bbox"][0] = min(ln["bbox"][0], w["x0"])
                    ln["bbox"][1] = min(ln["bbox"][1], w["top"])
                    ln["bbox"][2] = max(ln["bbox"][2], w["x1"])
                    ln["bbox"][3] = max(ln["bbox"][3], w["bottom"])

                # 标题探测：行内全部大写 或 字号显著比同行大（pdfplumber 拿不到字号，用启发式）
                # 简单启发：行首是数字编号 + 黑体标记 → heading
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
                # 电子版但没抽 words（行块太复杂）→ 整页当纯文本
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
                # 整页没文本 → 扫描件
                scan_fallback = True
                break

            # 2. 表格：extract_tables 拿结构化表格
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

            # 3. 图片：page.images 拿 bbox 和 raw_stream
            for img_info in page.images:
                try:
                    img_bytes = img_info.get("stream", b"") or b""
                    if not img_bytes:
                        continue
                    # pdfplumber image 的 bbox 字段名不太统一，兜底取默认值
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
            # 图文配对
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


def build_layout_chunks(elements):
    """按 heading 边界切分版面感知 chunks

    规则:
    - 遇到新 heading → 如果当前块已有内容 → 结束当前块，heading 成为新块标题
    - 单块上限 config.LAYOUT_MAX_WORDS 字 / config.LAYOUT_MAX_IMAGES 张图，超过强制切
    - 同 parent_id 内的元素优先聚合
    """
    if not elements:
        return []

    # 按 parent_id + bbox.top 全局排序
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

        # 触发新块的条件
        force_new = False

        # 1) 遇到 heading 且当前块已有正文内容
        if el.type == "heading" and current.elements and current.word_count > 20:
            force_new = True

        # 2) 跨 parent_id 且当前块非空（PDF 跨页 / PPT 跨 slide）
        if current.parent_id != el.parent_id and current.elements:
            force_new = True

        # 3) 超限强制切
        if (current.word_count >= config.LAYOUT_MAX_WORDS or
                current.image_count >= config.LAYOUT_MAX_IMAGES):
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
    return [c for c in chunks if c.word_count > 0 or c.image_count > 0]


def layout_chunks_to_records(chunks):
    """把 LayoutChunk 列表转换成 chunks.json 的记录格式 + 图片路径列表

    返回:
        chunk_texts: list[str]        # 文本块，给 bge-m3 编
        parent_texts: list[str|None]  # 父块文本（这里就等于自身文本，后续可选父子分块时再拆）
        chunk_images: list[str|None]  # 每个块的图片路径（多图用分号拼接，存 kb_images 目录）
        layout_elements: list[list]   # 每个块的 elements 原始数据（给前端展示图文关联）
    """
    chunk_texts = []
    parent_texts = []
    chunk_images = []
    layout_elements = []

    for ch in chunks:
        text = ch.to_text()
        chunk_texts.append(text)
        parent_texts.append(text)  # 暂时父块 = 自身，后续父子分块再扩展

        # 先落盘图片 → 建立"旧临时路径 → 新持久路径"映射
        src_to_dst = {}
        if ch.image_paths:
            persisted = []
            for src in ch.image_paths:
                if not src or not os.path.exists(src):
                    continue
                dst = os.path.join(config.KB_IMG_DIR, f"{uuid.uuid4().hex}{os.path.splitext(src)[1]}")
                try:
                    shutil.copy2(src, dst)
                    persisted.append(dst)
                    src_to_dst[src] = dst
                except Exception:
                    pass
            chunk_images.append(";".join(persisted) if persisted else None)
        else:
            chunk_images.append(None)

        # 保存 elements 元数据（图片用持久后的路径）
        el_data = []
        for el in ch.elements:
            rec = {
                "type": el.type,
                "content": el.content,
                "parent_id": el.parent_id,
                "anchor_to": el.anchor_to,
            }
            if el.type == "image":
                new_path = src_to_dst.get(el.image_path, el.image_path)
                rec["image_path"] = new_path  # 持久化后的路径（API 层再转成 URL）
            el_data.append(rec)

        # 找 anchor 对应的 text content 做 paired_text（前端展示图文关联）
        for rec in el_data:
            if rec.get("type") == "image" and rec.get("anchor_to"):
                for orig_el in ch.elements:
                    if orig_el.id == rec["anchor_to"] and orig_el.content:
                        rec["paired_text"] = orig_el.content
                        break
        layout_elements.append(el_data)

    return chunk_texts, parent_texts, chunk_images, layout_elements


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
