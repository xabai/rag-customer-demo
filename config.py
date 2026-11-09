# -*- coding: utf-8 -*-
"""全局配置"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_data_dir():
    """数据目录解析优先级：环境变量 RAG_DATA_DIR > 指针文件 data_location.txt > 默认 BASE_DIR/data。
    迁移数据目录后由后端写入指针文件，重启服务生效。"""
    env = os.environ.get("RAG_DATA_DIR", "").strip()
    if env:
        return env
    pointer = os.path.join(BASE_DIR, "data_location.txt")
    if os.path.exists(pointer):
        try:
            with open(pointer, "r", encoding="utf-8") as f:
                p = f.read().strip()
            if p:
                return p
        except Exception:
            pass
    return os.path.join(BASE_DIR, "data")


DATA_DIR = _resolve_data_dir()
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
VECTOR_PATH = os.path.join(DATA_DIR, "vectors.npy")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.json")
DB_PATH = os.path.join(DATA_DIR, "chat.db")

# Ollama
OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "bge-m3:latest"                       # 纯文本向量，常驻显存
VL_EMBED_MODEL = "batiai/qwen3-vl-embed-2b:q6"      # 图文统一向量，常驻显存
VL_EMBED_DIM = 2048                                 # qwen3-vl-embed 输出维度
CHAT_MODEL = "qwen2.5:3b-instruct"                  # 冷启（用完释放）
VISION_MODEL = "qwen2.5vl:3b"                       # 冷启（用完释放，仅多模态问答/图片入库时调用）
PROBE_MODEL = "qwen2.5:3b-instruct"                 # 冷启

# 显存调度：keep_alive 秒，-1=常驻不释放，0=用完立即释放
KEEP_ALIVE_EMBED = -1      # 文本向量模型常驻（高频调用）
KEEP_ALIVE_VL_EMBED = -1   # 图文向量模型常驻（高频调用）
KEEP_ALIVE_CHAT = 0        # 聊天模型冷启（仅用户发消息时需要）
KEEP_ALIVE_VISION = 0      # 视觉模型冷启（仅多模态问答/图片入库时需要）
KEEP_ALIVE_PROBE = 0       # 工具探测模型冷启

# RAG 参数
CHUNK_SIZE = 500        # 每个切块的目标字符数
CHUNK_OVERLAP = 50      # 切块重叠字符数
TOP_K = 5               # 检索返回的块数
SCORE_THRESHOLD = 0.35  # 余弦相似度阈值，低于则不引用

# 版面感知分块参数
LAYOUT_MAX_WORDS = 800      # 单块字数上限（版面感知分块用，替换 500 字固定切分）
LAYOUT_MAX_IMAGES = 3       # 单块图片数量上限

# 支持的文件类型
ALLOWED_EXT = {".txt", ".md", ".markdown", ".docx", ".doc", ".xlsx", ".xls", ".pdf", ".pptx", ".ppt"}
# 图片知识库支持的图片类型
ALLOWED_IMG_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

KB_IMG_DIR = os.path.join(DATA_DIR, "kb_images")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(KB_IMG_DIR, exist_ok=True)
