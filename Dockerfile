# ====================================================================
# RAG-BOT 后端镜像（FastAPI + Uvicorn）
# 基于 python:3.12-slim，生产依赖 only，多阶段构建
# ====================================================================

# ---------- 阶段 1：安装依赖 ----------
FROM python:3.12-slim AS builder

# 系统依赖：psycopg 编译 + pg_isready（等 postgres 用）
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        gcc libpq-dev curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先拷贝 requirements.txt 单独 pip install（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
 && pip install --no-cache-dir --prefix=/install uvicorn[standard] gunicorn

# ---------- 阶段 2：运行时镜像 ----------
FROM python:3.12-slim AS runtime

# 运行时系统依赖：libpq（psycopg 运行要）+ curl（healthcheck）+ tzdata
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libpq5 curl tzdata ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
 && echo "Asia/Shanghai" > /etc/timezone

# 从 builder 拿 Python 包
COPY --from=builder /install /usr/local

# 非 root 用户（安全）
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /sbin/nologin appuser

WORKDIR /app

# 拷贝应用代码（.dockerignore 已经排除了 __pycache__、digital_human/pylibs/**、data/ 等）
COPY *.py ./           # app.py, db.py, config.py 等根目录 Python
COPY core/ ./core/
COPY static/ ./static/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY entrypoint.sh ./

# 先建目录，chown 一次搞定
RUN mkdir -p /app/data/logs /app/data/uploads /app/data/chroma /app/data/kb_images \
 && chmod +x entrypoint.sh \
 && chown -R appuser:appuser /app

USER appuser

# 数据持久化（这些目录挂 volume 不会丢数据）
VOLUME ["/app/data"]

EXPOSE 8008

# 默认：后端直接 serve static + API（最简模式，Nginx 可选）
# 用 2 worker 基本够单实例
CMD ["./entrypoint.sh"]
