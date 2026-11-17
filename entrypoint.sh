#!/bin/bash
# ====================================================================
# RAG-BOT 容器启动入口
# 1) 如果配置了 PostgreSQL，等它 ready（最多 60 秒）
# 2) 启动 uvicorn（--host 0.0.0.0 容器内必须，不然外面连不到）
# ====================================================================
set -euo pipefail

echo "=== RAG-BOT Entrypoint ==="
echo "APP_ENV=${APP_ENV:-dev}"
echo "DB_DRIVER=${DB_DRIVER:-sqlite}"
echo "DB_HOST=${DB_HOST:-localhost}"
echo "API_PORT=${API_PORT:-8008}"

# ---------- 等 PostgreSQL ready ----------
if [ "${DB_DRIVER:-}" = "postgresql" ]; then
    echo "[entrypoint] 等待 PostgreSQL ${DB_HOST}:${DB_PORT:-5432} ..."
    for i in $(seq 1 60); do
        if python -c "
import sys, psycopg
try:
    psycopg.connect(
        host='${DB_HOST}', port=${DB_PORT:-5432},
        user='${DB_USER:-postgres}', password='${DB_PASSWORD:-}',
        dbname='${DB_NAME:-rag_bot}',
        connect_timeout=2,
    )
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            echo "[entrypoint] ✅ PostgreSQL ready"
            break
        fi
        echo "[entrypoint] ⏳ 等待 PostgreSQL... ($i/60)"
        sleep 2
    done
fi

# ---------- 等 Ollama ready（如果用本地 Ollama provider）----------
if [ "${LLM_PROVIDER:-}" = "ollama" ] || [ -z "${LLM_PROVIDER:-}" ]; then
    OLLAMA_HOST="${OLLAMA_HOST:-ollama}"
    OLLAMA_PORT="${OLLAMA_PORT:-11434}"
    echo "[entrypoint] 等待 Ollama ${OLLAMA_HOST}:${OLLAMA_PORT} ..."
    for i in $(seq 1 30); do
        if curl -sf "http://${OLLAMA_HOST}:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
            echo "[entrypoint] ✅ Ollama ready"
            break
        fi
        echo "[entrypoint] ⏳ 等待 Ollama... ($i/30)"
        sleep 2
    done
fi

# ---------- 启动 uvicorn ----------
echo "[entrypoint] 🚀 启动 uvicorn 0.0.0.0:${API_PORT:-8008} ..."
exec uvicorn app:app \
    --host 0.0.0.0 \
    --port "${API_PORT:-8008}" \
    --workers "${UVICORN_WORKERS:-2}" \
    --loop uvloop \
    --no-access-log \
    --log-level "${LOG_LEVEL:-info}"
