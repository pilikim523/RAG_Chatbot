#!/bin/bash
# Canvas RAG Chatbot — 서비스 시작 스크립트 (WSL2 Ubuntu)
# 사용법: bash scripts/start_service.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_FILE="${REPO_DIR}/config/server.config"
LOG_DIR="${REPO_DIR}/logs"

mkdir -p "$LOG_DIR"

# ── 설정 파일 확인 ──────────────────────────────────────────────────────────
if [ ! -f "$CONFIG_FILE" ]; then
  echo "ERROR: config/server.config 파일이 없습니다."
  echo "  cp config/server.config.example config/server.config 후 값을 설정하세요."
  exit 1
fi
set -a; source "$CONFIG_FILE"; set +a

# ── GPU 확인 ────────────────────────────────────────────────────────────────
echo "[1/4] GPU 상태 확인..."
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
  echo "  WARNING: nvidia-smi 없음. CPU 모드로 실행됩니다."
fi

# ── Qdrant 시작 ────────────────────────────────────────────────────────────
echo ""
echo "[2/4] Qdrant 시작..."
cd "$REPO_DIR"
docker compose -f docker-compose.server.yml up -d qdrant
echo "  → Qdrant: http://localhost:6333"

# ── Ollama 시작 ────────────────────────────────────────────────────────────
echo ""
echo "[3/4] Ollama 시작..."
if systemctl is-active --quiet ollama 2>/dev/null; then
  echo "  → Ollama 이미 실행 중"
else
  sudo systemctl start ollama
  sleep 3
fi
echo "  → Ollama: http://localhost:11434"

# ── FastAPI 시작 ───────────────────────────────────────────────────────────
echo ""
echo "[4/4] FastAPI 서버 시작..."
cd "$REPO_DIR"

# 기존 프로세스 종료
pkill -f "uvicorn src.api.main" 2>/dev/null || true
sleep 1

# 환경변수 로드 후 서버 시작
nohup env $(cat "$CONFIG_FILE" | grep -v '^#' | grep -v '^$' | xargs) \
  uv run uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port 8080 \
    --workers 1 \
  >> "$LOG_DIR/api.log" 2>&1 &

echo $! > "$LOG_DIR/api.pid"
sleep 4

if kill -0 "$(cat "$LOG_DIR/api.pid")" 2>/dev/null; then
  echo "  → FastAPI: http://0.0.0.0:8080 (PID: $(cat "$LOG_DIR/api.pid"))"
  echo "  → 로그: $LOG_DIR/api.log"
else
  echo "  ERROR: FastAPI 시작 실패. 로그 확인:"
  tail -20 "$LOG_DIR/api.log"
  exit 1
fi

echo ""
echo "======================================"
echo " 모든 서비스 시작 완료"
echo "======================================"
echo "  Qdrant:  http://localhost:6333"
echo "  Ollama:  http://localhost:11434"
echo "  API:     http://0.0.0.0:8080"
echo "  UI:      http://0.0.0.0:8080/"
echo ""
echo "서비스 상태 확인: bash scripts/status_service.sh"
echo "서비스 중지:      bash scripts/stop_service.sh"
