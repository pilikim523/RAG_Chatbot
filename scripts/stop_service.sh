#!/bin/bash
# Canvas RAG Chatbot — 서비스 중지 스크립트

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"

echo "[1/3] FastAPI 중지..."
pkill -f "uvicorn src.api.main" 2>/dev/null && echo "  → 중지 완료" || echo "  → 실행 중 아님"
rm -f "$LOG_DIR/api.pid"

echo "[2/3] Qdrant 중지..."
cd "$REPO_DIR"
docker compose -f docker-compose.server.yml stop qdrant && echo "  → 중지 완료"

echo "[3/3] Ollama는 systemd 서비스로 유지됩니다."
echo "  Ollama 중지 원할 때: sudo systemctl stop ollama"

echo ""
echo "서비스 중지 완료."
