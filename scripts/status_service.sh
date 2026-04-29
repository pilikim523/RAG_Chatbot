#!/bin/bash
# Canvas RAG Chatbot — 서비스 상태 확인

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"

echo "======================================"
echo " 서비스 상태"
echo "======================================"

# GPU
echo ""
echo "[GPU]"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits 2>/dev/null \
  | awk -F',' '{printf "  %-30s | VRAM: %s/%s MB used | GPU: %s%%\n", $1, $2, $2+$3, $4}' \
  || echo "  nvidia-smi 없음"

# Qdrant
echo ""
echo "[Qdrant]"
if curl -sf http://localhost:6333/healthz > /dev/null 2>&1; then
  POINTS=$(curl -sf http://localhost:6333/collections/canvas_guides 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result']['points_count'])" 2>/dev/null || echo "?")
  echo "  ✅ 실행 중 | canvas_guides 벡터: ${POINTS}개"
else
  echo "  ❌ 응답 없음"
fi

# Ollama
echo ""
echo "[Ollama]"
if curl -sf http://localhost:11434/api/tags > /dev/null 2>&1; then
  MODELS=$(curl -sf http://localhost:11434/api/tags | python3 -c \
    "import sys,json; models=json.load(sys.stdin)['models']; [print('  -', m['name']) for m in models]" 2>/dev/null)
  echo "  ✅ 실행 중"
  echo "$MODELS"
else
  echo "  ❌ 응답 없음"
fi

# FastAPI
echo ""
echo "[FastAPI]"
HEALTH=$(curl -sf http://localhost:8080/health 2>/dev/null || echo "")
if [ -n "$HEALTH" ]; then
  echo "  ✅ 실행 중"
  echo "  $HEALTH"
else
  echo "  ❌ 응답 없음"
fi

echo ""
