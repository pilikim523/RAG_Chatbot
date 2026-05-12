#!/usr/bin/env bash
# Canvas LMS 전체 파이프라인: discover → fetch → markdown → chunk → index
# 사용법: bash scripts/pipeline_canvas.sh [--force]
#   --force: 기존 manifest를 무시하고 전체 재처리
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

CONFIG_FILE="config/server.config"
if [ -f "$CONFIG_FILE" ]; then
  set -a; source "$CONFIG_FILE"; set +a
elif [ -f ".env" ]; then
  set -a; source ".env"; set +a
fi

PYTHON="${PYTHON:-uv run python}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pipeline_canvas_$(date +%Y%m%d_%H%M%S).log"
FORCE="${1:-}"

echo "[Canvas Pipeline] 시작: $(date)" | tee -a "$LOG"

# ── 1. URL 수집 ─────────────────────────────────────────────────────────────
echo "[1/5] URL 수집 (discover_canvas)" | tee -a "$LOG"
if [ "$FORCE" = "--force" ]; then
  rm -f data/manifests/canvas_urls.jsonl
fi
$PYTHON -m src.crawler.discover_canvas \
  --start-url "https://community.instructure.com/en/all-guides" \
  --product canvas \
  --out data/manifests/canvas_urls.jsonl 2>&1 | tee -a "$LOG"

# ── 2. HTML 페치 ─────────────────────────────────────────────────────────────
echo "[2/5] HTML 페치" | tee -a "$LOG"
$PYTHON -m src.crawler.fetch \
  --manifest data/manifests/canvas_urls.jsonl \
  --out-dir data/raw_html/canvas \
  --concurrency 4 \
  --delay-ms 800 2>&1 | tee -a "$LOG"

# ── 3. Markdown 변환 ─────────────────────────────────────────────────────────
echo "[3/5] Markdown 변환" | tee -a "$LOG"
if [ "$FORCE" = "--force" ]; then
  rm -f data/manifests/canvas_docs.jsonl
fi
$PYTHON -m src.cleaner.to_markdown \
  --input-manifest data/manifests/canvas_urls.jsonl \
  --raw-html-dir data/raw_html/canvas \
  --out-dir data/markdown/canvas \
  --out-manifest data/manifests/canvas_docs.jsonl 2>&1 | tee -a "$LOG"

# ── 4. 청킹 ─────────────────────────────────────────────────────────────────
echo "[4/5] 청킹" | tee -a "$LOG"
$PYTHON -m src.indexing.chunk \
  --manifest data/manifests/canvas_docs.jsonl \
  --markdown-dir data/markdown/canvas \
  --out data/chunks/canvas_chunks.jsonl \
  --chunk-size 900 --chunk-overlap 120 2>&1 | tee -a "$LOG"

# ── 5. Qdrant 적재 ──────────────────────────────────────────────────────────
echo "[5/5] Qdrant 적재 (canvas_guides)" | tee -a "$LOG"
$PYTHON -m src.indexing.qdrant_index \
  --chunks data/chunks/canvas_chunks.jsonl \
  --collection canvas_guides \
  --qdrant-url "${QDRANT_URL:-http://localhost:6333}" 2>&1 | tee -a "$LOG"

echo "[Canvas Pipeline] 완료: $(date)" | tee -a "$LOG"
echo "로그: $LOG"
