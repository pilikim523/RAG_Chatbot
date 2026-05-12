#!/usr/bin/env bash
# Zoom Developer Docs 전체 파이프라인: discover → fetch → markdown → chunk → index
# 사용법: bash scripts/pipeline_zoom.sh [--force]
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
LOG="$LOG_DIR/pipeline_zoom_$(date +%Y%m%d_%H%M%S).log"
FORCE="${1:-}"

echo "[Zoom Pipeline] 시작: $(date)" | tee -a "$LOG"

# ── 1. URL 수집 (sitemap 기반) ───────────────────────────────────────────────
echo "[1/5] URL 수집 (discover_zoom, sitemap 기반)" | tee -a "$LOG"
if [ "$FORCE" = "--force" ]; then
  rm -f data/manifests/zoom_urls.jsonl
fi
$PYTHON -m src.crawler.discover_zoom \
  --product zoom \
  --url-prefix /docs/ \
  --out data/manifests/zoom_urls.jsonl 2>&1 | tee -a "$LOG"

# ── 2. HTML 페치 ─────────────────────────────────────────────────────────────
echo "[2/5] HTML 페치" | tee -a "$LOG"
$PYTHON -m src.crawler.fetch \
  --manifest data/manifests/zoom_urls.jsonl \
  --out-dir data/raw_html/zoom \
  --concurrency 2 \
  --delay-ms 1000 2>&1 | tee -a "$LOG"

# ── 3. Markdown 변환 (Next.js MDX 추출) ─────────────────────────────────────
echo "[3/5] Markdown 변환" | tee -a "$LOG"
if [ "$FORCE" = "--force" ]; then
  rm -f data/manifests/zoom_docs.jsonl
fi
$PYTHON -m src.cleaner.to_markdown \
  --input-manifest data/manifests/zoom_urls.jsonl \
  --raw-html-dir data/raw_html/zoom \
  --out-dir data/markdown/zoom \
  --out-manifest data/manifests/zoom_docs.jsonl 2>&1 | tee -a "$LOG"

# ── 4. 청킹 ─────────────────────────────────────────────────────────────────
echo "[4/5] 청킹" | tee -a "$LOG"
$PYTHON -m src.indexing.chunk \
  --manifest data/manifests/zoom_docs.jsonl \
  --markdown-dir data/markdown/zoom \
  --out data/chunks/zoom_chunks.jsonl \
  --chunk-size 900 --chunk-overlap 120 2>&1 | tee -a "$LOG"

# ── 5. Qdrant 적재 ──────────────────────────────────────────────────────────
echo "[5/5] Qdrant 적재 (zoom_docs)" | tee -a "$LOG"
$PYTHON -m src.indexing.qdrant_index \
  --chunks data/chunks/zoom_chunks.jsonl \
  --collection zoom_docs \
  --qdrant-url "${QDRANT_URL:-http://localhost:6333}" 2>&1 | tee -a "$LOG"

echo "[Zoom Pipeline] 완료: $(date)" | tee -a "$LOG"
echo "로그: $LOG"
