#!/bin/bash
# Canvas RAG Chatbot — 데이터 이전 스크립트
#
# [Mac → WSL2 서버] 사용법:
#   Mac에서:    bash scripts/migrate_data.sh export
#   서버에서:   bash scripts/migrate_data.sh import

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPORT_FILE="/tmp/canvas_rag_data.tar.gz"

case "${1:-help}" in

  # ── Mac에서 실행 ─────────────────────────────────────────────────────────
  export)
    echo "데이터 묶음 생성 중..."
    cd "$REPO_DIR"
    tar -czf "$EXPORT_FILE" \
      data/manifests/ \
      data/chunks/ \
      data/markdown/ \
      --exclude='data/raw_html'  # HTML 원본 제외 (용량 절약)
    SIZE=$(du -sh "$EXPORT_FILE" | cut -f1)
    echo "  → ${EXPORT_FILE} (${SIZE}) 생성 완료"
    echo ""
    echo "서버로 전송:"
    echo "  scp ${EXPORT_FILE} user@서버IP:/tmp/"
    echo ""
    echo "또는 WSL2 로컬 서버라면:"
    echo "  cp ${EXPORT_FILE} /mnt/c/Users/사용자명/Downloads/"
    ;;

  # ── 서버(WSL2)에서 실행 ──────────────────────────────────────────────────
  import)
    echo "데이터 압축 해제 중..."

    if [ ! -f "$EXPORT_FILE" ]; then
      echo "ERROR: ${EXPORT_FILE} 파일이 없습니다."
      echo "  Windows에서 복사: cp /mnt/c/Users/사용자명/Downloads/canvas_rag_data.tar.gz /tmp/"
      exit 1
    fi

    cd "$REPO_DIR"
    mkdir -p data
    tar -xzf "$EXPORT_FILE" -C .
    echo "  → data/ 디렉토리 복원 완료"

    echo ""
    echo "Qdrant 재색인 시작 (CUDA 사용)..."
    source config/server.config
    uv run python -m src.indexing.qdrant_index \
      --chunks data/chunks/all_guides_chunks.jsonl \
      --collection canvas_guides \
      --qdrant-url "${QDRANT_URL:-http://localhost:6333}" \
      --embedder auto

    # developer docs 청크가 있으면 함께 색인
    if [ -f data/chunks/developer_docs_chunks.jsonl ]; then
      uv run python -m src.indexing.qdrant_index \
        --chunks data/chunks/developer_docs_chunks.jsonl \
        --collection canvas_guides \
        --qdrant-url "${QDRANT_URL:-http://localhost:6333}" \
        --embedder auto
    fi

    echo ""
    echo "데이터 이전 완료."
    ;;

  *)
    echo "사용법:"
    echo "  Mac에서:  bash scripts/migrate_data.sh export"
    echo "  서버에서: bash scripts/migrate_data.sh import"
    ;;
esac
