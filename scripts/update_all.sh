#!/usr/bin/env bash
# 전체 RAG 콘텐츠 주기적 업데이트 스크립트
# cron 등록 예시 (서버 /etc/cron.d/rag-update):
#   # Canvas: 매주 일요일 02:00
#   0 2 * * 0 linusdev bash /mnt/e/Linus-Dev/RAG_Chatbot/scripts/update_all.sh canvas
#   # Zoom: 격주 일요일 03:00 (홀수 주)
#   0 3 * * 0 [ $(date +\%W) -eq $(( $(date +\%W) / 2 * 2 )) ] && bash /mnt/e/Linus-Dev/RAG_Chatbot/scripts/update_all.sh zoom
#   # Panopto: 매월 1일 04:00
#   0 4 1 * * linusdev bash /mnt/e/Linus-Dev/RAG_Chatbot/scripts/update_all.sh panopto
#
# 사용법:
#   bash scripts/update_all.sh          # 전체 (canvas + zoom + panopto)
#   bash scripts/update_all.sh canvas   # Canvas만
#   bash scripts/update_all.sh zoom     # Zoom만
#   bash scripts/update_all.sh panopto  # Panopto만
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

TARGET="${1:-all}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
SUMMARY_LOG="$LOG_DIR/update_all_$(date +%Y%m%d_%H%M%S).log"

run_pipeline() {
  local product="$1"
  local script="scripts/pipeline_${product}.sh"
  if [ ! -f "$script" ]; then
    echo "[SKIP] $product: $script 없음" | tee -a "$SUMMARY_LOG"
    return
  fi
  echo "" | tee -a "$SUMMARY_LOG"
  echo "======================================" | tee -a "$SUMMARY_LOG"
  echo "[UPDATE] $product 시작: $(date)" | tee -a "$SUMMARY_LOG"
  echo "======================================" | tee -a "$SUMMARY_LOG"
  if bash "$script" 2>&1 | tee -a "$SUMMARY_LOG"; then
    echo "[OK] $product 완료: $(date)" | tee -a "$SUMMARY_LOG"
  else
    echo "[FAIL] $product 실패: $(date)" | tee -a "$SUMMARY_LOG"
    # 실패해도 다음 제품 계속 진행
  fi
}

echo "=== RAG 콘텐츠 업데이트 시작: $(date) ===" | tee -a "$SUMMARY_LOG"
echo "대상: $TARGET" | tee -a "$SUMMARY_LOG"

case "$TARGET" in
  canvas)  run_pipeline canvas ;;
  zoom)    run_pipeline zoom ;;
  panopto) run_pipeline panopto ;;
  all)
    run_pipeline canvas
    run_pipeline zoom
    run_pipeline panopto
    ;;
  *)
    echo "사용법: $0 [canvas|zoom|panopto|all]" >&2
    exit 1
    ;;
esac

echo "" | tee -a "$SUMMARY_LOG"
echo "=== 전체 업데이트 완료: $(date) ===" | tee -a "$SUMMARY_LOG"
echo "요약 로그: $SUMMARY_LOG"
