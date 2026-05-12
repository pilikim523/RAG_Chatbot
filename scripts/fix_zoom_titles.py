"""
Zoom docs Qdrant payload의 title=null 항목을 재색인 없이 업데이트.

사용법:
  python scripts/fix_zoom_titles.py \
    --raw-html-dir data/raw_html/zoom \
    --manifest data/manifests/zoom_docs.jsonl \
    --qdrant-url http://localhost:6333 \
    --collection zoom_docs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, IsNullCondition, MatchValue, PayloadField

from src.cleaner.to_markdown import extract_page_title


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-html-dir", default="data/raw_html/zoom")
    parser.add_argument("--manifest", default="data/manifests/zoom_docs.jsonl")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="zoom_docs")
    args = parser.parse_args()

    raw_dir = Path(args.raw_html_dir)
    manifest_path = Path(args.manifest)

    # manifest에서 source_url → raw_html_path 매핑 로드
    entries: list[dict] = []
    with manifest_path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    client = QdrantClient(url=args.qdrant_url)

    updated = skipped = failed = 0

    for entry in entries:
        source_url = entry.get("source_url")
        raw_html_path = entry.get("raw_html_path")
        if not source_url or not raw_html_path:
            skipped += 1
            continue

        html_path = raw_dir / raw_html_path
        if not html_path.exists():
            skipped += 1
            continue

        try:
            html = html_path.read_text(encoding="utf-8")
            title = extract_page_title(html)
            if not title:
                skipped += 1
                continue

            # source_url로 해당 포인트 검색 후 payload 업데이트
            result = client.scroll(
                collection_name=args.collection,
                scroll_filter=Filter(
                    must=[FieldCondition(key="source_url", match=MatchValue(value=source_url))]
                ),
                limit=20,
                with_payload=False,
                with_vectors=False,
            )
            points = result[0]
            if not points:
                skipped += 1
                continue

            point_ids = [p.id for p in points]
            client.set_payload(
                collection_name=args.collection,
                payload={"title": title},
                points=point_ids,
            )
            updated += len(point_ids)
            print(f"[OK] {title[:50]!r:52s} ({len(point_ids)} pts) {source_url}")

        except Exception as e:
            print(f"[ERR] {source_url}: {e}", file=sys.stderr)
            failed += 1

    print(f"\n완료: updated={updated} points, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
