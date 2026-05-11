"""
Panopto SOAP API 문서 크롤러 + 청크 변환기.

대상: https://kr.support.panopto.com/resource/APIDocumentation/Help/html/
구조: 정적 HTML (.htm, UUID 파일명), 링크는 ../html/{uuid}.htm 패턴
언어: 한국어 (Korean-localized SOAP API reference)

출력: data/chunks/panopto_soap_chunks.jsonl
적재: uv run python -m src.indexing.qdrant_index --chunks ... --collection canvas_guides
"""
from __future__ import annotations

import re
import time
from collections import deque
from pathlib import Path

import click
import httpx
from rich.console import Console

from src.crawler.models import sha256_of
from src.indexing.chunk import chunk_markdown
from src.indexing.models import Chunk, make_chunk_id, save_chunks

console = Console(stderr=True)

_START_URL = "https://kr.support.panopto.com/resource/APIDocumentation/Help/html/c40c21c6-dc7a-d822-8c99-b276ee8337c2.htm"
_BASE_URL   = "https://kr.support.panopto.com/resource/APIDocumentation/Help/html/"
_HEADERS    = {"User-Agent": "Mozilla/5.0 (compatible; PanoptoRAGBot/1.0)"}


# ---------------------------------------------------------------------------
# HTML → Markdown 변환
# ---------------------------------------------------------------------------

def _extract_text(html: str) -> tuple[str, str]:
    """HTML에서 (title, body_text)를 추출한다."""
    title_m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.I)
    title = title_m.group(1).strip() if title_m else "Panopto SOAP API"

    # script / style / nav 제거
    html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.S | re.I)
    html = re.sub(r"<style[^>]*>.*?</style>",  " ", html, flags=re.S | re.I)
    # JSON 메타 블록 제거 (allLanguageUrls 등)
    html = re.sub(r'\{[^{}]{0,500}"allLanguageUrls"[^{}]{0,500}\}', " ", html, flags=re.S)

    # 태그 → 공백
    text = re.sub(r"<[^>]+>", " ", html)
    # 연속 공백 정리
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title, text.strip()


def _html_to_markdown(title: str, body: str, source_url: str) -> str:
    """추출된 텍스트를 RAG용 Markdown으로 포맷한다."""
    lines = [
        f"# {title}",
        "",
        f"**출처:** {source_url}",
        f"**API 유형:** Panopto SOAP API (Korean)",
        "",
        "---",
        "",
        body,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# BFS 크롤러
# ---------------------------------------------------------------------------

def _discover_links(html: str) -> list[str]:
    """페이지에서 같은 html/ 디렉터리 내 링크만 추출한다."""
    hrefs = re.findall(r'href=["\']\.\.\/html\/([a-f0-9\-]+\.htm[l]?)["\']', html, re.I)
    return [_BASE_URL + h for h in hrefs]


def crawl_soap_docs(
    max_pages: int,
    delay_ms: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    visited: set[str] = set()
    queue: deque[str] = deque([_START_URL])
    all_chunks: list[Chunk] = []
    errors = 0

    console.print(f"[blue]Panopto SOAP API 문서 크롤링 시작 (최대 {max_pages}페이지)...[/blue]")

    with httpx.Client(headers=_HEADERS, timeout=20, follow_redirects=True) as client:
        while queue and len(visited) < max_pages:
            url = queue.popleft()
            if url in visited:
                continue
            visited.add(url)

            try:
                resp = client.get(url)
                resp.raise_for_status()
                html = resp.text
            except Exception as e:
                console.print(f"  [yellow]WARN {url}: {e}[/yellow]")
                errors += 1
                continue

            # 새 링크 발견
            for link in _discover_links(html):
                if link not in visited:
                    queue.append(link)

            title, body = _extract_text(html)
            if len(body) < 100:
                continue  # 내용 없는 페이지 스킵

            markdown = _html_to_markdown(title, body, url)
            raw_chunks = chunk_markdown(markdown, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            total = len(raw_chunks)

            for idx, text in enumerate(raw_chunks):
                all_chunks.append(Chunk(
                    chunk_id=make_chunk_id(url, idx),
                    source_url=url,
                    canonical_url=url,
                    title=title,
                    product="panopto",
                    guide="soap-api-reference",
                    category="soap-api",
                    role=None,
                    chunk_index=idx,
                    chunk_total=total,
                    text=text,
                    char_count=len(text),
                    word_count=len(text.split()),
                    content_hash=sha256_of(text),
                ))

            console.print(
                f"  [dim][{len(visited)}/{max_pages}] {title[:60]} → {total} chunk(s)[/dim]"
            )
            time.sleep(delay_ms / 1000)

    console.print(
        f"[green]크롤링 완료: {len(visited)}페이지 방문, {errors}개 오류, "
        f"{len(all_chunks)} chunks 생성[/green]"
    )
    return all_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--out", default="data/chunks/panopto_soap_chunks.jsonl", show_default=True,
              type=click.Path(), help="출력 JSONL 경로")
@click.option("--max-pages", default=300, show_default=True, help="최대 크롤링 페이지 수")
@click.option("--delay-ms", default=200, show_default=True, help="요청 간 대기 (ms)")
@click.option("--chunk-size", default=900, show_default=True)
@click.option("--chunk-overlap", default=120, show_default=True)
def main(
    out: str,
    max_pages: int,
    delay_ms: int,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    """Panopto SOAP API 문서(kr.support.panopto.com) 크롤링 → JSONL 생성."""
    chunks = crawl_soap_docs(max_pages, delay_ms, chunk_size, chunk_overlap)

    if not chunks:
        console.print("[red]청크가 없습니다. 크롤링을 확인하세요.[/red]")
        return

    save_chunks(chunks, Path(out))
    console.print(f"\n[bold]총 {len(chunks)} chunks → {out}[/bold]")
    console.print("[green]다음 명령으로 Qdrant에 적재:[/green]")
    console.print(
        f"  uv run python -m src.indexing.qdrant_index "
        f"--chunks {out} --collection canvas_guides"
    )


if __name__ == "__main__":
    main()
