"""
Panopto 문서 수집 → Chunk JSONL 변환기.

두 가지 소스를 처리한다:
  1. Panopto REST API Swagger JSON  → 태그(그룹)별 Markdown
  2. GitHub Panopto 주요 레포 README → Markdown

출력: data/chunks/panopto_chunks.jsonl  (기존 Chunk 스키마 호환)
이후 qdrant_index.py 로 canvas_guides 컬렉션에 product=panopto 로 적재.

사용법:
    uv run python -m src.indexing.panopto_ingest [--out PATH] [--github-token TOKEN]
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import click
import httpx
from rich.console import Console

from src.crawler.models import sha256_of
from src.indexing.chunk import chunk_markdown
from src.indexing.models import Chunk, make_chunk_id, save_chunks

console = Console(stderr=True)

_SWAGGER_URL = "https://demo.hosted.panopto.com/Panopto/api/v1/panoptoApi.json"
_SWAGGER_BASE = "https://demo.hosted.panopto.com/Panopto/api/docs/index.html"

# RAG에 유용한 레포 목록: (owner, repo, category) 튜플
_GITHUB_REPOS: list[tuple[str, str, str]] = [
    # 공식 Panopto 샘플
    ("Panopto", "panopto-api-python-examples", "github-sample"),
    ("Panopto", "upload-python-sample",         "github-sample"),
    ("Panopto", "SOAP-API-Examples",             "github-sample"),
    ("Panopto", "python-soap",                   "github-sample"),
    ("Panopto", "SchedulingTool",                "github-sample"),
    ("Panopto", "UserManagement",                "github-sample"),
    ("Panopto", "UserManagementAPISample",        "github-sample"),
    ("Panopto", "SessionManagementAPISample",     "github-sample"),
    ("Panopto", "AuthManagementAPISample",        "github-sample"),
    ("Panopto", "UploadAPISamples",              "github-sample"),
    ("Panopto", "WatchFolderService",            "github-sample"),
    ("Panopto", "RecorderButton",                "github-sample"),
    ("Panopto", "panopto-index-connector",       "github-sample"),
    ("Panopto", "ViewingStats",                  "github-sample"),
    # mediaguycouk — 커뮤니티 튜토리얼 & 통합 코드
    ("mediaguycouk", "Collab-Panopto",    "github-community"),
    ("mediaguycouk", "PanoptoHierarchy",  "github-community"),
    ("mediaguycouk", "PanoptoRest",       "github-community"),
    ("mediaguycouk", "WebPanoptoAPI",     "github-community"),
]

# README 없는 코드 샘플 레포: {owner/repo: [(rel_path, lang, desc), ...]}
_GITHUB_CODE_REPOS: dict[str, dict] = {
    "Panopto/SOAP-API-Examples": {
        "desc": "Panopto SOAP API 호출 코드 예시 (Python, PHP, Ruby)",
        "files": [
            ("PythonSoapExample.py",                         "python"),
            ("PhpSoapExample.php",                           "php"),
            ("RubySoapExample.rb",                           "ruby"),
            ("wsdl_classes/SessionManagementWsdlClass.php",  "php"),
        ],
    },
    "Panopto/python-soap": {
        "desc": "Panopto SOAP API Python 클라이언트 (zeep 기반)",
        "files": [
            ("readme.rst",                                    "rst"),
            ("src/panopto_api/AuthenticatedClientFactory.py", "python"),
            ("src/panopto_api/ClientWrapper.py",              "python"),
            ("examples/usage_example.py",                     "python"),
            ("examples/stats_report_example.py",              "python"),
        ],
    },
    "Panopto/RecorderButton": {
        "desc": "Panopto Remote Recorder를 API로 시작/정지하는 C# 예시",
        "files": [
            ("README.txt",                          "text"),
            ("RecorderManageExample/Program.cs",    "csharp"),
        ],
    },
    "Panopto/UserManagement": {
        "desc": "Panopto 사용자·폴더·그룹 관리 C# 예시",
        "files": [
            ("README.txt",                           "text"),
            ("UserManagementExample/Program.cs",     "csharp"),
        ],
    },
}


# ---------------------------------------------------------------------------
# Swagger JSON → Markdown
# ---------------------------------------------------------------------------

def _param_table(params: list[dict]) -> str:
    if not params:
        return ""
    rows = ["| 파라미터 | 위치 | 타입 | 필수 | 설명 |",
            "|---------|------|------|------|------|"]
    for p in params:
        schema = p.get("schema", {})
        typ = schema.get("type", p.get("type", "-"))
        ref = schema.get("$ref", "")
        if ref:
            typ = ref.split("/")[-1]
        required = "✅" if p.get("required") else "-"
        desc = p.get("description", "-").replace("\n", " ").replace("|", "\\|")
        rows.append(f"| `{p.get('name','-')}` | {p.get('in','-')} | {typ} | {required} | {desc} |")
    return "\n".join(rows)


def _response_table(responses: dict) -> str:
    rows = ["| 코드 | 설명 |", "|------|------|"]
    for code, info in responses.items():
        desc = info.get("description", "-").replace("\n", " ").replace("|", "\\|")
        rows.append(f"| {code} | {desc} |")
    return "\n".join(rows)


def _body_schema(request_body: dict, components: dict) -> str:
    content = request_body.get("content", {})
    for mime, val in content.items():
        schema = val.get("schema", {})
        ref = schema.get("$ref", "")
        if ref:
            schema_name = ref.split("/")[-1]
            schema_def = components.get("schemas", {}).get(schema_name, {})
            props = schema_def.get("properties", {})
            if props:
                rows = ["| 필드 | 타입 | 설명 |", "|------|------|------|"]
                for fname, fdef in props.items():
                    ftype = fdef.get("type", fdef.get("$ref", "-").split("/")[-1])
                    fdesc = fdef.get("description", "-").replace("\n", " ").replace("|", "\\|")
                    rows.append(f"| `{fname}` | {ftype} | {fdesc} |")
                return f"**Request Body ({schema_name}):**\n\n" + "\n".join(rows)
    return ""


def swagger_to_markdown_by_tag(spec: dict) -> dict[str, str]:
    """Swagger spec을 태그(그룹)별 Markdown으로 변환한다."""
    components = spec.get("components", {})
    paths = spec.get("paths", {})
    servers = spec.get("servers", [{}])
    base = servers[0].get("url", "") if servers else ""

    tag_docs: dict[str, list[str]] = {}

    for path, methods in paths.items():
        for method, op in methods.items():
            if method.upper() not in ("GET", "POST", "PUT", "PATCH", "DELETE"):
                continue
            tags = op.get("tags", ["General"])
            summary = op.get("summary", "")
            description = op.get("description", "")
            params = op.get("parameters", [])
            request_body = op.get("requestBody", {})
            responses = op.get("responses", {})

            lines = [
                f"## {method.upper()} {path}",
                "",
                f"**Summary:** {summary}" if summary else "",
                f"**Full path:** `{base}{path}`" if base else "",
                "",
            ]
            if description:
                lines += [description, ""]

            param_md = _param_table(params)
            if param_md:
                lines += ["### Parameters", "", param_md, ""]

            body_md = _body_schema(request_body, components)
            if body_md:
                lines += [body_md, ""]

            resp_md = _response_table(responses)
            if resp_md:
                lines += ["### Responses", "", resp_md, ""]

            section = "\n".join(l for l in lines if l is not None)

            for tag in tags:
                tag_docs.setdefault(tag, []).append(section)

    result: dict[str, str] = {}
    for tag, sections in tag_docs.items():
        header = f"# Panopto REST API — {tag}\n\n"
        header += f"이 섹션은 Panopto REST API의 **{tag}** 그룹 엔드포인트를 설명한다.\n\n"
        result[tag] = header + "\n\n---\n\n".join(sections)

    return result


def fetch_swagger_chunks(chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    console.print("[blue]Fetching Panopto Swagger JSON...[/blue]")
    resp = httpx.get(_SWAGGER_URL, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    spec = resp.json()

    tag_docs = swagger_to_markdown_by_tag(spec)
    console.print(f"  API groups: {len(tag_docs)}")

    all_chunks: list[Chunk] = []
    for tag, markdown in tag_docs.items():
        source_url = f"{_SWAGGER_BASE}#{tag}"
        raw_chunks = chunk_markdown(markdown, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        total = len(raw_chunks)
        for idx, text in enumerate(raw_chunks):
            all_chunks.append(Chunk(
                chunk_id=make_chunk_id(source_url, idx),
                source_url=source_url,
                canonical_url=_SWAGGER_URL,
                title=f"Panopto REST API — {tag}",
                product="panopto",
                guide="rest-api",
                category=tag.lower().replace(" ", "-"),
                role=None,
                chunk_index=idx,
                chunk_total=total,
                text=text,
                char_count=len(text),
                word_count=len(text.split()),
                content_hash=sha256_of(text),
            ))
    console.print(f"  [green]Swagger → {len(all_chunks)} chunks[/green]")
    return all_chunks


# ---------------------------------------------------------------------------
# GitHub README → Chunks
# ---------------------------------------------------------------------------

def _github_headers(token: str | None) -> dict:
    h = {"Accept": "application/vnd.github.v3+json", "User-Agent": "panopto-rag-ingest/1.0"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def fetch_github_chunks(
    token: str | None,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    console.print("[blue]Fetching GitHub Panopto repos...[/blue]")
    headers = _github_headers(token)
    all_chunks: list[Chunk] = []

    for owner, repo, category in _GITHUB_REPOS:
        # raw.githubusercontent.com 으로 직접 fetch (API rate limit 미적용)
        fetched = False
        for branch in ("main", "master"):
            for readme_name in ("README.md", "readme.md", "README.MD"):
                raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{readme_name}"
                try:
                    r = httpx.get(raw_url, timeout=15, follow_redirects=True)
                    if r.status_code == 200:
                        content = r.text
                        source_url = f"https://github.com/{owner}/{repo}"

                        # 레포 설명은 API 여유 있을 때만 시도
                        repo_desc = ""
                        meta_r = httpx.get(
                            f"https://api.github.com/repos/{owner}/{repo}",
                            headers=headers, timeout=10,
                        )
                        if meta_r.status_code == 200:
                            repo_desc = meta_r.json().get("description") or ""

                        markdown = f"# {repo}\n\n"
                        if repo_desc:
                            markdown += f"**설명:** {repo_desc}\n\n"
                        markdown += f"**GitHub:** {source_url}\n\n---\n\n"
                        markdown += content

                        raw_chunks = chunk_markdown(markdown, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        total = len(raw_chunks)
                        for idx, text in enumerate(raw_chunks):
                            all_chunks.append(Chunk(
                                chunk_id=make_chunk_id(source_url, idx),
                                source_url=source_url,
                                canonical_url=source_url,
                                title=f"Panopto GitHub — {repo}",
                                product="panopto",
                                guide=repo,
                                category=category,
                                role=None,
                                chunk_index=idx,
                                chunk_total=total,
                                text=text,
                                char_count=len(text),
                                word_count=len(text.split()),
                                content_hash=sha256_of(text),
                            ))
                        console.print(f"  [dim]{owner}/{repo}: {total} chunk(s)[/dim]")
                        fetched = True
                        break
                except Exception as e:
                    console.print(f"  [yellow]WARN {owner}/{repo}: {e}[/yellow]")
                time.sleep(0.2)
            if fetched:
                break

    console.print(f"  [green]GitHub → {len(all_chunks)} chunks[/green]")
    return all_chunks


def fetch_github_code_chunks(chunk_size: int, chunk_overlap: int) -> list[Chunk]:
    """README 없는 코드 샘플 레포: git clone → 파일별 청크."""
    import subprocess, tempfile, shutil
    console.print("[blue]Fetching GitHub code-only repos (git clone)...[/blue]")
    all_chunks: list[Chunk] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for owner_repo, info in _GITHUB_CODE_REPOS.items():
            owner, repo = owner_repo.split("/")
            clone_url = f"https://github.com/{owner}/{repo}.git"
            dest = Path(tmpdir) / repo
            try:
                subprocess.run(
                    ["git", "clone", "--depth=1", "--quiet", clone_url, str(dest)],
                    check=True, capture_output=True, timeout=60,
                )
            except Exception as e:
                console.print(f"  [yellow]WARN clone {owner_repo}: {e}[/yellow]")
                continue

            source_base = f"https://github.com/{owner}/{repo}"
            repo_header = (
                f"# {repo}\n\n"
                f"**설명:** {info['desc']}\n"
                f"**GitHub:** {source_base}\n\n"
            )
            for rel_path, lang in info["files"]:
                fpath = dest / rel_path
                if not fpath.exists():
                    console.print(f"  [dim]SKIP {repo}/{rel_path} (not found)[/dim]")
                    continue
                content = fpath.read_text(encoding="utf-8", errors="replace")
                file_source_url = f"{source_base}/blob/master/{rel_path}"

                if lang in ("rst", "text"):
                    markdown = f"{repo_header}## {rel_path}\n\n{content}"
                else:
                    markdown = f"{repo_header}## {rel_path}\n\n```{lang}\n{content}\n```"

                raw_chunks = chunk_markdown(markdown, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                total = len(raw_chunks)
                for idx, text in enumerate(raw_chunks):
                    all_chunks.append(Chunk(
                        chunk_id=make_chunk_id(file_source_url, idx),
                        source_url=file_source_url,
                        canonical_url=source_base,
                        title=f"Panopto GitHub — {repo}/{rel_path}",
                        product="panopto",
                        guide=repo,
                        category="github-sample",
                        role=None,
                        chunk_index=idx,
                        chunk_total=total,
                        text=text,
                        char_count=len(text),
                        word_count=len(text.split()),
                        content_hash=sha256_of(text),
                    ))
                console.print(f"  [dim]{owner}/{repo}/{rel_path}: {total} chunk(s)[/dim]")

    console.print(f"  [green]Code repos → {len(all_chunks)} chunks[/green]")
    return all_chunks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--out", default="data/chunks/panopto_chunks.jsonl", show_default=True,
              type=click.Path(), help="출력 JSONL 경로")
@click.option("--github-token", envvar="GITHUB_TOKEN", default=None,
              help="GitHub Personal Access Token (rate limit 60→5000 req/h)")
@click.option("--chunk-size", default=900, show_default=True)
@click.option("--chunk-overlap", default=120, show_default=True)
@click.option("--skip-swagger", is_flag=True, help="Swagger 수집 건너뜀")
@click.option("--skip-github", is_flag=True, help="GitHub README 레포 건너뜀")
@click.option("--skip-code", is_flag=True, help="GitHub 코드 전용 레포 건너뜀")
def main(
    out: str,
    github_token: str | None,
    chunk_size: int,
    chunk_overlap: int,
    skip_swagger: bool,
    skip_github: bool,
    skip_code: bool,
) -> None:
    """Panopto API Docs + GitHub 레포 → panopto_chunks.jsonl 생성."""
    all_chunks: list[Chunk] = []

    if not skip_swagger:
        all_chunks.extend(fetch_swagger_chunks(chunk_size, chunk_overlap))

    if not skip_github:
        all_chunks.extend(fetch_github_chunks(github_token, chunk_size, chunk_overlap))

    if not skip_code:
        all_chunks.extend(fetch_github_code_chunks(chunk_size, chunk_overlap))

    console.print(f"\n[bold]총 {len(all_chunks)} chunks → {out}[/bold]")
    save_chunks(all_chunks, Path(out))
    console.print(f"[green]완료. 다음 명령으로 Qdrant에 적재:[/green]")
    console.print(
        f"  uv run python -m src.indexing.qdrant_index "
        f"--chunks {out} --collection canvas_guides"
    )


if __name__ == "__main__":
    main()
