---
name: canvas-rag-chatbot
description: Integrate Canvas RAG retrieval into chatbot_goover_context.py, domain routing, FastAPI chat API, Ollama local LLM, and source-grounded answer generation.
argument-hint: "[task]"
allowed-tools: Read Grep Glob Bash Write Edit
---

# Canvas RAG Chatbot Skill

## 목표

Canvas 관련 질문을 internal Canvas RAG로 우선 라우팅하고, `chatbot_goover_context.py`에 retriever context를 연결한다.

## M3 Pro 로컬 LLM 정책

- 개발 환경에서는 Ollama를 macOS host에서 실행한다.
- Docker 컨테이너에서 Ollama를 띄워 Apple GPU를 쓰려는 방식을 기본값으로 쓰지 않는다.
- 기본 local model은 7B/8B 계열로 둔다.
- 36GB 이상 unified memory에서만 14B 이상을 테스트한다.
- 운영 품질이 부족하면 LLM provider만 외부 API로 교체하고 retriever/router 구조는 유지한다.

## 구현 대상

```text
src/retrieval/router.py
src/retrieval/retriever.py
src/retrieval/models.py
src/chatbot_goover_context.py
src/api/main.py
src/api/schemas.py
tests/test_router*.py
tests/test_retriever*.py
tests/test_chatbot_context*.py
```

## Ollama smoke test

```bash
ollama serve

ollama pull qwen2.5:7b-instruct

curl http://localhost:11434/api/generate \
  -d '{
    "model": "qwen2.5:7b-instruct",
    "prompt": "Canvas LMS에서 assignment가 무엇인지 한국어로 한 문장으로 설명해줘.",
    "stream": false
  }'
```

## Answer policy

LLM system message에 다음 정책을 넣는다.

```text
Canvas 관련 질문은 제공된 RAG context만 근거로 답변한다.
근거가 부족하면 부족하다고 말한다.
답변은 한국어로 작성하되 Canvas 공식 기능명은 영어 원문을 병기한다.
최종 답변에는 사용자가 바로 실행할 단계와 공식 출처를 포함한다.
```

## 완료 검증

```bash
uv run pytest -q tests/test_router*.py tests/test_retriever*.py tests/test_chatbot_context*.py

curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Canvas에서 과제 due date와 availability date 차이를 알려줘",
    "domain": "canvas",
    "user_role": "instructor"
  }'
```
