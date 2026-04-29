# M3 Pro MacBook GPU 사용 Runbook

## 원칙

M3 Pro MacBook에서 CUDA는 없다. Apple Metal 기반 런타임만 사용한다.

| 역할 | 런타임 | 비고 |
|---|---|---|
| Embedding (bge-m3) | PyTorch MPS | Metal Performance Shaders |
| LLM 추론 | Ollama | Metal GPU acceleration |
| 벡터 DB | Docker (Qdrant) | GPU 불필요 |

**절대 하지 않는 것:**
- CUDA 설치 시도
- Docker 컨테이너 안에서 Apple GPU 사용 시도
- 18GB 이하 unified memory에서 14B 이상 모델 기본값 사용

---

## 1. 하드웨어 확인

```bash
system_profiler SPHardwareDataType | grep -E "Chip|Memory"
# 예: Apple M3 Pro, Memory: 36 GB
```

---

## 2. MPS (PyTorch) 확인

```bash
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("mps_available:", torch.backends.mps.is_available())
print("mps_built:", torch.backends.mps.is_built())
PY
```

예상:
```
mps_available: True
mps_built: True
```

---

## 3. bge-m3 임베딩 설정

```bash
# .env
EMBEDDING_PROVIDER=auto
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3
LOCAL_EMBEDDING_DEVICE=mps
EMBEDDING_BATCH_SIZE=8   # 반드시 8 이하 — batch=64 시 21GB OOM 발생
```

`MpsEmbedder`는 `src/indexing/embedder.py`에 구현됨.
- 첫 호출 시 모델 lazy load (~7초)
- 이후 warm 상태에서 쿼리당 ~30ms

OpenAI API fallback:
```bash
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

---

## 4. Ollama Metal LLM

```bash
# 설치
brew install ollama

# 서버 시작 (Metal 자동 활성화)
ollama serve

# 모델 다운로드
ollama pull qwen2.5:7b-instruct   # 36GB: 7B/14B 가능
```

메모리별 권장 모델:

| Unified Memory | 권장 모델 | 비고 |
|---:|---|---|
| 18GB | qwen2.5:7b-instruct | 기본값 |
| 36GB | qwen2.5:7b-instruct 또는 14b | 14B 테스트 가능 |
| 48GB+ | 14B 또는 일부 32B quantized | 응답 지연 감수 |

동작 확인:
```bash
curl http://localhost:11434/api/generate \
  -d '{"model":"qwen2.5:7b-instruct","prompt":"Canvas LMS를 한 문장으로 설명해줘.","stream":false}'
```

---

## 5. Qdrant (Docker)

```bash
docker compose -f docker-compose.macos.yml up -d qdrant
curl http://localhost:6333/readyz
```

OrbStack 사용 시 Docker CLI는 `~/.orbstack/bin/docker`에 있음.

---

## 6. 전체 스택 smoke test

```bash
# Qdrant
curl -s http://localhost:6333/collections/canvas_guides | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print('points:', d['result']['points_count'])"

# Ollama
curl -s http://localhost:11434/api/tags | python3 -c \
  "import sys,json; d=json.load(sys.stdin); [print(m['name']) for m in d['models']]"

# Retriever (bge-m3 MPS)
uv run python -m src.retrieval.smoke_test \
  --query "Canvas에서 assignment 제출하는 방법" \
  --top-k 3

# API end-to-end
curl -s -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"Canvas에서 assignment 제출하는 방법은?"}' | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print('sources:', len(d['sources'])); print(d['answer'][:200])"
```

---

## 7. 트러블슈팅

| 증상 | 원인 | 해결 |
|---|---|---|
| `RuntimeError: Invalid buffer size: 21.29 GiB` | bge-m3 encode batch 너무 큼 | `EMBEDDING_BATCH_SIZE=8` |
| `AttributeError: 'QdrantClient' object has no attribute 'search'` | qdrant-client ≥1.17 API 변경 | `query_points()` 사용 (이미 수정됨) |
| `ollama is not installed` | Ollama 미설치 | `brew install ollama` |
| API sources=0 (canvas 질문인데) | role 필터로 0 결과 | ChatHandler가 자동 fallback (이미 수정됨) |
| HF_TOKEN 경고 | HuggingFace 인증 없음 | `.env`에 `HF_TOKEN=hf_...` 추가 (선택) |
