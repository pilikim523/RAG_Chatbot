#!/bin/bash
# WSL2 Ubuntu 22.04 — Canvas RAG Chatbot 서버 환경 설치
#
# 전제 조건 (Windows 측):
#   1. Windows 11 또는 Windows 10 21H2+
#   2. NVIDIA 드라이버 527.41+ 설치 (GeForce/Quadro 공통)
#   3. WSL2 활성화: PowerShell(관리자) → wsl --install -d Ubuntu-22.04
#   4. WSL2 Ubuntu 22.04 터미널에서 본 스크립트 실행

set -euo pipefail

REPO_DIR="${HOME}/RAG_Chatbot"

echo "======================================"
echo " Canvas RAG Chatbot — WSL2 CUDA 설치"
echo "======================================"

# ── 1. 시스템 업데이트 ──────────────────────────────────────────────────────
echo ""
echo "[1/8] 시스템 패키지 업데이트..."
sudo apt-get update -qq && sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq \
  curl wget git build-essential ca-certificates gnupg lsb-release \
  jq htop net-tools

# ── 2. WSL2 systemd 활성화 ──────────────────────────────────────────────────
echo ""
echo "[2/8] WSL2 systemd 활성화..."
if ! grep -q "systemd=true" /etc/wsl.conf 2>/dev/null; then
  sudo tee /etc/wsl.conf > /dev/null <<'EOF'
[boot]
systemd=true

[interop]
appendWindowsPath=false
EOF
  echo "  → /etc/wsl.conf 작성 완료. 설치 완료 후 WSL2 재시작 필요: wsl --shutdown"
fi

# ── 3. CUDA Toolkit 12.4 설치 (드라이버는 Windows에서 공유) ────────────────
echo ""
echo "[3/8] CUDA Toolkit 12.4 설치..."
if ! command -v nvcc &>/dev/null; then
  wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  rm cuda-keyring_1.1-1_all.deb
  sudo apt-get update -qq
  sudo apt-get install -y -qq cuda-toolkit-12-4
fi

# CUDA PATH 설정
if ! grep -q "cuda/bin" ~/.bashrc; then
  echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
  echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
fi
export PATH=/usr/local/cuda/bin:$PATH

# ── 4. Docker 설치 ─────────────────────────────────────────────────────────
echo ""
echo "[4/8] Docker 설치..."
if ! command -v docker &>/dev/null; then
  sudo install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
    | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  sudo chmod a+r /etc/apt/keyrings/docker.gpg
  echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" \
    | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
  sudo usermod -aG docker "$USER"
  echo "  → docker 그룹 추가됨 (재로그인 후 적용)"
fi

# ── 5. NVIDIA Container Toolkit ────────────────────────────────────────────
echo ""
echo "[5/8] NVIDIA Container Toolkit 설치..."
if ! dpkg -l nvidia-container-toolkit &>/dev/null; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update -qq
  sudo apt-get install -y -qq nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

# ── 6. uv 설치 ─────────────────────────────────────────────────────────────
echo ""
echo "[6/8] uv (Python 패키지 매니저) 설치..."
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:$PATH"
fi

# ── 7. Ollama 설치 ─────────────────────────────────────────────────────────
echo ""
echo "[7/8] Ollama 설치 및 모델 다운로드..."
if ! command -v ollama &>/dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi

# Ollama systemd 서비스 활성화
sudo systemctl enable ollama
sudo systemctl start ollama
sleep 3

# LLM 모델 다운로드
echo "  → exaone3.5:7.8b 다운로드 중 (약 5GB)..."
ollama pull exaone3.5:7.8b

# ── 8. 앱 의존성 설치 ─────────────────────────────────────────────────────
echo ""
echo "[8/8] Python 의존성 설치..."
if [ -d "$REPO_DIR" ]; then
  cd "$REPO_DIR"
  uv sync
  uv sync --group embedding
fi

echo ""
echo "======================================"
echo " 설치 완료"
echo "======================================"
echo ""
echo "다음 단계:"
echo "  1. PowerShell에서 WSL2 재시작: wsl --shutdown"
echo "  2. WSL2 재시작 후 GPU 확인:   nvidia-smi"
echo "  3. 데이터 이전:               bash scripts/migrate_data.sh"
echo "  4. 서비스 시작:               bash scripts/start_service.sh"
