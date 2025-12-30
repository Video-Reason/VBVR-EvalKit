#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="dynamicrafter-512"

print_section "System Dependencies"
ensure_ffmpeg_dependencies

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
# Use torch 2.1.0 + xformers 0.0.22 for H100 (SM90) support
pip install -q torch==2.1.0+cu118 torchvision==0.16.0+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q xformers==0.0.22.post7+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q numpy==1.24.2 decord==0.6.0 einops==0.3.0 imageio==2.9.0 omegaconf==2.1.1
pip install -q opencv-python==4.8.1.78 Pillow==9.5.0 pytorch_lightning==1.9.3 PyYAML==6.0
pip install -q tqdm==4.65.0 transformers==4.25.1 moviepy==1.0.3 av==13.1.0
pip install -q gradio==4.44.1 timm==0.9.16 kornia==0.7.2 pandas==2.0.0 pydantic==2.12.5 pydantic-settings==2.12.0 python-dotenv==1.2.1 requests==2.32.5 httpx==0.28.1 imageio-ffmpeg==0.6.0
pip install -q open-clip-torch==2.20.0

deactivate

print_section "Checkpoints"
download_checkpoint_by_path "${MODEL_CHECKPOINT_PATHS[$MODEL]}"

print_success "${MODEL} setup complete"

