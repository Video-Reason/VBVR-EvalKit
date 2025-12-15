#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="videocrafter2-512"

print_section "System Dependencies"
ensure_ffmpeg_dependencies

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q omegaconf==2.3.0 pytorch-lightning==2.0.9 einops==0.8.1 transformers==4.25.1
pip install -q Pillow==9.5.0 numpy==1.24.2 pandas==2.0.0 tqdm==4.65.0 pydantic==2.12.5 pydantic-settings==2.12.0 python-dotenv==1.2.1 requests==2.32.5 httpx==0.28.1
pip install -q opencv-python==4.8.1.78 imageio==2.9.0 imageio-ffmpeg==0.6.0 av==13.1.0 moviepy==1.0.3

deactivate

print_section "Checkpoints"
download_checkpoint_by_path "${MODEL_CHECKPOINT_PATHS[$MODEL]}"

print_success "${MODEL} setup complete"

