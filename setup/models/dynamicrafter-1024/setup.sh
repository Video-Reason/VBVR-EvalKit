#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/common.sh"

MODEL="dynamicrafter-1024"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q numpy==1.24.2 decord==0.6.0 einops==0.3.0 imageio==2.9.0 omegaconf==2.1.1
pip install -q opencv-python==4.8.1.78 Pillow==9.5.0 pytorch_lightning==1.9.3 PyYAML==6.0
pip install -q tqdm==4.65.0 transformers==4.25.1 moviepy==1.0.3 av xformers==0.0.18
pip install -q gradio timm kornia pandas==2.0.0 pydantic pydantic-settings python-dotenv requests httpx imageio-ffmpeg

deactivate

print_section "Checkpoints"
download_checkpoint_by_path "${MODEL_CHECKPOINT_PATHS[$MODEL]}"

print_success "${MODEL} setup complete"
