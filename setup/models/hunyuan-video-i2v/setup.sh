#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/common.sh"

MODEL="hunyuan-video-i2v"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q diffusers==0.31.0 transformers==4.39.3 accelerate==1.1.1
pip install -q Pillow 'numpy<2' pandas tqdm pydantic pydantic-settings python-dotenv requests httpx
pip install -q opencv-python==4.8.1.78 einops imageio imageio-ffmpeg safetensors peft loguru deepspeed

deactivate

print_section "Checkpoints"
print_info "Weights download on first run"

print_success "${MODEL} setup complete"
