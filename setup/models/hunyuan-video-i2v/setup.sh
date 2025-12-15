#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="hunyuan-video-i2v"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q diffusers==0.31.0 transformers==4.39.3 accelerate==1.1.1
pip install -q Pillow==9.5.0 'numpy<2' pandas==2.0.0 tqdm==4.65.0 pydantic==2.12.5 pydantic-settings==2.12.0 python-dotenv==1.2.1 requests==2.32.5 httpx==0.28.1
pip install -q opencv-python==4.8.1.78 einops==0.8.1 imageio==2.37.2 imageio-ffmpeg==0.6.0 safetensors==0.5.3 peft==0.15.2 loguru==0.7.3 deepspeed==0.16.7

deactivate

print_section "Checkpoints"
print_info "Weights download on first run"

print_success "${MODEL} setup complete"
