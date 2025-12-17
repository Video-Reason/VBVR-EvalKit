#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="cogvideox1.5-5b-i2v"

print_section "System Dependencies"
ensure_ffmpeg_dependencies

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.5.1+cu118 torchvision==0.20.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q diffusers==0.32.2 transformers==4.46.2 accelerate==1.2.1 imageio-ffmpeg==0.5.1 sentencepiece==0.2.0
pip install -q Pillow==10.4.0 numpy==1.24.4 pydantic==2.10.6 pydantic-settings==2.7.1 python-dotenv==1.2.1 requests==2.32.3
pip install -q opencv-python==4.10.0.84

deactivate

print_section "Model Weights"
# Diffusers auto-downloads from HuggingFace on first run
print_info "Model weights will be downloaded on first run (~11GB)"
print_info "HuggingFace repo: THUDM/CogVideoX1.5-5B-I2V"
print_info "Cache location: ~/.cache/huggingface/hub/models--THUDM--CogVideoX1.5-5B-I2V"

print_success "${MODEL} setup complete"
print_info "Generated videos: 10 seconds (81 frames @ 16fps) at 1360x768 resolution"
print_info "GPU Memory: ~10GB with optimizations (sequential offload + VAE tiling)"

