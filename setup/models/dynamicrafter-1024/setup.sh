#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="dynamicrafter-1024"

print_section "System Dependencies"
ensure_ffmpeg_dependencies

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
# Use torch 2.1.0 + xformers 0.0.22 for H100 (SM90) support
pip install -q torch==2.7.1 torchvision==0.22.1
pip install -q xformers==0.0.31
pip install -q "numpy>=2.0.0" decord==0.6.0 einops==0.3.0 "imageio>=2.34.0" omegaconf==2.1.1
pip install -q "opencv-python>=4.9.0" "Pillow>=10.0.0" pytorch_lightning==1.9.3 "PyYAML>=6.0.1"
pip install -q "tqdm>=4.65.0" "transformers>=4.25.1" moviepy==1.0.3 av==13.1.0
pip install -q gradio==4.44.1 timm==0.9.16 kornia==0.7.2 "pandas>=2.0.0" pydantic==2.12.5 pydantic-settings==2.12.0 python-dotenv==1.2.1 requests==2.32.5 httpx==0.28.1 imageio-ffmpeg==0.6.0
pip install -q open-clip-torch==2.20.0

deactivate

print_section "Checkpoints"
download_checkpoint_by_path "$(get_model_checkpoint_path "$MODEL")"

print_success "${MODEL} setup complete"
