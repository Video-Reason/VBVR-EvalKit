#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="sana-video-2b-longlive"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -q diffusers==0.35.2 transformers==4.57.3 accelerate==1.12.0 sentencepiece==0.2.1 xformers==0.0.28.post3
pip install -q numpy==2.2.6 Pillow==12.0.0 tqdm==4.67.1 pydantic==2.12.5 pydantic-settings==2.12.0 python-dotenv==1.2.1 imageio==2.37.2 imageio-ffmpeg==0.6.0

deactivate

print_section "Checkpoints"
print_info "Weights download on first run via HuggingFace"

print_success "${MODEL} setup complete"

