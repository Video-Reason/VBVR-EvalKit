#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="morphic-frames-to-video"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
# Install flash-attn from prebuilt wheel to avoid cross-device link error during build
pip install -q https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.2.post1/flash_attn-2.7.2.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -q diffusers==0.31.0 transformers==4.51.3 accelerate==1.1.1 sentencepiece==0.2.1 xformers==0.0.28.post3 einops==0.8.1 decord==0.6.0 av==14.4.0 omegaconf==2.3.0
pip install -q opencv-python==4.11.0.86 opencv-contrib-python==4.11.0.86
pip install -q "numpy>=1.23.5,<2" Pillow==10.4.0 pandas==2.2.3 tqdm==4.67.1 pydantic==2.10.3 pydantic-settings==2.6.1 python-dotenv==1.0.1 requests==2.32.3 httpx==0.28.1
pip install -q imageio==2.36.0 imageio-ffmpeg==0.5.1 matplotlib==3.9.2 moviepy==1.0.3 cairosvg==2.7.1 ftfy==6.3.1 aiohttp==3.10.10 tenacity==9.0.0 boto3==1.35.50
pip install -q peft==0.13.2
pip install -q -r "${SUBMODULES_DIR}/morphic-frames-to-video/requirements.txt" --no-deps
pip install -q easydict dashscope librosa "tokenizers>=0.20.3"

deactivate

print_section "Checkpoints"
ensure_morphic_assets

print_section "Creating Symlinks"
# Create symlink in submodule directory to access weights
MORPHIC_SUBMODULE="${SUBMODULES_DIR}/morphic-frames-to-video"
if [[ ! -L "${MORPHIC_SUBMODULE}/weights" ]]; then
    ln -sf "${WEIGHTS_DIR}" "${MORPHIC_SUBMODULE}/weights"
    print_success "Created weights symlink: ${MORPHIC_SUBMODULE}/weights -> ${WEIGHTS_DIR}"
else
    print_skip "Weights symlink already exists"
fi

print_success "${MODEL} setup complete"

