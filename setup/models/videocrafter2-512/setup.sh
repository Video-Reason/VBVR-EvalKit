#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/common.sh"

MODEL="videocrafter2-512"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q omegaconf pytorch-lightning==2.0.9 einops transformers
pip install -q Pillow numpy pandas tqdm pydantic pydantic-settings python-dotenv requests httpx
pip install -q opencv-python==4.8.1.78 imageio imageio-ffmpeg av moviepy

deactivate

print_section "Checkpoints"
download_checkpoint_by_path "${MODEL_CHECKPOINT_PATHS[$MODEL]}"

print_success "${MODEL} setup complete"

