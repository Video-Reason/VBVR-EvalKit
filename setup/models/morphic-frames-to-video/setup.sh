#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/common.sh"

MODEL="morphic-frames-to-video"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -q diffusers transformers accelerate sentencepiece xformers==0.0.28.post3 einops decord av omegaconf
pip install -q opencv-python opencv-contrib-python
pip install -q numpy Pillow pandas tqdm pydantic pydantic-settings python-dotenv requests httpx
pip install -q imageio imageio-ffmpeg matplotlib moviepy cairosvg ftfy aiohttp tenacity boto3
pip install -q -r "${SUBMODULES_DIR}/morphic-frames-to-video/requirements.txt"

deactivate

print_section "Checkpoints"
ensure_morphic_assets

print_success "${MODEL} setup complete"

