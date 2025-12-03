#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/common.sh"

MODEL="ltx-video-13b-distilled"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
pip install -q diffusers transformers accelerate sentencepiece xformers==0.0.28.post3 einops
pip install -q numpy Pillow pandas tqdm pydantic pydantic-settings python-dotenv requests httpx imageio imageio-ffmpeg

deactivate

print_section "Checkpoints"
print_info "Weights download on first run"

print_success "${MODEL} setup complete"
