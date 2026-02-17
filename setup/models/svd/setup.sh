#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="svd"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch torchvision torchaudio
pip install -q diffusers transformers accelerate
pip install -q numpy Pillow pandas tqdm pydantic pydantic-settings python-dotenv requests httpx imageio imageio-ffmpeg

deactivate

print_section "Checkpoints"
print_info "Weights download on first run"

print_success "${MODEL} setup complete"
