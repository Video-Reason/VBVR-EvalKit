#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="LTX-2"
LTX2_DIR="${VMEVAL_ROOT}/LTX-2"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
# Clone LTX-2 repository to project root
if [ -d "${LTX2_DIR}" ] && [ -d "${LTX2_DIR}/packages" ]; then
    print_skip "LTX-2 repository already exists"
else
    print_info "Cloning LTX-2 repository..."
    git clone https://github.com/Lightricks/LTX-2.git "${LTX2_DIR}"
fi

cd "${LTX2_DIR}"

# Install dependencies using uv sync
print_info "Installing dependencies..."
uv sync

source .venv/bin/activate

print_section "Checkpoints"
cd "${LTX2_DIR}"

# Download Gemma-3 model (check if already exists)
GEMMA_DIR="${LTX2_DIR}/gemma3-12b-it-qat-q4_0-unquantized"
if [ -d "${GEMMA_DIR}" ] && [ -n "$(ls -A "${GEMMA_DIR}" 2>/dev/null)" ]; then
    print_skip "Gemma-3 model already downloaded"
else
    print_info "Downloading Gemma-3 model..."
    hf download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir gemma3-12b-it-qat-q4_0-unquantized
fi

# Download LTX-2 checkpoint (check if already exists)
CHECKPOINT_FILE="${LTX2_DIR}/ltx-2-19b-distilled-fp8.safetensors"
if [ -f "${CHECKPOINT_FILE}" ]; then
    print_skip "LTX-2 checkpoint already downloaded"
else
    print_info "Downloading LTX-2 checkpoint..."
    huggingface-cli download Lightricks/LTX-2 ltx-2-19b-distilled-fp8.safetensors --local-dir ./
fi

print_success "${MODEL} setup complete"
# Note: Requires ~40GB VRAM, one A6000 takes ~6 mins for inference
