#!/bin/bash
set -x
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="LTX-2"
LTX2_DIR="${VMEVAL_ROOT}/LTX-2"

print_section "Clone & Install"
# Clone LTX-2 repository to project root
if [ -d "${LTX2_DIR}" ] && [ -d "${LTX2_DIR}/packages" ]; then
    print_skip "LTX-2 repository already exists"
else
    print_info "Cloning LTX-2 repository..."
    git clone https://github.com/Lightricks/LTX-2.git "${LTX2_DIR}"
fi

cd "${LTX2_DIR}"

# Install via uv sync (uses uv.lock for exact dependency resolution)
print_info "Running uv sync..."
uv sync

# Symlink .venv -> envs/LTX-2 so the inference runner can find the Python interpreter
print_info "Creating venv symlink..."
mkdir -p "${VMEVAL_ROOT}/envs"
rm -rf "${VMEVAL_ROOT}/envs/${MODEL}"
ln -s "${LTX2_DIR}/.venv" "${VMEVAL_ROOT}/envs/${MODEL}"
print_success "Symlinked envs/${MODEL} -> LTX-2/.venv"

print_section "Checkpoints"
cd "${LTX2_DIR}"

# Activate the venv so hf CLI is available
source .venv/bin/activate

# Download Gemma-3 model (check if already exists)
GEMMA_DIR="${LTX2_DIR}/gemma3-12b-it-qat-q4_0-unquantized"
if find "${GEMMA_DIR}" -name '*.safetensors' -print -quit 2>/dev/null | grep -q .; then
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
    hf download Lightricks/LTX-2 ltx-2-19b-distilled-fp8.safetensors --local-dir ./
fi

print_success "${MODEL} setup complete"
# Note: Requires ~40GB VRAM, one A6000 takes ~6 mins for inference
