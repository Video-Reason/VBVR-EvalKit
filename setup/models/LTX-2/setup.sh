#!/bin/bash
set -x
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

# Install dependencies using pip (based on uv.lock requirements)
print_info "Installing PyTorch stack..."
pip install -q "torch~=2.7" "torchaudio>=2.5.0" "torchvision>=0.21.0"

print_info "Installing core dependencies..."
pip install -q einops numpy transformers safetensors "accelerate>=1.2.1" "scipy>=1.14"
pip install -q "av>=14.2.1" tqdm "pillow>=10.0.0"
pip install -q xformers
pip install -q python-dotenv

print_info "Installing additional LTX-2 dependencies..."
# Image/video processing
pip install -q "imageio>=2.37.0" "imageio-ffmpeg>=0.6.0" "opencv-python>=4.11.0.86"
pip install -q "pillow-heif>=0.21.0" "scenedetect>=0.6.5.2"

# ML/quantization tools
pip install -q "optimum-quanto>=0.2.6" "peft>=0.14.0"
pip install -q "bitsandbytes>=0.45.2"  # Linux only

# Utilities
pip install -q "pydantic>=2.10.4" "rich>=13.9.4" "typer>=0.15.1"
pip install -q "sentencepiece>=0.2.0" "pandas>=2.2.3"
pip install -q "setuptools>=80.9.0"

# Install local packages in editable mode
print_info "Installing LTX-2 local packages..."
pip install -q -e "${LTX2_DIR}/packages/ltx-core"
pip install -q -e "${LTX2_DIR}/packages/ltx-pipelines"

# Hugging Face Hub with CLI and xet support
pip install -q "huggingface_hub[cli,hf-xet]>=0.31.4"


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
    hf download Lightricks/LTX-2 ltx-2-19b-distilled-fp8.safetensors --local-dir ./
fi

print_success "${MODEL} setup complete"
# Note: Requires ~40GB VRAM, one A6000 takes ~6 mins for inference
