#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="hunyuan-video-i2v"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"

# Install PyTorch 2.4.0 with CUDA 12.1 (compatible with system CUDA 12.8)
# Official docs recommend pytorch-cuda=12.4 via conda, we use pip equivalent
# Force reinstall to ensure torchvision custom ops (e.g., nms) are present
pip install -q torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Verify torchvision has compiled ops (fail fast if a CPU-only wheel slips in)
python - <<'PY'
import torchvision
assert hasattr(torchvision.ops, "nms"), "torchvision.ops.nms missing"
assert getattr(torchvision.extension, "_has_ops")(), "torchvision C++ ops missing"
PY

# Install EXACT versions from HunyuanVideo-I2V requirements.txt
pip install -q "opencv-python>=4.9.0"
pip install -q diffusers==0.31.0
pip install -q accelerate==1.1.1
pip install -q "pandas>=2.0.3"
pip install -q "numpy>=2.0.0"
pip install -q einops==0.7.0
pip install -q tqdm==4.66.2
pip install -q loguru==0.7.2
pip install -q imageio==2.34.0
pip install -q imageio-ffmpeg==0.5.1
pip install -q "safetensors>=0.4.3"
pip install -q peft==0.13.2
pip install -q transformers==4.39.3
pip install -q "tokenizers>=0.15.0"
pip install -q deepspeed==0.15.1
pip install -q "pyarrow>=14.0.1"
pip install -q tensorboard==2.19.0
pip install -q --no-cache-dir git+https://github.com/openai/CLIP.git

# Additional utilities for VMEvalKit
pip install -q "Pillow>=10.0.0"
pip install -q pydantic==2.12.5 pydantic-settings==2.12.0 python-dotenv==1.2.1
pip install -q requests==2.32.5 httpx==0.28.1
pip install -q "huggingface_hub[cli]>=0.26.2"

# Install flash-attn v2.6.3 (per official docs)
# Use --no-cache-dir to avoid cross-device link errors on different filesystems
pip install -q ninja
pip install -q --no-cache-dir "flash-attn>=2.7.0" --no-build-isolation

# NOTE: xfuser is NOT installed - single GPU mode only (avoids dependency conflicts)
# For multi-GPU, manually install xfuser and reinstall torch afterwards

# Final verification: torchvision must have compiled C++ ops
python - <<'VERIFY'
import torchvision
from torchvision.ops import nms
assert getattr(torchvision.extension, "_has_ops", lambda: False)(), \
    "FATAL: torchvision C++ ops missing. Check CUDA compatibility."
print("✓ torchvision C++ ops verified")
VERIFY

print_section "Checkpoints"

HUNYUAN_CKPTS_DIR="${SUBMODULES_DIR}/HunyuanVideo-I2V/ckpts"
HUNYUAN_MODEL_DIR="${HUNYUAN_CKPTS_DIR}/hunyuan-video-i2v-720p"
TEXT_ENCODER_DIR="${HUNYUAN_CKPTS_DIR}/text_encoder_i2v"
CLIP_TEXT_ENCODER_DIR="${HUNYUAN_CKPTS_DIR}/text_encoder_2"

if [[ -d "${HUNYUAN_MODEL_DIR}" ]] && [[ -n "$(ls -A "${HUNYUAN_MODEL_DIR}" 2>/dev/null)" ]]; then
    print_skip "HunyuanVideo-I2V weights already present at ${HUNYUAN_MODEL_DIR}"
else
    print_download "Downloading HunyuanVideo-I2V weights to ${HUNYUAN_CKPTS_DIR} (large download, resume supported)..."
    mkdir -p "${HUNYUAN_CKPTS_DIR}"
    huggingface-cli download tencent/HunyuanVideo-I2V \
        --local-dir "${HUNYUAN_CKPTS_DIR}" \
        --local-dir-use-symlinks False \
        --resume-download
    print_success "Weights ready at ${HUNYUAN_MODEL_DIR}"
fi

# Check if text encoder has the required preprocessor_config.json (correct model)
if [[ -d "${TEXT_ENCODER_DIR}" ]] && [[ -f "${TEXT_ENCODER_DIR}/preprocessor_config.json" ]]; then
    print_skip "HunyuanVideo-I2V text encoder already present at ${TEXT_ENCODER_DIR}"
else
    # Remove incomplete/wrong text encoder if it exists
    if [[ -d "${TEXT_ENCODER_DIR}" ]]; then
        print_download "Removing incomplete text encoder..."
        rm -rf "${TEXT_ENCODER_DIR}"
    fi
    print_download "Downloading HunyuanVideo-I2V text encoder (xtuner/llava-llama-3-8b-v1_1-transformers) to ${TEXT_ENCODER_DIR}..."
    mkdir -p "${TEXT_ENCODER_DIR}"
    huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers \
        --local-dir "${TEXT_ENCODER_DIR}" \
        --local-dir-use-symlinks False \
        --resume-download
    print_success "Text encoder ready at ${TEXT_ENCODER_DIR}"
fi

# HunyuanVideo also expects a second text encoder/tokenizer (CLIP-L) at ckpts/text_encoder_2
if [[ -d "${CLIP_TEXT_ENCODER_DIR}" ]] && [[ -n "$(ls -A "${CLIP_TEXT_ENCODER_DIR}" 2>/dev/null)" ]]; then
    print_skip "HunyuanVideo-I2V CLIP text encoder already present at ${CLIP_TEXT_ENCODER_DIR}"
else
    if [[ -d "${CLIP_TEXT_ENCODER_DIR}" ]]; then
        print_download "Removing incomplete CLIP text encoder..."
        rm -rf "${CLIP_TEXT_ENCODER_DIR}"
    fi
    print_download "Downloading CLIP-L text encoder (openai/clip-vit-large-patch14) to ${CLIP_TEXT_ENCODER_DIR}..."
    mkdir -p "${CLIP_TEXT_ENCODER_DIR}"
    huggingface-cli download openai/clip-vit-large-patch14 \
        --local-dir "${CLIP_TEXT_ENCODER_DIR}" \
        --local-dir-use-symlinks False \
        --resume-download
    print_success "CLIP text encoder ready at ${CLIP_TEXT_ENCODER_DIR}"
fi

deactivate

print_success "${MODEL} setup complete"
