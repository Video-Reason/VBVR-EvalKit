#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="hunyuan-video-i2v"

print_section "Conda Environment (Python 3.10)"
create_model_conda_env "$MODEL" "3.10"
activate_model_conda_env "$MODEL"

print_section "Dependencies"

# Install PyTorch with CUDA support
pip install -q torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1

# Install flash-attn first (needs ninja + --no-build-isolation)
pip install -q ninja
pip install -q --no-cache-dir "flash-attn>=2.7.0" --no-build-isolation

# Install all deps from requirements.txt
pip install -q -r "${SCRIPT_DIR}/requirements.txt"

# Install deepspeed without building CUDA ops (only needed at import time, not for inference)
DS_BUILD_OPS=0 pip install -q "deepspeed==0.18.6"

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
    hf download tencent/HunyuanVideo-I2V \
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
    hf download xtuner/llava-llama-3-8b-v1_1-transformers \
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
    hf download openai/clip-vit-large-patch14 \
        --local-dir "${CLIP_TEXT_ENCODER_DIR}" \
        --local-dir-use-symlinks False \
        --resume-download
    print_success "CLIP text encoder ready at ${CLIP_TEXT_ENCODER_DIR}"
fi

conda deactivate

print_success "${MODEL} setup complete"
