#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="morphic-frames-to-video"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
# Follow Morphic's official setup_env.sh exactly
pip install -q torch==2.7.1 torchvision==0.22.1

# Build flash-attn from source (like original Morphic setup)
# Install build dependencies required by flash-attn
pip install -q packaging ninja psutil wheel setuptools

# Install flash-attn with --no-cache-dir to avoid cross-device link errors
# This can occur when pip cache and temp directories are on different filesystems
pip install -q --no-cache-dir flash-attn==2.7.0.post2 --no-build-isolation

# Install compatible versions to avoid dependency hell:
# - transformers<=4.51.3 (required by Morphic, no modeling_layers module)
# - peft<0.13.0 (peft 0.13.0+ requires transformers.modeling_layers)
# - diffusers<0.33.0 (diffusers 0.33.0+ requires peft>=0.17.0)
pip install -q "diffusers>=0.31.0,<0.33.0"
pip install -q "peft>=0.12.0,<0.13.0"

# Install decord explicitly (sometimes fails silently from requirements.txt)
pip install -q decord==0.6.0

# Install from requirements.txt (respects version ranges from original)
pip install -q -r "${SUBMODULES_DIR}/morphic-frames-to-video/requirements.txt"

# Additional utilities for VMEvalKit validation and runtime
pip install -q Pillow==10.4.0
pip install -q pydantic==2.12.5 pydantic-settings==2.12.0 python-dotenv==1.2.1
pip install -q requests==2.32.5 httpx==0.28.1
# Note: This version conflicts with transformers 4.51.3's preference (>=0.30.0)
# but is pinned for VMEvalKit consistency. The warning is non-critical.
pip install -q "huggingface_hub[cli]>=0.26.2"

print_section "Checkpoints"
ensure_morphic_assets

deactivate

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

