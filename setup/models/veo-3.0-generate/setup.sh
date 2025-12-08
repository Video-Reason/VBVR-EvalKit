#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/common.sh"

MODEL="veo-3.0-generate"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q google-genai
pip install -q pydantic pydantic-settings python-dotenv
pip install -q Pillow numpy imageio imageio-ffmpeg

deactivate

print_section "API Configuration"
load_env_file
ENV_VAR="$(get_commercial_env_var "$MODEL")"
if check_api_key "$ENV_VAR"; then
    value="${!ENV_VAR}"
    print_success "${ENV_VAR} configured"
else
    print_warning "${ENV_VAR} not set. Add to ${VMEVAL_ROOT}/.env"
fi

print_success "${MODEL} setup complete"
