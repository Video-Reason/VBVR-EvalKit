#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="veo-3.1-fast"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q google-genai==1.55.0 google-generativeai==0.8.3
pip install -q pydantic==2.12.5 pydantic-settings==2.12.0 python-dotenv==1.2.1
pip install -q Pillow==12.0.0 numpy==2.2.6 imageio==2.37.2 imageio-ffmpeg==0.6.0

deactivate

print_section "API Configuration"
load_env_file
ENV_VAR="$(get_commercial_env_var "$MODEL")"
if check_api_key "$ENV_VAR"; then
    print_success "${ENV_VAR} configured"
else
    print_warning "${ENV_VAR} not set. Add to ${VMEVAL_ROOT}/.env"
fi

print_success "${MODEL} setup complete"

