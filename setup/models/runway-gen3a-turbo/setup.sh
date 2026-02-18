#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="runway-gen3a-turbo"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q httpx==0.28.1 aiohttp==3.12.12 requests==2.32.5 tenacity==9.1.2
pip install -q pydantic==2.12.5 pydantic-settings==2.12.0 python-dotenv==1.2.1
pip install -q Pillow==12.0.0 numpy==2.2.6 imageio==2.37.2 imageio-ffmpeg==0.6.0
pip install -q boto3==1.42.11 opencv-python==4.10.0.84

deactivate

print_section "API Configuration"
load_env_file
ENV_VAR="$(get_commercial_env_var "$MODEL")"
if check_api_key "$ENV_VAR"; then
    value="${!ENV_VAR}"
    print_success "${ENV_VAR} configured (${value:0:8}...${value: -4})"
else
    print_warning "${ENV_VAR} not set. Add to ${VMEVAL_ROOT}/.env"
fi

print_success "${MODEL} setup complete"
