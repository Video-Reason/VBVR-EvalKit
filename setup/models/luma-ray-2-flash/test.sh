#!/bin/bash
##############################################################################
# Luma Ray 2 Flash Test Script
# Tests the model by generating sample videos
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/common.sh"

MODEL_NAME="luma-ray-2-flash"

print_header "Testing ${MODEL_NAME}"

# Check if environment exists
if ! model_venv_exists "$MODEL_NAME"; then
    print_error "Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Check API key
API_KEY_VAR="LUMA_API_KEY"
load_env_file
if ! check_api_key "$API_KEY_VAR"; then
    print_error "${API_KEY_VAR} not configured"
    print_info "Add to ${VMEVAL_ROOT}/.env:"
    echo "         ${API_KEY_VAR}=your_api_key_here"
    exit 1
fi

# Run validation
if validate_model "$MODEL_NAME"; then
    print_header "✅ ${MODEL_NAME} test PASSED"
    exit 0
else
    print_header "❌ ${MODEL_NAME} test FAILED"
    exit 1
fi

