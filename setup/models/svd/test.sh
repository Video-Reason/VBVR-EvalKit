#!/bin/bash
##############################################################################
# Stable Video Diffusion (SVD) Test Script
# Tests the model by generating sample videos
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/common.sh"

MODEL_NAME="svd"

print_header "Testing ${MODEL_NAME}"

# Check if environment exists
if ! model_venv_exists "$MODEL_NAME"; then
    print_error "Virtual environment not found. Run setup.sh first."
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

