#!/bin/bash
##############################################################################
# VideoCrafter2 512 Test Script
# Tests the model by generating sample videos
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/common.sh"

MODEL_NAME="videocrafter2-512"

print_header "Testing ${MODEL_NAME}"

# Check if environment exists
if ! model_venv_exists "$MODEL_NAME"; then
    print_error "Virtual environment not found. Run setup.sh first."
    exit 1
fi

# Check if checkpoint exists
CHECKPOINT_PATH="${MODEL_CHECKPOINT_PATHS[$MODEL_NAME]}"
if ! checkpoint_exists "$CHECKPOINT_PATH"; then
    print_error "Model checkpoint not found. Run setup.sh first."
    print_info "Expected: ${SUBMODULES_DIR}/${CHECKPOINT_PATH}"
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

