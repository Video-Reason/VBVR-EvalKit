#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/common.sh"

MODEL="sana-video-2b-480p"

print_header "Testing ${MODEL}"

if ! model_venv_exists "$MODEL"; then
    print_error "Virtual environment not found. Run setup.sh first."
    exit 1
fi

print_section "Validation"
validate_model "$MODEL"

if [[ $? -eq 0 ]]; then
    print_success "All tests passed for ${MODEL}"
else
    print_error "Tests failed for ${MODEL}"
    exit 1
fi

