#!/bin/bash
##############################################################################
# Install and test individual models
#
# Usage:
#   ./setup/install_model.sh --model ltx-video
#   ./setup/install_model.sh --model ltx-video --validate
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
    cat <<USAGE
Usage: $(basename "$0") --model <name> [--validate]

Options:
  --model <name>       Model name (required)
  --validate           Test model after installation
  -h, --help           Show this help

Examples:
  ./setup/install_model.sh --model ltx-video
  ./setup/install_model.sh --model ltx-video --validate
USAGE
}

MODEL=""
VALIDATE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --validate)
            VALIDATE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -z "$MODEL" ]]; then
                MODEL="$1"
                shift
            else
                print_error "Unknown argument: $1"
                usage
                exit 1
            fi
            ;;
    esac
done

if [[ -z "$MODEL" ]]; then
    usage
    exit 1
fi

if ! is_opensource_model "$MODEL" && ! is_commercial_model "$MODEL"; then
    print_error "Unknown model: ${MODEL}"
    exit 1
fi

print_header "Installing: ${MODEL}"

# Run model-specific setup script
SETUP_SCRIPT="${SCRIPT_DIR}/models/${MODEL}/setup.sh"

if [[ ! -f "$SETUP_SCRIPT" ]]; then
    print_error "Setup script not found: ${SETUP_SCRIPT}"
    exit 1
fi

bash "$SETUP_SCRIPT"

# Run validation if requested
if [[ "$VALIDATE" == "true" ]]; then
    print_section "Validation"
    
    if ! model_venv_exists "$MODEL"; then
        print_error "Virtual environment not found for ${MODEL}"
        exit 1
    fi
    
    validate_model "$MODEL"
fi

print_header "✅ ${MODEL} ready"
