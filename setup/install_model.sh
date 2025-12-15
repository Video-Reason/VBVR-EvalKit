#!/bin/bash
##############################################################################
# Install and test models (individual, by category, or all)
#
# Usage:
#   ./setup/install_model.sh --model ltx-video
#   ./setup/install_model.sh --model ltx-video --validate
#   ./setup/install_model.sh --opensource
#   ./setup/install_model.sh --opensource --validate
#   ./setup/install_model.sh --commercial --validate
#   ./setup/install_model.sh --all
#   ./setup/install_model.sh --all --validate
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

usage() {
    cat <<USAGE
Usage: $(basename "$0") [--model <name>|--all|--opensource|--commercial] [--validate]

Options:
  --model <name>       Model name (install single model)
  --all                Install all models (both open-source and commercial)
  --opensource         Install only open-source models
  --commercial         Install only commercial models
  --list               List all available models
  --validate           Test model(s) after installation
  -h, --help           Show this help

Examples:
  ./setup/install_model.sh --list
  ./setup/install_model.sh --model ltx-video
  ./setup/install_model.sh --model ltx-video --validate
  ./setup/install_model.sh --opensource
  ./setup/install_model.sh --opensource --validate
  ./setup/install_model.sh --commercial --validate
  ./setup/install_model.sh --all
  ./setup/install_model.sh --all --validate
USAGE
}

list_models() {
    print_header "Available Models"
    
    echo "OPEN-SOURCE MODELS (${#OPENSOURCE_MODELS[@]}):"
    echo ""
    for model in "${OPENSOURCE_MODELS[@]}"; do
        echo "  • ${model}"
    done
    
    echo ""
    echo "COMMERCIAL MODELS (${#COMMERCIAL_MODELS[@]}):"
    echo ""
    for model in "${COMMERCIAL_MODELS[@]}"; do
        local api_key
        api_key=$(get_commercial_env_var "$model")
        echo "  • ${model} (requires ${api_key})"
    done
    
    echo ""
    echo "Total: $((${#OPENSOURCE_MODELS[@]} + ${#COMMERCIAL_MODELS[@]})) models"
    echo ""
}

MODEL=""
INSTALL_ALL=false
INSTALL_OPENSOURCE=false
INSTALL_COMMERCIAL=false
VALIDATE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --all)
            INSTALL_ALL=true
            shift
            ;;
        --opensource)
            INSTALL_OPENSOURCE=true
            shift
            ;;
        --commercial)
            INSTALL_COMMERCIAL=true
            shift
            ;;
        --list)
            list_models
            exit 0
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
            if [[ -z "$MODEL" && "$INSTALL_ALL" == "false" && "$INSTALL_OPENSOURCE" == "false" && "$INSTALL_COMMERCIAL" == "false" ]]; then
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

if [[ -z "$MODEL" && "$INSTALL_ALL" == "false" && "$INSTALL_OPENSOURCE" == "false" && "$INSTALL_COMMERCIAL" == "false" ]]; then
    usage
    exit 1
fi

# Check for conflicting options
OPTION_COUNT=0
[[ -n "$MODEL" ]] && ((OPTION_COUNT++)) || true
[[ "$INSTALL_ALL" == "true" ]] && ((OPTION_COUNT++)) || true
[[ "$INSTALL_OPENSOURCE" == "true" ]] && ((OPTION_COUNT++)) || true
[[ "$INSTALL_COMMERCIAL" == "true" ]] && ((OPTION_COUNT++)) || true

if [[ $OPTION_COUNT -gt 1 ]]; then
    print_error "Cannot specify multiple installation options (--model, --all, --opensource, --commercial)"
    exit 1
fi

# Build list of models to install
MODELS_TO_INSTALL=()

if [[ "$INSTALL_ALL" == "true" ]]; then
    print_header "Installing ALL models"
    MODELS_TO_INSTALL=("${OPENSOURCE_MODELS[@]}" "${COMMERCIAL_MODELS[@]}")
elif [[ "$INSTALL_OPENSOURCE" == "true" ]]; then
    print_header "Installing OPEN-SOURCE models only"
    MODELS_TO_INSTALL=("${OPENSOURCE_MODELS[@]}")
elif [[ "$INSTALL_COMMERCIAL" == "true" ]]; then
    print_header "Installing COMMERCIAL models only"
    MODELS_TO_INSTALL=("${COMMERCIAL_MODELS[@]}")
else
    if ! is_opensource_model "$MODEL" && ! is_commercial_model "$MODEL"; then
        print_error "Unknown model: ${MODEL}"
        exit 1
    fi
    MODELS_TO_INSTALL=("$MODEL")
fi

# Install each model
FAILED_MODELS=()
SUCCESSFUL_MODELS=()

for model in "${MODELS_TO_INSTALL[@]}"; do
    print_header "Installing: ${model}"
    
    # Run model-specific setup script
    SETUP_SCRIPT="${SCRIPT_DIR}/models/${model}/setup.sh"
    
    if [[ ! -f "$SETUP_SCRIPT" ]]; then
        print_error "Setup script not found: ${SETUP_SCRIPT}"
        FAILED_MODELS+=("${model}")
        continue
    fi
    
    if bash "$SETUP_SCRIPT"; then
        print_success "${model} installed successfully"
        SUCCESSFUL_MODELS+=("${model}")
    else
        print_error "${model} installation failed"
        FAILED_MODELS+=("${model}")
    fi
done

# Run validation if requested
if [[ "$VALIDATE" == "true" ]]; then
    print_header "Validation Phase"
    
    VALIDATION_FAILED=()
    VALIDATION_PASSED=()
    
    for model in "${SUCCESSFUL_MODELS[@]}"; do
        print_section "Validating: ${model}"
        
        if ! model_venv_exists "$model"; then
            print_error "Virtual environment not found for ${model}"
            VALIDATION_FAILED+=("${model}")
            continue
        fi
        
        if validate_model "$model"; then
            VALIDATION_PASSED+=("${model}")
        else
            VALIDATION_FAILED+=("${model}")
        fi
    done
    
    # Print validation summary
    print_header "Validation Summary"
    
    if [[ ${#VALIDATION_PASSED[@]} -gt 0 ]]; then
        print_success "Passed (${#VALIDATION_PASSED[@]}):"
        for model in "${VALIDATION_PASSED[@]}"; do
            echo "      ✓ ${model}"
        done
    fi
    
    if [[ ${#VALIDATION_FAILED[@]} -gt 0 ]]; then
        echo ""
        print_error "Failed (${#VALIDATION_FAILED[@]}):"
        for model in "${VALIDATION_FAILED[@]}"; do
            echo "      ✗ ${model}"
        done
    fi
fi

# Print final summary
print_header "Installation Summary"

if [[ ${#SUCCESSFUL_MODELS[@]} -gt 0 ]]; then
    print_success "Installed (${#SUCCESSFUL_MODELS[@]}):"
    for model in "${SUCCESSFUL_MODELS[@]}"; do
        echo "      ✓ ${model}"
    done
fi

if [[ ${#FAILED_MODELS[@]} -gt 0 ]]; then
    echo ""
    print_error "Failed (${#FAILED_MODELS[@]}):"
    for model in "${FAILED_MODELS[@]}"; do
        echo "      ✗ ${model}"
    done
    exit 1
fi

print_header "✅ All installations completed successfully"
