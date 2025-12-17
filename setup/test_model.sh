#!/bin/bash
##############################################################################
# Test models (individual, by category, or all)
#
# Usage:
#   ./setup/test_model.sh --model ltx-video
#   ./setup/test_model.sh --opensource
#   ./setup/test_model.sh --commercial
#   ./setup/test_model.sh --all
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/share.sh"

usage() {
    cat <<USAGE
Usage: $(basename "$0") [--model <name>|--all|--opensource|--commercial]

Options:
  --model <name>       Model name (test single model)
  --all                Test all models (both open-source and commercial)
  --opensource         Test only open-source models
  --commercial         Test only commercial models
  --list               List all available models
  -h, --help           Show this help

Examples:
  ./setup/test_model.sh --list
  ./setup/test_model.sh --model ltx-video
  ./setup/test_model.sh --opensource
  ./setup/test_model.sh --commercial
  ./setup/test_model.sh --all
USAGE
}

list_models() {
    print_header "Available Models"
    
    echo "OPEN-SOURCE MODELS (${#OPENSOURCE_MODELS[@]}):"
    echo ""
    for model in "${OPENSOURCE_MODELS[@]}"; do
        if model_venv_exists "$model"; then
            echo "  ✓ ${model}"
        else
            echo "  ✗ ${model} (not installed)"
        fi
    done
    
    echo ""
    echo "COMMERCIAL MODELS (${#COMMERCIAL_MODELS[@]}):"
    echo ""
    for model in "${COMMERCIAL_MODELS[@]}"; do
        local api_key
        api_key=$(get_commercial_env_var "$model")
        if model_venv_exists "$model"; then
            echo "  ✓ ${model} (requires ${api_key})"
        else
            echo "  ✗ ${model} (not installed, requires ${api_key})"
        fi
    done
    
    echo ""
    echo "Total: $((${#OPENSOURCE_MODELS[@]} + ${#COMMERCIAL_MODELS[@]})) models"
    echo ""
}

MODEL=""
TEST_ALL=false
TEST_OPENSOURCE=false
TEST_COMMERCIAL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --all)
            TEST_ALL=true
            shift
            ;;
        --opensource)
            TEST_OPENSOURCE=true
            shift
            ;;
        --commercial)
            TEST_COMMERCIAL=true
            shift
            ;;
        --list)
            list_models
            exit 0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            if [[ -z "$MODEL" && "$TEST_ALL" == "false" && "$TEST_OPENSOURCE" == "false" && "$TEST_COMMERCIAL" == "false" ]]; then
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

if [[ -z "$MODEL" && "$TEST_ALL" == "false" && "$TEST_OPENSOURCE" == "false" && "$TEST_COMMERCIAL" == "false" ]]; then
    usage
    exit 1
fi

# Check for conflicting options
OPTION_COUNT=0
[[ -n "$MODEL" ]] && ((OPTION_COUNT++)) || true
[[ "$TEST_ALL" == "true" ]] && ((OPTION_COUNT++)) || true
[[ "$TEST_OPENSOURCE" == "true" ]] && ((OPTION_COUNT++)) || true
[[ "$TEST_COMMERCIAL" == "true" ]] && ((OPTION_COUNT++)) || true

if [[ $OPTION_COUNT -gt 1 ]]; then
    print_error "Cannot specify multiple test options (--model, --all, --opensource, --commercial)"
    exit 1
fi

# Build list of models to test
MODELS_TO_TEST=()

if [[ "$TEST_ALL" == "true" ]]; then
    print_header "Testing ALL models"
    MODELS_TO_TEST=("${OPENSOURCE_MODELS[@]}" "${COMMERCIAL_MODELS[@]}")
elif [[ "$TEST_OPENSOURCE" == "true" ]]; then
    print_header "Testing OPEN-SOURCE models only"
    MODELS_TO_TEST=("${OPENSOURCE_MODELS[@]}")
elif [[ "$TEST_COMMERCIAL" == "true" ]]; then
    print_header "Testing COMMERCIAL models only"
    MODELS_TO_TEST=("${COMMERCIAL_MODELS[@]}")
else
    if ! is_opensource_model "$MODEL" && ! is_commercial_model "$MODEL"; then
        print_error "Unknown model: ${MODEL}"
        exit 1
    fi
    MODELS_TO_TEST=("$MODEL")
fi

# Test each model
VALIDATION_FAILED=()
VALIDATION_PASSED=()
NOT_INSTALLED=()

for model in "${MODELS_TO_TEST[@]}"; do
    print_header "Testing: ${model}"
    
    # Check if environment exists
    if ! model_venv_exists "$model"; then
        print_error "Virtual environment not found for ${model}. Run install_model.sh first."
        NOT_INSTALLED+=("${model}")
        continue
    fi
    
    # Run validation
    if validate_model "$model"; then
        VALIDATION_PASSED+=("${model}")
    else
        VALIDATION_FAILED+=("${model}")
    fi
done

# Print final summary
print_header "Test Summary"

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

if [[ ${#NOT_INSTALLED[@]} -gt 0 ]]; then
    echo ""
    print_warning "Not Installed (${#NOT_INSTALLED[@]}):"
    for model in "${NOT_INSTALLED[@]}"; do
        echo "      ⊘ ${model}"
    done
fi

# Exit with error if any tests failed
if [[ ${#VALIDATION_FAILED[@]} -gt 0 ]] || [[ ${#NOT_INSTALLED[@]} -gt 0 ]]; then
    echo ""
    print_header "❌ Some tests failed or models not installed"
    exit 1
fi

print_header "✅ All tests passed successfully"

