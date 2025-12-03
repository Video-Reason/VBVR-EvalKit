#!/bin/bash
##############################################################################
# VMEvalKit - Verify Weights Structure
##############################################################################
# This script verifies that the weights directory structure is correct
# and all expected paths are properly configured.
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

print_header "VMEvalKit Weights Structure Verification"

# Check if weights directory exists
print_section "Checking Weights Directory"
if [[ -d "${WEIGHTS_DIR}" ]]; then
    print_success "Weights directory exists: ${WEIGHTS_DIR}"
    
    # Show structure
    print_info "Current structure:"
    if command -v tree &> /dev/null; then
        tree -L 2 -d "${WEIGHTS_DIR}" 2>/dev/null || true
    else
        find "${WEIGHTS_DIR}" -maxdepth 2 -type d 2>/dev/null || true
    fi
else
    print_warning "Weights directory not found: ${WEIGHTS_DIR}"
    print_info "Run ./setup/RUN_SETUP.sh to download weights"
fi

# Check for legacy weight locations
print_section "Checking for Legacy Weight Locations"

LEGACY_FOUND=0

if [[ -d "${VMEVAL_ROOT}/Wan2.2-I2V-A14B" ]] && [[ -n "$(ls -A "${VMEVAL_ROOT}/Wan2.2-I2V-A14B" 2>/dev/null)" ]]; then
    print_warning "Legacy Wan2.2 weights found at: ${VMEVAL_ROOT}/Wan2.2-I2V-A14B"
    print_info "Run ./setup/migrate_weights.sh to migrate"
    ((LEGACY_FOUND++))
fi

if [[ -d "${VMEVAL_ROOT}/morphic-frames-lora-weights" ]] && [[ -n "$(ls -A "${VMEVAL_ROOT}/morphic-frames-lora-weights" 2>/dev/null)" ]]; then
    print_warning "Legacy Morphic weights found at: ${VMEVAL_ROOT}/morphic-frames-lora-weights"
    print_info "Run ./setup/migrate_weights.sh to migrate"
    ((LEGACY_FOUND++))
fi

if [[ -d "${SUBMODULES_DIR}/DynamiCrafter/checkpoints" ]] && [[ -n "$(ls -A "${SUBMODULES_DIR}/DynamiCrafter/checkpoints" 2>/dev/null)" ]]; then
    print_warning "Legacy DynamiCrafter weights found at: ${SUBMODULES_DIR}/DynamiCrafter/checkpoints"
    print_info "Run ./setup/migrate_weights.sh to migrate"
    ((LEGACY_FOUND++))
fi

if [[ -d "${SUBMODULES_DIR}/VideoCrafter/checkpoints" ]] && [[ -n "$(ls -A "${SUBMODULES_DIR}/VideoCrafter/checkpoints" 2>/dev/null)" ]]; then
    print_warning "Legacy VideoCrafter weights found at: ${SUBMODULES_DIR}/VideoCrafter/checkpoints"
    print_info "Run ./setup/migrate_weights.sh to migrate"
    ((LEGACY_FOUND++))
fi

if [[ $LEGACY_FOUND -eq 0 ]]; then
    print_success "No legacy weight locations found"
fi

# Check environment variables
print_section "Checking Environment Configuration"

load_env_file

if [[ -n "${MORPHIC_WAN2_CKPT_DIR:-}" ]]; then
    if [[ "$MORPHIC_WAN2_CKPT_DIR" == *"weights/wan"* ]]; then
        print_success "MORPHIC_WAN2_CKPT_DIR uses new structure: ${MORPHIC_WAN2_CKPT_DIR}"
    else
        print_warning "MORPHIC_WAN2_CKPT_DIR uses legacy path: ${MORPHIC_WAN2_CKPT_DIR}"
        print_info "Update .env to: MORPHIC_WAN2_CKPT_DIR=./weights/wan/Wan2.2-I2V-A14B"
    fi
else
    print_info "MORPHIC_WAN2_CKPT_DIR not set (will use default)"
fi

if [[ -n "${MORPHIC_LORA_WEIGHTS_PATH:-}" ]]; then
    if [[ "$MORPHIC_LORA_WEIGHTS_PATH" == *"weights/morphic"* ]]; then
        print_success "MORPHIC_LORA_WEIGHTS_PATH uses new structure: ${MORPHIC_LORA_WEIGHTS_PATH}"
    else
        print_warning "MORPHIC_LORA_WEIGHTS_PATH uses legacy path: ${MORPHIC_LORA_WEIGHTS_PATH}"
        print_info "Update .env to: MORPHIC_LORA_WEIGHTS_PATH=./weights/morphic/lora_interpolation_high_noise_final.safetensors"
    fi
else
    print_info "MORPHIC_LORA_WEIGHTS_PATH not set (will use default)"
fi

# Check git ignore
print_section "Checking Git Configuration"

if grep -q "^weights/" "${VMEVAL_ROOT}/.gitignore" 2>/dev/null; then
    print_success "weights/ directory is in .gitignore"
else
    print_error "weights/ directory not found in .gitignore"
    print_info "This should not happen - please check .gitignore file"
fi

# Check for untracked model files in git
print_section "Checking Git Status for Model Files"

cd "${VMEVAL_ROOT}"
UNTRACKED_MODELS=$(git status --porcelain 2>/dev/null | grep -E "^\?\?" | grep -E "\.(pth|safetensors|ckpt|bin|pt)$" || true)

if [[ -z "$UNTRACKED_MODELS" ]]; then
    print_success "No untracked model weight files in git"
else
    print_warning "Found untracked model weight files:"
    echo "$UNTRACKED_MODELS"
    print_info "These should be in the weights/ directory"
fi

# Summary
print_section "Summary"

TOTAL_SIZE="0"
if [[ -d "${WEIGHTS_DIR}" ]]; then
    TOTAL_SIZE=$(du -sh "${WEIGHTS_DIR}" 2>/dev/null | cut -f1 || echo "0")
fi

print_info "Weights directory: ${WEIGHTS_DIR}"
print_info "Total size: ${TOTAL_SIZE}"
print_info "Legacy locations found: ${LEGACY_FOUND}"

if [[ $LEGACY_FOUND -gt 0 ]]; then
    echo ""
    print_warning "Action Required: Run ./setup/migrate_weights.sh to migrate legacy weights"
else
    print_success "Weights structure is properly configured!"
fi

print_info "For more information, see: docs/WEIGHTS_STRUCTURE.md"

