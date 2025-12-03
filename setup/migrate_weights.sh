#!/bin/bash
##############################################################################
# VMEvalKit - Migrate Model Weights to Centralized Structure
##############################################################################
# This script migrates existing model weights from legacy locations to the
# new centralized weights/ directory structure.
##############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/common.sh"

print_header "VMEvalKit Weights Migration"

# Create weights directory structure
print_section "Creating Weights Directory Structure"
mkdir -p "${WEIGHTS_DIR}"/{dynamicrafter,videocrafter,wan,morphic,hunyuan,ltx-video,svd}
print_success "Weights directory structure created"

# Track what was migrated
MIGRATED_COUNT=0
TOTAL_SIZE=0

# Function to migrate directory
migrate_dir() {
    local src="$1"
    local dest="$2"
    local name="$3"
    
    if [[ -d "$src" ]] && [[ -n "$(ls -A "$src" 2>/dev/null)" ]]; then
        local size
        size=$(du -sh "$src" | cut -f1)
        print_step "Migrating ${name} (${size})..."
        
        # Create destination parent directory
        mkdir -p "$(dirname "$dest")"
        
        # Move the directory
        if mv "$src" "$dest"; then
            print_success "${name} migrated to ${dest}"
            ((MIGRATED_COUNT++))
            return 0
        else
            print_error "Failed to migrate ${name}"
            return 1
        fi
    else
        print_skip "${name} not found at ${src}"
        return 0
    fi
}

# Function to migrate file
migrate_file() {
    local src="$1"
    local dest="$2"
    local name="$3"
    
    if [[ -f "$src" ]]; then
        local size
        size=$(du -sh "$src" | cut -f1)
        print_step "Migrating ${name} (${size})..."
        
        # Create destination directory
        mkdir -p "$(dirname "$dest")"
        
        # Move the file
        if mv "$src" "$dest"; then
            print_success "${name} migrated to ${dest}"
            ((MIGRATED_COUNT++))
            return 0
        else
            print_error "Failed to migrate ${name}"
            return 1
        fi
    else
        print_skip "${name} not found at ${src}"
        return 0
    fi
}

print_section "Migrating Model Weights"

# Migrate DynamiCrafter weights from submodules
if [[ -d "${SUBMODULES_DIR}/DynamiCrafter/checkpoints" ]]; then
    for variant in dynamicrafter_256_v1 dynamicrafter_512_v1 dynamicrafter_1024_v1; do
        migrate_dir \
            "${SUBMODULES_DIR}/DynamiCrafter/checkpoints/${variant}" \
            "${WEIGHTS_DIR}/dynamicrafter/${variant}" \
            "DynamiCrafter ${variant}"
    done
fi

# Migrate VideoCrafter weights from submodules
migrate_dir \
    "${SUBMODULES_DIR}/VideoCrafter/checkpoints/base_512_v2" \
    "${WEIGHTS_DIR}/videocrafter/base_512_v2" \
    "VideoCrafter base_512_v2"

# Migrate Wan2.2 weights from root
migrate_dir \
    "${VMEVAL_ROOT}/Wan2.2-I2V-A14B" \
    "${WEIGHTS_DIR}/wan/Wan2.2-I2V-A14B" \
    "Wan2.2-I2V-A14B"

# Migrate Morphic LoRA weights from root
if [[ -d "${VMEVAL_ROOT}/morphic-frames-lora-weights" ]]; then
    print_step "Migrating Morphic LoRA weights..."
    
    # Copy all files from morphic-frames-lora-weights to weights/morphic
    if cp -r "${VMEVAL_ROOT}/morphic-frames-lora-weights/"* "${WEIGHTS_DIR}/morphic/"; then
        print_success "Morphic LoRA weights migrated"
        
        # Remove old directory
        rm -rf "${VMEVAL_ROOT}/morphic-frames-lora-weights"
        print_success "Removed old morphic-frames-lora-weights directory"
        ((MIGRATED_COUNT++))
    else
        print_error "Failed to migrate Morphic LoRA weights"
    fi
else
    print_skip "Morphic LoRA weights not found"
fi

# Migrate HunyuanVideo weights if they exist in submodules
if [[ -d "${SUBMODULES_DIR}/HunyuanVideo-I2V/ckpts" ]] && [[ -n "$(ls -A "${SUBMODULES_DIR}/HunyuanVideo-I2V/ckpts" 2>/dev/null)" ]]; then
    migrate_dir \
        "${SUBMODULES_DIR}/HunyuanVideo-I2V/ckpts" \
        "${WEIGHTS_DIR}/hunyuan/ckpts" \
        "HunyuanVideo-I2V"
fi

print_section "Migration Summary"

if [[ $MIGRATED_COUNT -eq 0 ]]; then
    print_info "No weights found to migrate"
    print_info "Weights may already be in the correct location or not yet downloaded"
else
    print_success "Successfully migrated ${MIGRATED_COUNT} model weight(s)"
    
    # Show new weights directory structure
    print_info "New weights directory structure:"
    tree -L 2 "${WEIGHTS_DIR}" 2>/dev/null || find "${WEIGHTS_DIR}" -maxdepth 2 -type d
    
    # Calculate total size
    TOTAL_SIZE=$(du -sh "${WEIGHTS_DIR}" | cut -f1)
    print_info "Total weights size: ${TOTAL_SIZE}"
fi

print_section "Next Steps"
print_info "1. Update your .env file if you have custom weight paths"
print_info "2. Run validation: ./setup/3a_validate_opensource.sh"
print_info "3. Test models: ./setup/4_test_models.sh --opensource"

print_success "Migration complete!"

