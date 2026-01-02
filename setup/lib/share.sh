#!/bin/bash
##############################################################################
# VMEvalKit Setup - Shared Library
##############################################################################

set -euo pipefail

# Project root - dynamically determine from script location
SHARE_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VMEVAL_ROOT="$(cd "${SHARE_LIB_DIR}/../.." && pwd)"
export ENVS_DIR="${VMEVAL_ROOT}/envs"
export SUBMODULES_DIR="${VMEVAL_ROOT}/submodules"
export LOGS_DIR="${VMEVAL_ROOT}/logs"
export TESTS_DIR="${VMEVAL_ROOT}/setup/test_assets/test_task"
export WEIGHTS_DIR="${VMEVAL_ROOT}/weights"

# ============================================================================
# MODEL REGISTRY
# ============================================================================

declare -a OPENSOURCE_MODELS=(
    "ltx-video"
    "ltx-video-13b-distilled"
    "svd"
    "morphic-frames-to-video"
    "hunyuan-video-i2v"
    "dynamicrafter-256"
    "dynamicrafter-512"
    "dynamicrafter-1024"
    "videocrafter2-512"
    "cogvideox-5b-i2v"
    "cogvideox1.5-5b-i2v"
    "sana-video-2b-480p"
    "wan-2.2-i2v-a14b"
)

declare -a COMMERCIAL_MODELS=(
    "luma-ray-2"
    "luma-ray-flash-2"
    "veo-2"
    "veo-3.0-generate"
    "veo-3.1-fast"
    "wavespeed-wan-2.1-i2v-480p"
    "runway-gen4-turbo"
    "openai-sora-2"
)

declare -A COMMERCIAL_API_KEYS=(
    ["luma-ray-2"]="LUMA_API_KEY"
    ["luma-ray-flash-2"]="LUMA_API_KEY"
    ["veo-2"]="GEMINI_API_KEY"
    ["veo-3.0-generate"]="GEMINI_API_KEY"
    ["veo-3.1-fast"]="WAVESPEED_API_KEY"
    ["wavespeed-wan-2.1-i2v-480p"]="WAVESPEED_API_KEY"
    ["runway-gen4-turbo"]="RUNWAYML_API_SECRET"
    ["openai-sora-2"]="OPENAI_API_KEY"
)

declare -a CHECKPOINTS=(
    "dynamicrafter/dynamicrafter_256_v1/model.ckpt|https://huggingface.co/Doubiiu/DynamiCrafter/resolve/main/model.ckpt|3.5GB"
    "dynamicrafter/dynamicrafter_512_v1/model.ckpt|https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt|5.2GB"
    "dynamicrafter/dynamicrafter_1024_v1/model.ckpt|https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt|9.7GB"
    "videocrafter/base_512_v2/model.ckpt|https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt|5.5GB"
)

declare -A MODEL_CHECKPOINT_PATHS=(
    ["dynamicrafter-256"]="dynamicrafter/dynamicrafter_256_v1/model.ckpt"
    ["dynamicrafter-512"]="dynamicrafter/dynamicrafter_512_v1/model.ckpt"
    ["dynamicrafter-1024"]="dynamicrafter/dynamicrafter_1024_v1/model.ckpt"
    ["videocrafter2-512"]="videocrafter/base_512_v2/model.ckpt"
)

# ============================================================================
# MODEL HELPERS
# ============================================================================

is_opensource_model() {
    local target="$1"
    for model in "${OPENSOURCE_MODELS[@]}"; do
        [[ "$model" == "$target" ]] && return 0
    done
    return 1
}

is_commercial_model() {
    local target="$1"
    for model in "${COMMERCIAL_MODELS[@]}"; do
        [[ "$model" == "$target" ]] && return 0
    done
    return 1
}

get_commercial_env_var() {
    echo "${COMMERCIAL_API_KEYS[$1]:-}"
}

# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

print_header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "$1"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
}

print_section() {
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "$1"
    echo "────────────────────────────────────────────────────────────────"
}

print_success() { echo "   ✅ $1"; }
print_error()   { echo "   ❌ $1"; }
print_warning() { echo "   ⚠️  $1"; }
print_skip()    { echo "   ⏭️  $1"; }
print_info()    { echo "   📌 $1"; }
print_step()    { echo "🔧 $1"; }
print_download(){ echo "📥 $1"; }

# ============================================================================
# VENV FUNCTIONS
# ============================================================================

get_model_venv_path() {
    echo "${ENVS_DIR}/$1"
}

model_venv_exists() {
    [[ -f "$(get_model_venv_path "$1")/bin/python" ]]
}

activate_model_venv() {
    source "$(get_model_venv_path "$1")/bin/activate"
}

create_model_venv() {
    local model="$1"
    local venv_path
    venv_path="$(get_model_venv_path "$model")"
    
    # Always start fresh - remove existing environment if present
    if [[ -d "$venv_path" ]]; then
        print_step "Removing existing environment: ${model}"
        rm -rf "$venv_path"
        print_success "Old environment removed"
    fi
    
    print_step "Creating virtual environment: ${model}"
    mkdir -p "${ENVS_DIR}"
    python3 -m venv "$venv_path"
    
    source "${venv_path}/bin/activate"
    pip install -q --upgrade pip setuptools wheel
    deactivate
    
    print_success "Virtual environment created: ${model}"
}

# ============================================================================
# CHECKPOINT FUNCTIONS
# ============================================================================

download_checkpoint_asset() {
    local rel_path="$1"
    local url="$2"
    local size_desc="${3:-}"
    local full_path="${WEIGHTS_DIR}/${rel_path}"
    local dir_path
    dir_path="$(dirname "$full_path")"

    if [[ -f "$full_path" ]]; then
        print_skip "Checkpoint exists: $(basename "$rel_path")"
        return 0
    fi

    print_download "Downloading $(basename "$(dirname "$rel_path")") ${size_desc:+- ${size_desc}}"
    mkdir -p "$dir_path"
    wget -q --show-progress -c "$url" -O "$full_path"
    print_success "Checkpoint ready"
}

download_checkpoint_by_path() {
    local rel_path="$1"
    for entry in "${CHECKPOINTS[@]}"; do
        IFS='|' read -r path url size_desc <<< "$entry"
        if [[ "$path" == "$rel_path" ]]; then
            download_checkpoint_asset "$path" "$url" "$size_desc"
            return 0
        fi
    done
    print_warning "Unknown checkpoint: ${rel_path}"
    return 1
}

ensure_morphic_assets() {
    local wan_dir="${WEIGHTS_DIR}/wan/Wan2.2-I2V-A14B"
    local lora_dir="${WEIGHTS_DIR}/morphic"

    if [[ -d "$wan_dir" ]] && [[ -n "$(ls -A "$wan_dir" 2>/dev/null)" ]]; then
        print_skip "Wan2.2-I2V-A14B weights exist"
    else
        print_download "Wan2.2-I2V-A14B (~27GB)..."
        mkdir -p "$(dirname "$wan_dir")"
        huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir "$wan_dir"
        print_success "Wan2.2-I2V-A14B ready"
    fi

    if [[ -f "$lora_dir/lora_interpolation_high_noise_final.safetensors" ]]; then
        print_skip "Morphic LoRA weights exist"
    else
        print_download "Morphic LoRA weights..."
        mkdir -p "$lora_dir"
        huggingface-cli download morphic/Wan2.2-frames-to-video --local-dir "$lora_dir"
        print_success "Morphic LoRA ready"
    fi
}

# ============================================================================
# COMMERCIAL API FUNCTIONS
# ============================================================================

check_api_key() {
    local value="${!1:-}"
    [[ -n "$value" ]]
}

load_env_file() {
    local env_file="${VMEVAL_ROOT}/.env"
    if [[ -f "$env_file" ]]; then
        set -a
        source "$env_file"
        set +a
    fi
}

# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

validate_model() {
    local model="$1"
    local test_output="${VMEVAL_ROOT}/test_outputs"
    
    # Set timeout based on model complexity
    # High-quality distributed models need more time
    local timeout_seconds=10800  # Default: 3 hours
    
    print_step "Validating ${model}... (timeout: ${timeout_seconds}s)"
    echo ""
    
    activate_model_venv "$model"
    set +e
    timeout "$timeout_seconds" python "${VMEVAL_ROOT}/examples/generate_videos.py" \
        --questions-dir "${TESTS_DIR}" \
        --output-dir "$test_output" \
        --model "$model" \
        --task-id test_0001 test_0002
    local exit_code=$?
    set -e
    deactivate
    
    local video_count
    local validation_output="${test_output}/${model}/test_task"
    local model_output_dir="${test_output}/${model}"
    video_count=$(find "$validation_output" \( -name "*.mp4" -o -name "*.webm" \) 2>/dev/null | wc -l)
    
    # Clean up all task directories except tests_task
    if [[ -d "$model_output_dir" ]]; then
        find "$model_output_dir" -mindepth 1 -maxdepth 1 -type d ! -name 'test_task' -exec rm -rf {} +
    fi
    
    if [[ $exit_code -eq 0 ]] && [[ $video_count -ge 2 ]]; then
        print_success "${model}: ${video_count} videos generated ✓"
        return 0
    elif [[ $exit_code -eq 124 ]]; then
        print_warning "${model}: TIMEOUT (>${timeout_seconds}s)"
        return 1
    else
        print_error "${model}: FAILED - see output above"
        return 1
    fi
}

# ============================================================================
# SYSTEM DEPENDENCIES
# ============================================================================

ensure_ffmpeg_dependencies() {
    print_step "Checking FFmpeg system dependencies..."
    
    # Check if FFmpeg libraries are installed
    if pkg-config --exists libavformat libavcodec libavdevice libavutil libavfilter libswscale libswresample 2>/dev/null; then
        print_success "FFmpeg libraries already installed"
        return 0
    fi
    
    print_warning "FFmpeg development libraries not found"
    print_step "Installing FFmpeg dependencies (requires sudo)..."
    
    sudo apt update
    sudo apt install -y ffmpeg libavcodec-dev libavformat-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev pkg-config
    
    print_success "FFmpeg dependencies installed"
}

cd "${VMEVAL_ROOT}"
