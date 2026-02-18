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
    "LTX-2"
    "svd"
    "morphic-frames-to-video"
    "hunyuan-video-i2v"
    "cogvideox-5b-i2v"
    "cogvideox1.5-5b-i2v"
    "sana-video-2b-480p"
    "sana"
    "wan-2.1-i2v-480p"
    "wan-2.1-i2v-720p"
    "wan-2.2-i2v-a14b"
    "wan-2.2-ti2v-5b"
)

declare -a COMMERCIAL_MODELS=(
    "luma-ray-2"
    "luma-ray-flash-2"
    "veo-2"
    "veo-2.0-generate"
    "veo-3.0-fast-generate"
    "veo-3.0-generate"
    "veo-3.1-generate"
    "veo-3.1-fast"
    "kling-v2-6"
    "kling-v2-5-turbo"
    "kling-v2-1-master"
    "kling-v2-master"
    "kling-v1-6"
    "runway-gen45"
    "runway-gen3a-turbo"
    "runway-gen4-aleph"
    "runway-gen4-turbo"
    "openai-sora-2"
    "openai-sora-2-pro"
)

# Commercial API keys lookup (bash 3.2 compatible - no associative arrays)
_get_api_key_for_model() {
    case "$1" in
        luma-ray-2|luma-ray-flash-2) echo "LUMA_API_KEY" ;;
        veo-2|veo-2.0-generate|veo-3.0-fast-generate|veo-3.0-generate|veo-3.1-generate|veo-3.1-fast) echo "GEMINI_API_KEY" ;;
        kling-v2-6|kling-v2-5-turbo|kling-v2-1-master|kling-v2-master|kling-v1-6) echo "KLING_API_KEY" ;;
        runway-gen45|runway-gen3a-turbo|runway-gen4-aleph|runway-gen4-turbo) echo "RUNWAYML_API_SECRET" ;;
        openai-sora-2|openai-sora-2-pro) echo "OPENAI_API_KEY" ;;
        *) echo "" ;;
    esac
}

declare -a CHECKPOINTS=(
)

# Model checkpoint paths lookup (bash 3.2 compatible - no associative arrays)
get_model_checkpoint_path() {
    case "$1" in
        *) echo "" ;;
    esac
}

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
    _get_api_key_for_model "$1"
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

# --- Conda-based environment helpers ---

create_model_conda_env() {
    local model="$1"
    local python_version="${2:-3.10}"
    local env_path
    env_path="$(get_model_venv_path "$model")"

    if [[ -d "$env_path" ]]; then
        print_step "Removing existing environment: ${model}"
        rm -rf "$env_path"
        print_success "Old environment removed"
    fi

    print_step "Creating conda environment: ${model} (Python ${python_version})"
    mkdir -p "${ENVS_DIR}"
    conda create -y -p "$env_path" python="${python_version}" pip setuptools wheel -q
    print_success "Conda environment created: ${model}"
}

activate_model_conda_env() {
    # Ensure conda shell hooks are available (needed in non-interactive scripts)
    eval "$(conda shell.bash hook)"
    conda activate "$(get_model_venv_path "$1")"
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
        hf download Wan-AI/Wan2.2-I2V-A14B --local-dir "$wan_dir"
        print_success "Wan2.2-I2V-A14B ready"
    fi

    if [[ -f "$lora_dir/lora_interpolation_high_noise_final.safetensors" ]]; then
        print_skip "Morphic LoRA weights exist"
    else
        print_download "Morphic LoRA weights..."
        mkdir -p "$lora_dir"
        hf download morphic/Wan2.2-frames-to-video --local-dir "$lora_dir"
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
