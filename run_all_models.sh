#!/bin/bash
##############################################################################
# VMEvalKit - Unified Open-Source Models Inference Runner
# 
# This script runs all open-source models on the complete dataset
# - 1,643 questions across 11 task types
# - 16 open-source models (excludes wan-2.1-vace-1.3b - doesn't exist)
# - Total: ~26,288 video generations (1,643 × 16)
#
# Hardware: 8x NVIDIA H200 (140GB VRAM each)
# Resume: Automatically skips completed tasks
#
# Usage:
#   ./run_all_models.sh              # Sequential (one at a time)
#   ./run_all_models.sh --parallel   # Parallel (use all GPUs)
##############################################################################

# Configuration
OUTPUT_DIR="data/outputs/pilot_experiment"
LOG_DIR="logs/opensource_inference"
mkdir -p "$LOG_DIR"

# Working models (wan-2.1-vace-1.3b removed - doesn't exist on HF)
MODELS=(
    "ltx-video"
    "ltx-video-13b-distilled"
    "svd"
    "hunyuan-video-i2v"
    "videocrafter2-512"
    "dynamicrafter-256"
    "dynamicrafter-512"
    "dynamicrafter-1024"
    "wan"
    "wan-2.1-flf2v-720p"
    "wan-2.2-i2v-a14b"
    "wan-2.1-i2v-480p"
    "wan-2.1-i2v-720p"
    "wan-2.2-ti2v-5b"
    "wan-2.1-vace-14b"
    "morphic-frames-to-video"
)

# Parse arguments
PARALLEL=false
if [ "$1" = "--parallel" ]; then
    PARALLEL=true
fi

echo "════════════════════════════════════════════════════════════════"
echo "         VMEvalKit Open-Source Models Inference Runner"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Configuration:"
echo "  • Models: ${#MODELS[@]}"
echo "  • Questions: 1,643"
echo "  • Total: ~26,288 generations"
if [ "$PARALLEL" = true ]; then
    echo "  • Mode: PARALLEL (all GPUs)"
else
    echo "  • Mode: SEQUENTIAL (one model at a time)"
fi
echo "  • Output: ${OUTPUT_DIR}"
echo ""

# Function to get correct venv for a model
get_venv_for_model() {
    local MODEL=$1
    case "$MODEL" in
        hunyuan-video-i2v)
            echo "envs/venv_hunyuan"
            ;;
        dynamicrafter-*)
            echo "envs/venv_dynamicrafter"
            ;;
        videocrafter*)
            echo "envs/venv_videocrafter"
            ;;
        *)
            echo "envs/venv_main"
            ;;
    esac
}

# Activate venv
cd /home/hokindeng/VMEvalKit

if [ "$PARALLEL" = true ]; then
    echo "🚀 Starting PARALLEL execution..."
    echo "  Wave 1: 5 lightweight models on GPUs 0-4"
    echo "  Wave 2: 4 medium models on GPUs 0-3"
    echo "  Wave 3: 3 heavy models on GPUs 0-2"
    echo "  Wave 4: 4 very heavy models - 2 at a time"
    echo ""
    
    # Wave 1: Lightweight (5 models)
    echo "🌊 Wave 1: Lightweight models..."
    (source envs/venv_main/bin/activate && CUDA_VISIBLE_DEVICES=0 python examples/generate_videos.py --model ltx-video --all-tasks 2>&1 | tee "${LOG_DIR}/ltx-video.log" | sed "s/^/[GPU0-ltx-video] /") &
    (source envs/venv_main/bin/activate && CUDA_VISIBLE_DEVICES=1 python examples/generate_videos.py --model ltx-video-13b-distilled --all-tasks 2>&1 | tee "${LOG_DIR}/ltx-video-13b.log" | sed "s/^/[GPU1-ltx-13b] /") &
    (source envs/venv_videocrafter/bin/activate && CUDA_VISIBLE_DEVICES=2 python examples/generate_videos.py --model videocrafter2-512 --all-tasks 2>&1 | tee "${LOG_DIR}/videocrafter.log" | sed "s/^/[GPU2-videocrafter] /") &
    (source envs/venv_dynamicrafter/bin/activate && CUDA_VISIBLE_DEVICES=3 python examples/generate_videos.py --model dynamicrafter-256 --all-tasks 2>&1 | tee "${LOG_DIR}/dynamicrafter-256.log" | sed "s/^/[GPU3-dynamicrafter-256] /") &
    (source envs/venv_main/bin/activate && CUDA_VISIBLE_DEVICES=4 python examples/generate_videos.py --model svd --all-tasks 2>&1 | tee "${LOG_DIR}/svd.log" | sed "s/^/[GPU4-svd] /") &
    wait
    echo "✅ Wave 1 done!"
    
    # Wave 2: Medium (4 models)  
    echo "🌊 Wave 2: Medium models..."
    (source envs/venv_dynamicrafter/bin/activate && CUDA_VISIBLE_DEVICES=0 python examples/generate_videos.py --model dynamicrafter-512 --all-tasks 2>&1 | tee "${LOG_DIR}/dynamicrafter-512.log" | sed "s/^/[GPU0-dynamicrafter-512] /") &
    (source envs/venv_dynamicrafter/bin/activate && CUDA_VISIBLE_DEVICES=1 python examples/generate_videos.py --model dynamicrafter-1024 --all-tasks 2>&1 | tee "${LOG_DIR}/dynamicrafter-1024.log" | sed "s/^/[GPU1-dynamicrafter-1024] /") &
    (source envs/venv_hunyuan/bin/activate && CUDA_VISIBLE_DEVICES=2 python examples/generate_videos.py --model hunyuan-video-i2v --all-tasks 2>&1 | tee "${LOG_DIR}/hunyuan.log" | sed "s/^/[GPU2-hunyuan] /") &
    (source envs/venv_main/bin/activate && CUDA_VISIBLE_DEVICES=3 python examples/generate_videos.py --model wan-2.1-i2v-480p --all-tasks 2>&1 | tee "${LOG_DIR}/wan-480p.log" | sed "s/^/[GPU3-wan-480p] /") &
    wait
    echo "✅ Wave 2 done!"
    
    # Wave 3: Heavy (3 models) - All WAN models use main venv
    echo "🌊 Wave 3: Heavy models..."
    (source envs/venv_main/bin/activate && CUDA_VISIBLE_DEVICES=0 python examples/generate_videos.py --model wan-2.2-ti2v-5b --all-tasks 2>&1 | tee "${LOG_DIR}/wan-ti2v.log" | sed "s/^/[GPU0-wan-ti2v] /") &
    (source envs/venv_main/bin/activate && CUDA_VISIBLE_DEVICES=1 python examples/generate_videos.py --model wan-2.1-i2v-720p --all-tasks 2>&1 | tee "${LOG_DIR}/wan-720p.log" | sed "s/^/[GPU1-wan-720p] /") &
    wait
    echo "✅ Wave 3 done!"
    
    # Wave 4: Very heavy (4 models, 2 at a time)
    echo "🌊 Wave 4: Very heavy models (48GB each, 2 at a time)..."
    (source envs/venv_main/bin/activate && CUDA_VISIBLE_DEVICES=0 python examples/generate_videos.py --model wan --all-tasks 2>&1 | tee "${LOG_DIR}/wan.log" | sed "s/^/[GPU0-wan] /") &
    (source envs/venv_main/bin/activate && CUDA_VISIBLE_DEVICES=1 python examples/generate_videos.py --model wan-2.1-flf2v-720p --all-tasks 2>&1 | tee "${LOG_DIR}/wan-flf2v.log" | sed "s/^/[GPU1-wan-flf2v] /") &
    wait
    
    (source envs/venv_main/bin/activate && CUDA_VISIBLE_DEVICES=0 python examples/generate_videos.py --model wan-2.2-i2v-a14b --all-tasks 2>&1 | tee "${LOG_DIR}/wan-a14b.log" | sed "s/^/[GPU0-wan-a14b] /") &
    (source envs/venv_main/bin/activate && CUDA_VISIBLE_DEVICES=1 python examples/generate_videos.py --model wan-2.1-vace-14b --all-tasks 2>&1 | tee "${LOG_DIR}/wan-vace.log" | sed "s/^/[GPU1-wan-vace] /") &
    wait
    echo "✅ Wave 4 done!"
    
    # Wave 5: Morphic (all GPUs) - uses main venv
    echo "🌊 Wave 5: Morphic (distributed across all 8 GPUs)..."
    (source envs/venv_main/bin/activate && python examples/generate_videos.py --model morphic-frames-to-video --all-tasks 2>&1 | tee "${LOG_DIR}/morphic.log" | sed "s/^/[ALL-GPUs-morphic] /")
    echo "✅ Wave 5 done!"
    
else
    echo "🚀 Starting SEQUENTIAL execution..."
    echo ""
    
    COMPLETED=0
    FAILED=0
    
    for i in "${!MODELS[@]}"; do
        MODEL="${MODELS[$i]}"
        NUM=$((i + 1))
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "Model ${NUM}/${#MODELS[@]}: ${MODEL}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Assign GPU (rotate)
        GPU=$((i % 8))
        
        # Get correct venv
        VENV=$(get_venv_for_model "$MODEL")
        echo "Using: $VENV"
        
        # Run with appropriate venv
        source ${VENV}/bin/activate
        CUDA_VISIBLE_DEVICES=${GPU} python examples/generate_videos.py \
            --model "${MODEL}" \
            --all-tasks \
            2>&1 | tee "${LOG_DIR}/${MODEL}_gpu${GPU}.log"
        
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            COMPLETED=$((COMPLETED + 1))
            echo "✅ Completed: ${MODEL}"
        else
            FAILED=$((FAILED + 1))
            echo "❌ Failed: ${MODEL}"
        fi
        
        echo "Progress: ${COMPLETED} done, ${FAILED} failed, $((${#MODELS[@]} - NUM)) remaining"
    done
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "                    ✅ ALL DONE!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Outputs: ${OUTPUT_DIR}"
echo "Logs: ${LOG_DIR}"

