#!/bin/bash
##############################################################################
# Step 2: Download Model Checkpoints for Submodule-Based Models
#
# Downloads checkpoints for models that require local weights:
# - DynamiCrafter (256/512/1024) - ~18.4GB total
# - VideoCrafter2 - ~5.5GB
#
# Note: Diffusers models (LTX, SVD, WAN, Hunyuan) download automatically on first use
##############################################################################

set -e
cd /home/hokindeng/VMEvalKit

echo "════════════════════════════════════════════════════════════════"
echo "Step 2: Downloading Model Checkpoints"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Total download size: ~24GB"
echo "This will take 10-30 minutes depending on your network speed"
echo ""

# Create checkpoint directories
mkdir -p submodules/DynamiCrafter/checkpoints/dynamicrafter_256_v1
mkdir -p submodules/DynamiCrafter/checkpoints/dynamicrafter_512_v1
mkdir -p submodules/DynamiCrafter/checkpoints/dynamicrafter_1024_v1
mkdir -p submodules/VideoCrafter/checkpoints/base_512_v2

echo "📥 [1/4] DynamiCrafter 256 (256x256) - 3.5GB..."
if [ ! -f "submodules/DynamiCrafter/checkpoints/dynamicrafter_256_v1/model.ckpt" ]; then
    wget -q --show-progress -c https://huggingface.co/Doubiiu/DynamiCrafter/resolve/main/model.ckpt \
        -O submodules/DynamiCrafter/checkpoints/dynamicrafter_256_v1/model.ckpt
    echo "   ✅ Downloaded"
else
    echo "   ⏭️  Already exists"
fi
echo ""

echo "📥 [2/4] DynamiCrafter 512 (320x512) - 5.2GB..."
if [ ! -f "submodules/DynamiCrafter/checkpoints/dynamicrafter_512_v1/model.ckpt" ]; then
    wget -q --show-progress -c https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt \
        -O submodules/DynamiCrafter/checkpoints/dynamicrafter_512_v1/model.ckpt
    echo "   ✅ Downloaded"
else
    echo "   ⏭️  Already exists"
fi
echo ""

echo "📥 [3/4] DynamiCrafter 1024 (576x1024) - 9.7GB..."
if [ ! -f "submodules/DynamiCrafter/checkpoints/dynamicrafter_1024_v1/model.ckpt" ]; then
    wget -q --show-progress -c https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt \
        -O submodules/DynamiCrafter/checkpoints/dynamicrafter_1024_v1/model.ckpt
    echo "   ✅ Downloaded"
else
    echo "   ⏭️  Already exists"
fi
echo ""

echo "📥 [4/4] VideoCrafter2 (320x512) - 5.5GB..."
if [ ! -f "submodules/VideoCrafter/checkpoints/base_512_v2/model.ckpt" ]; then
    wget -q --show-progress -c https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt \
        -O submodules/VideoCrafter/checkpoints/base_512_v2/model.ckpt
    echo "   ✅ Downloaded"
else
    echo "   ⏭️  Already exists"
fi
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "✅ All checkpoints downloaded!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next: Run ./setup/3_validate_setup.sh"

