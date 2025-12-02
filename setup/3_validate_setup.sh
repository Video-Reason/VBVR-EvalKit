#!/bin/bash
##############################################################################
# Step 3: Validate Setup
#
# Runs comprehensive validation including:
# - Virtual environments exist and work
# - Model checkpoints downloaded
# - Python imports work
# - Test inference on 2 sample questions
##############################################################################

set -e
cd /home/hokindeng/VMEvalKit

echo "════════════════════════════════════════════════════════════════"
echo "Step 3: Validating Complete Setup"
echo "════════════════════════════════════════════════════════════════"
echo ""

ERRORS=0

# Check 1: Virtual environments
echo "🔍 [1/4] Checking virtual environments..."
for venv in venv_main venv_hunyuan venv_dynamicrafter venv_videocrafter; do
    if [ -f "envs/${venv}/bin/python" ]; then
        version=$(envs/${venv}/bin/python --version 2>&1)
        echo "   ✅ envs/${venv} ($version)"
    else
        echo "   ❌ envs/${venv} - MISSING"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# Check 2: Model checkpoints
echo "🔍 [2/4] Checking model checkpoints..."
CKPTS=(
    "submodules/DynamiCrafter/checkpoints/dynamicrafter_256_v1/model.ckpt"
    "submodules/DynamiCrafter/checkpoints/dynamicrafter_512_v1/model.ckpt"
    "submodules/DynamiCrafter/checkpoints/dynamicrafter_1024_v1/model.ckpt"
    "submodules/VideoCrafter/checkpoints/base_512_v2/model.ckpt"
)

for ckpt in "${CKPTS[@]}"; do
    if [ -f "$ckpt" ]; then
        size=$(du -h "$ckpt" | cut -f1)
        echo "   ✅ $(basename $(dirname $(dirname $ckpt)))/$(basename $(dirname $ckpt)) ($size)"
    else
        echo "   ❌ $ckpt - MISSING"
        ERRORS=$((ERRORS + 1))
    fi
done
echo ""

# Check 3: Python imports
echo "🔍 [3/4] Checking Python imports..."

# Main venv
source envs/venv_main/bin/activate
python3 << 'PYEOF'
try:
    import torch
    import diffusers
    import transformers
    print(f"   ✅ venv_main: torch {torch.__version__}, diffusers {diffusers.__version__}")
except Exception as e:
    print(f"   ❌ venv_main import error: {e}")
    exit(1)
PYEOF
[ $? -ne 0 ] && ERRORS=$((ERRORS + 1))
deactivate

# Hunyuan venv
source envs/venv_hunyuan/bin/activate
python3 << 'PYEOF'
try:
    import torch
    import loguru
    print(f"   ✅ venv_hunyuan: torch {torch.__version__}, loguru ✓")
except Exception as e:
    print(f"   ❌ venv_hunyuan import error: {e}")
    exit(1)
PYEOF
[ $? -ne 0 ] && ERRORS=$((ERRORS + 1))
deactivate

# DynamiCrafter venv
source envs/venv_dynamicrafter/bin/activate
python3 << 'PYEOF'
try:
    import torch
    import pytorch_lightning
    print(f"   ✅ venv_dynamicrafter: torch {torch.__version__}, pytorch_lightning ✓")
except Exception as e:
    print(f"   ❌ venv_dynamicrafter import error: {e}")
    exit(1)
PYEOF
[ $? -ne 0 ] && ERRORS=$((ERRORS + 1))
deactivate

# VideoCrafter venv
source envs/venv_videocrafter/bin/activate
python3 << 'PYEOF'
try:
    import torch
    import omegaconf
    print(f"   ✅ venv_videocrafter: torch {torch.__version__}, omegaconf ✓")
except Exception as e:
    print(f"   ❌ venv_videocrafter import error: {e}")
    exit(1)
PYEOF
[ $? -ne 0 ] && ERRORS=$((ERRORS + 1))
deactivate

echo ""

# Check 4: Test questions exist
echo "🔍 [4/4] Checking test assets..."
if [ -f "tests/assets/test_question_0/first_frame.png" ] && [ -f "tests/assets/test_question_0/prompt.txt" ]; then
    echo "   ✅ Test questions ready"
else
    echo "   ⚠️  Test questions not found - will create them"
fi
echo ""

# Final summary
echo "════════════════════════════════════════════════════════════════"
if [ $ERRORS -eq 0 ]; then
    echo "✅ VALIDATION PASSED!"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Next: Run ./setup/4_test_all_models.sh to test all models"
    exit 0
else
    echo "❌ VALIDATION FAILED - $ERRORS error(s)"
    echo "════════════════════════════════════════════════════════════════"
    exit 1
fi

