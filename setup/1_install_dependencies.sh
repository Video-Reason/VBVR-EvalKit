#!/bin/bash
##############################################################################
# Step 1: Install All Dependencies and Create Virtual Environments
#
# Creates 4 isolated venvs in envs/ directory:
# - venv_main: torch 2.5.1 (LTX, SVD, WAN, Morphic)
# - venv_hunyuan: torch 2.0.0 (HunyuanVideo)
# - venv_dynamicrafter: torch 2.0.0 (DynamiCrafter)
# - venv_videocrafter: torch 2.0.0 (VideoCrafter)
##############################################################################

set -e
cd /home/hokindeng/VMEvalKit

echo "════════════════════════════════════════════════════════════════"
echo "Step 1: Installing Dependencies"
echo "════════════════════════════════════════════════════════════════"
echo ""

mkdir -p envs

# 1. Main venv (torch 2.5.1) - LTX, SVD, WAN, Morphic
echo "🔧 [1/4] Creating envs/venv_main..."
python3 -m venv envs/venv_main
source envs/venv_main/bin/activate

echo "   Installing PyTorch 2.5.1..."
pip install -q --upgrade pip setuptools wheel
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

echo "   Installing diffusers ecosystem..."
pip install -q diffusers transformers accelerate sentencepiece
pip install -q xformers==0.0.28.post3 einops decord av omegaconf

echo "   Installing utilities..."
pip install -q opencv-python opencv-contrib-python
pip install -q numpy Pillow pandas matplotlib tqdm
pip install -q pydantic pydantic-settings python-dotenv
pip install -q requests httpx aiohttp tenacity boto3
pip install -q imageio imageio-ffmpeg moviepy cairosvg
pip install -q ftfy  # For WAN text processing

deactivate
echo "   ✅ venv_main ready (torch 2.5.1)"
echo ""

# 2. Hunyuan venv (torch 2.0.0)
echo "🔧 [2/4] Creating envs/venv_hunyuan..."
python3 -m venv envs/venv_hunyuan
source envs/venv_hunyuan/bin/activate

pip install -q --upgrade pip setuptools wheel
pip install -q torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -q diffusers==0.31.0 transformers==4.39.3 accelerate==1.1.1
pip install -q opencv-python einops imageio imageio-ffmpeg
pip install -q safetensors peft loguru
pip install -q Pillow numpy pandas tqdm pydantic python-dotenv requests

deactivate
echo "   ✅ venv_hunyuan ready (torch 2.0.0)"
echo ""

# 3. DynamiCrafter venv (torch 2.0.0)
echo "🔧 [3/4] Creating envs/venv_dynamicrafter..."
python3 -m venv envs/venv_dynamicrafter
source envs/venv_dynamicrafter/bin/activate

pip install -q --upgrade pip setuptools wheel
pip install -q torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -q decord einops imageio omegaconf opencv-python
pip install -q Pillow pytorch_lightning PyYAML tqdm transformers
pip install -q moviepy av xformers==0.0.18 gradio timm
pip install -q pandas pydantic python-dotenv requests

deactivate
echo "   ✅ venv_dynamicrafter ready (torch 2.0.0)"
echo ""

# 4. VideoCrafter venv (torch 2.0.0)
echo "🔧 [4/4] Creating envs/venv_videocrafter..."
python3 -m venv envs/venv_videocrafter
source envs/venv_videocrafter/bin/activate

pip install -q --upgrade pip setuptools wheel
pip install -q torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
pip install -q omegaconf pytorch-lightning einops transformers
pip install -q opencv-python imageio av moviepy
pip install -q Pillow numpy pandas tqdm pydantic python-dotenv requests

deactivate
echo "   ✅ venv_videocrafter ready (torch 2.0.0)"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "✅ All 4 virtual environments created!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Next: Run ./setup/2_download_checkpoints.sh"

