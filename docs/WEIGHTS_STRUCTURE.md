# Model Weights Structure

## Overview

All model weights in VMEvalKit are stored in a centralized `weights/` directory at the project root. This provides a clean separation between model code (in `submodules/`) and model weights.

## Directory Structure

```
weights/
├── dynamicrafter/
│   ├── dynamicrafter_256_v1/
│   │   └── model.ckpt
│   ├── dynamicrafter_512_v1/
│   │   └── model.ckpt
│   └── dynamicrafter_1024_v1/
│       └── model.ckpt
├── videocrafter/
│   └── base_512_v2/
│       └── model.ckpt
├── wan/
│   └── Wan2.2-I2V-A14B/
│       ├── configuration.json
│       ├── high_noise_model/
│       ├── low_noise_model/
│       ├── models_t5_umt5-xxl-enc-bf16.pth
│       └── Wan2.1_VAE.pth
├── morphic/
│   ├── lora_interpolation_high_noise_final.safetensors
│   └── README.md
├── hunyuan/
│   └── (HunyuanVideo-I2V weights downloaded on first run)
├── ltx-video/
│   └── (LTX-Video weights downloaded on first run)
└── svd/
    └── (Stable Video Diffusion weights downloaded on first run)
```

## Weight Management

### Automatic Download

Weights are automatically downloaded during setup:

```bash
./setup/RUN_SETUP.sh
```

This will:
1. Create the `weights/` directory structure
2. Download DynamiCrafter checkpoints (~24GB)
3. Download VideoCrafter checkpoint (~5.5GB)
4. Download Wan2.2 weights (~27GB)
5. Download Morphic LoRA weights

### Manual Download

You can also download weights manually:

```bash
# DynamiCrafter 256
wget https://huggingface.co/Doubiiu/DynamiCrafter/resolve/main/model.ckpt \
  -O weights/dynamicrafter/dynamicrafter_256_v1/model.ckpt

# DynamiCrafter 512
wget https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt \
  -O weights/dynamicrafter/dynamicrafter_512_v1/model.ckpt

# DynamiCrafter 1024
wget https://huggingface.co/Doubiiu/DynamiCrafter_1024/resolve/main/model.ckpt \
  -O weights/dynamicrafter/dynamicrafter_1024_v1/model.ckpt

# VideoCrafter2
wget https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt \
  -O weights/videocrafter/base_512_v2/model.ckpt

# Wan2.2-I2V-A14B
huggingface-cli download Wan-AI/Wan2.2-I2V-A14B \
  --local-dir weights/wan/Wan2.2-I2V-A14B

# Morphic LoRA weights
huggingface-cli download morphic/Wan2.2-frames-to-video \
  --local-dir weights/morphic
```

## Configuration

### Environment Variables

Model weight paths can be overridden via environment variables in `.env`:

```env
# Morphic Frames-to-Video
MORPHIC_WAN2_CKPT_DIR=./weights/wan/Wan2.2-I2V-A14B
MORPHIC_LORA_WEIGHTS_PATH=./weights/morphic/lora_interpolation_high_noise_final.safetensors
```

### Code References

Model inference classes automatically use the centralized weights directory:

- **DynamiCrafter**: `vmevalkit/models/dynamicrafter_inference.py`
  - Checkpoint path: `weights/dynamicrafter/{model_variant}_v1/model.ckpt`

- **VideoCrafter**: `vmevalkit/models/videocrafter_inference.py`
  - Checkpoint path: `weights/videocrafter/base_512_v2/model.ckpt`

- **Morphic**: `vmevalkit/models/morphic_inference.py`
  - Wan2.2 path: `weights/wan/Wan2.2-I2V-A14B/`
  - LoRA path: `weights/morphic/lora_interpolation_high_noise_final.safetensors`

## Git Ignore

The `weights/` directory is excluded from git tracking via `.gitignore`:

```gitignore
# Model weights - centralized storage for all downloaded models
weights/

# Legacy model weight directories (kept for backwards compatibility)
Wan2.2-I2V-A14B/
morphic-frames-lora-weights/
*.pth
*.safetensors
*.ckpt
*.bin
*.pt
```

## Disk Space Requirements

| Model Family | Size | Location |
|--------------|------|----------|
| DynamiCrafter 256 | 3.5GB | `weights/dynamicrafter/dynamicrafter_256_v1/` |
| DynamiCrafter 512 | 5.2GB | `weights/dynamicrafter/dynamicrafter_512_v1/` |
| DynamiCrafter 1024 | 9.7GB | `weights/dynamicrafter/dynamicrafter_1024_v1/` |
| VideoCrafter2 | 5.5GB | `weights/videocrafter/base_512_v2/` |
| Wan2.2-I2V-A14B | ~27GB | `weights/wan/Wan2.2-I2V-A14B/` |
| Morphic LoRA | ~500MB | `weights/morphic/` |
| HunyuanVideo | ~30GB | `weights/hunyuan/` |
| LTX-Video | ~10GB | `weights/ltx-video/` |
| SVD | ~5GB | `weights/svd/` |

**Total**: ~100GB for all models

## Migration from Legacy Structure

If you have existing weights in the old locations, you can migrate them:

```bash
# Create weights directory
mkdir -p weights/{dynamicrafter,videocrafter,wan,morphic}

# Move DynamiCrafter weights (if they exist in submodules)
if [ -d "submodules/DynamiCrafter/checkpoints" ]; then
  mv submodules/DynamiCrafter/checkpoints/* weights/dynamicrafter/
fi

# Move VideoCrafter weights (if they exist in submodules)
if [ -d "submodules/VideoCrafter/checkpoints" ]; then
  mv submodules/VideoCrafter/checkpoints/* weights/videocrafter/
fi

# Move Wan2.2 weights (if they exist in root)
if [ -d "Wan2.2-I2V-A14B" ]; then
  mv Wan2.2-I2V-A14B weights/wan/
fi

# Move Morphic LoRA weights (if they exist in root)
if [ -d "morphic-frames-lora-weights" ]; then
  mv morphic-frames-lora-weights/* weights/morphic/
  rmdir morphic-frames-lora-weights
fi
```

## Troubleshooting

### Missing Weights

If you get errors about missing weights:

1. Check if weights exist:
   ```bash
   ls -lh weights/
   ```

2. Re-download specific model:
   ```bash
   ./setup/2_download_checkpoints.sh
   ```

3. Verify paths in `.env` file match the weights directory structure

### Disk Space Issues

To free up space, remove unused model weights:

```bash
# Remove specific model
rm -rf weights/dynamicrafter/dynamicrafter_1024_v1/

# Remove all weights (will need to re-download)
rm -rf weights/
```

### Permission Issues

Ensure the weights directory is writable:

```bash
chmod -R u+w weights/
```

