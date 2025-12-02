# VMEvalKit Setup Guide 🚀

## Quick Start (3 Commands)

```bash
# 1. Clone and initialize
git clone https://github.com/hokindeng/VMEvalKit.git
cd VMEvalKit
git submodule update --init --recursive

# 2. Run ONE-CLICK setup (30-60 min)
./setup_complete.sh

# 3. Run all models in parallel
./run_all_models.sh --parallel
```

## What `setup_complete.sh` Does

This **ONE script** handles everything:

### 1️⃣ Virtual Environments (4 venvs)
- **`envs/venv_main`** - PyTorch 2.5.1
  - ltx-video, ltx-video-13b-distilled
  - svd, morphic-frames-to-video
  - WAN 2.1/2.2 models (8 variants)
  
- **`envs/venv_hunyuan`** - PyTorch 2.0.0
  - hunyuan-video-i2v
  
- **`envs/venv_dynamicrafter`** - PyTorch 2.0.0
  - dynamicrafter-256/512/1024
  
- **`envs/venv_videocrafter`** - PyTorch 2.0.0
  - videocrafter2-512

### 2️⃣ Model Checkpoints (~24GB)
- DynamiCrafter 256 (3.5GB)
- DynamiCrafter 512 (5.2GB)
- DynamiCrafter 1024 (9.7GB)
- VideoCrafter2 (5.5GB)

### 3️⃣ Validation
- Checks all venvs exist
- Verifies all checkpoints downloaded
- Reports any missing components

## Requirements

- **Python**: 3.8+
- **CUDA**: 11.8 or 12.x
- **GPU Memory**: 24GB+ recommended (works on H200/A100/RTX 4090)
- **Disk Space**: 50GB free
- **Time**: 30-60 minutes (mostly download time)

## After Setup

### Run All Models (Parallel - Uses All GPUs)
```bash
./run_all_models.sh --parallel
```

**Execution Plan:**
- Wave 1: 5 lightweight models on GPUs 0-4
- Wave 2: 4 medium models on GPUs 0-3
- Wave 3: 2 heavy models on GPUs 0-1
- Wave 4: 4 very heavy models (2 at a time)
- Wave 5: Morphic (uses all 8 GPUs)

### Run Single Model
```bash
source envs/venv_main/bin/activate
python examples/generate_videos.py --model ltx-video --all-tasks
```

### Monitor Progress
```bash
# Watch logs in real-time
tail -f logs/opensource_inference/*.log

# Check GPU usage
watch -n 1 nvidia-smi
```

## Troubleshooting

### Setup Failed?
```bash
# Re-run setup (it will skip existing downloads)
./setup_complete.sh
```

### Clean Start?
```bash
# Remove everything and start fresh
rm -rf envs/
rm -rf submodules/*/checkpoints/
./setup_complete.sh
```

### Only One GPU in Use?
Check that you ran with `--parallel`:
```bash
./run_all_models.sh --parallel  # ✅ Correct
./run_all_models.sh             # ❌ Sequential (one GPU at a time)
```

## Model Overview

| Model | Resolution | VRAM | Speed | Quality |
|-------|-----------|------|-------|---------|
| ltx-video-13b | 768x512 | 18GB | Fast | High |
| ltx-video | 768x512 | 9GB | Very Fast | High |
| svd | 576x1024 | 12GB | Fast | Good |
| dynamicrafter-1024 | 576x1024 | 18GB | Medium | Good |
| dynamicrafter-512 | 320x512 | 13GB | Fast | Good |
| dynamicrafter-256 | 256x256 | 12GB | Fast | Medium |
| videocrafter2 | 320x512 | 14GB | Medium | Good |
| hunyuan-video-i2v | 544x960 | 32GB | Slow | Very High |
| WAN variants | Various | 12-48GB | Fast-Medium | High |
| morphic | 512x512 | 40GB | Medium | Good |

**Total: 16 open-source models ready to run!**

## Next Steps

1. ✅ Setup complete? Run: `./run_all_models.sh --parallel`
2. 📊 View results in: `data/outputs/pilot_experiment/`
3. 📝 Check logs in: `logs/opensource_inference/`

## Support

- 📖 Full docs: [docs/](docs/)
- 🐛 Issues: [GitHub Issues](https://github.com/hokindeng/VMEvalKit/issues)
- 💬 Community: [Slack](https://join.slack.com/t/growingailikeachild/shared_invite/zt-309yqd0sl-W8xzOkdBPha1Jh5rnee78A)

