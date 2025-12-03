# VMEvalKit Setup Guide

## Quick Start (2 Commands)

```bash
# 1. Clone and initialize
git clone https://github.com/hokindeng/VMEvalKit.git
cd VMEvalKit
git submodule update --init --recursive

# 2. Run setup (30-60 min)
./setup/RUN_SETUP.sh
```

## Requirements

- **Python**: 3.8+
- **CUDA**: 11.8 or 12.x
- **GPU Memory**: 24GB+ recommended
- **Disk Space**: 50GB free
- **Time**: 30-60 minutes

## Architecture

### Open-Source Models (15 models)

All run locally with downloaded weights:

| Family | Models | VRAM |
|--------|--------|------|
| LTX-Video | ltx-video, ltx-video-13b-distilled | 9-18GB |
| SVD | svd | 12GB |
| WAN | wan-2.1/2.2 variants (4 models) | 12-48GB |
| HunyuanVideo | hunyuan-video-i2v | 32GB |
| DynamiCrafter | 256/512/1024 (3 models) | 12-18GB |
| VideoCrafter | videocrafter2-512 | 14GB |
| Morphic | morphic-frames-to-video | 40GB |

### Commercial Models (8 models)

Require API keys in `.env`:

```env
LUMA_API_KEY=your_key           # Luma Ray 2, Flash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json  # Veo 2, 3.0
WAVESPEED_API_KEY=your_key      # Veo 3.1, WaveSpeed WAN
RUNWAY_API_SECRET=your_key      # Runway Gen4
OPENAI_API_KEY=your_key         # OpenAI Sora
```

## Setup Scripts

### Master Script

```bash
./setup/RUN_SETUP.sh              # Full setup
./setup/RUN_SETUP.sh --yes        # Skip confirmation
./setup/RUN_SETUP.sh --skip-download  # Skip checkpoints
```

### Individual Steps

```bash
./setup/1_install_dependencies.sh  # Create 4 venvs
./setup/2_download_checkpoints.sh  # Download ~24GB
./setup/3a_validate_opensource.sh  # Validate local models
./setup/3b_validate_commercial.sh  # Validate API keys
./setup/4_test_models.sh           # Generate test videos
```

### Validation

```bash
# Validate open-source (venvs, checkpoints, imports)
./setup/3a_validate_opensource.sh

# Validate commercial (API key presence)
./setup/3b_validate_commercial.sh

# Quick status check
./setup/CHECK_STATUS.sh
```

### Testing

```bash
# Test open-source models only
./setup/4_test_models.sh --opensource

# Test commercial models only
./setup/4_test_models.sh --commercial

# Test specific model
./setup/4_test_models.sh --model ltx-video
```

## Running Inference

### Parallel (All GPUs)

```bash
./run_all_models.sh --parallel
```

### Single Model

```bash
source envs/venv_main/bin/activate
python examples/generate_videos.py --model ltx-video --all-tasks
```

### Monitor

```bash
tail -f logs/opensource_inference/*.log
watch -n 1 nvidia-smi
```

## Troubleshooting

### Reset Everything

```bash
rm -rf envs/ weights/
./setup/RUN_SETUP.sh
```

### Model-Specific Issues

Check logs:
```bash
cat logs/model_tests/MODEL_NAME.log
```

### GPU Memory

```bash
nvidia-smi                    # Check usage
pkill -f python               # Kill processes
```

## File Structure

```
setup/
├── lib/
│   └── common.sh              # Shared config (single source of truth)
├── 1_install_dependencies.sh  # Venv creation
├── 2_download_checkpoints.sh  # Weight download
├── 3a_validate_opensource.sh  # Open-source validation
├── 3b_validate_commercial.sh  # Commercial validation
├── 4_test_models.sh           # Model testing
├── RUN_SETUP.sh               # Master orchestrator
├── CHECK_STATUS.sh            # Status reporter
├── README.md                  # Quick reference
└── SETUP.md                   # Full documentation
```

All configuration (venv names, checkpoint URLs, model lists) lives in `lib/common.sh`.
