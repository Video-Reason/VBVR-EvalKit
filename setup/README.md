# VMEvalKit Setup Guide

## Quick Start

```bash
cd /home/hokindeng/VMEvalKit
./setup/RUN_SETUP.sh
```

One command, 30-60 minutes.

## What Gets Set Up

### Virtual Environments (4 venvs in `envs/`)

| Environment | PyTorch | Models |
|-------------|---------|--------|
| `venv_main` | 2.5.1 | LTX, SVD, WAN, Morphic (10 models) |
| `venv_hunyuan` | 2.0.0 | HunyuanVideo (1 model) |
| `venv_dynamicrafter` | 2.0.0 | DynamiCrafter 256/512/1024 (3 models) |
| `venv_videocrafter` | 2.0.0 | VideoCrafter2 (1 model) |

**Total: 15 open-source models**

### Model Checkpoints (~24GB in `weights/`)

All model weights are now stored in a centralized `weights/` directory:

- DynamiCrafter 256 (3.5GB) → `weights/dynamicrafter/dynamicrafter_256_v1/`
- DynamiCrafter 512 (5.2GB) → `weights/dynamicrafter/dynamicrafter_512_v1/`
- DynamiCrafter 1024 (9.7GB) → `weights/dynamicrafter/dynamicrafter_1024_v1/`
- VideoCrafter2 (5.5GB) → `weights/videocrafter/base_512_v2/`
- Wan2.2-I2V-A14B (~27GB) → `weights/wan/Wan2.2-I2V-A14B/`
- Morphic LoRA weights → `weights/morphic/`

### Commercial API Keys (Optional)

Configure in `.env` file:

| API Key | Models |
|---------|--------|
| `LUMA_API_KEY` | Luma Ray 2, Ray 2 Flash |
| `GOOGLE_APPLICATION_CREDENTIALS` | Veo 2, Veo 3.0 |
| `WAVESPEED_API_KEY` | Veo 3.1 Flash, WaveSpeed WAN |
| `RUNWAY_API_SECRET` | Runway Gen4 Turbo |
| `OPENAI_API_KEY` | OpenAI Sora |

## Script Reference

```
setup/
├── lib/
│   └── common.sh              # Shared functions and config
├── 1_install_dependencies.sh  # Create venvs (~20 min)
├── 2_download_checkpoints.sh  # Download weights (~15 min)
├── 3a_validate_opensource.sh  # Validate open-source setup
├── 3b_validate_commercial.sh  # Validate API keys
├── 4_test_models.sh           # Test model inference
├── RUN_SETUP.sh               # Master script (runs 1-3)
└── CHECK_STATUS.sh            # Quick status check
```

### Run Individual Steps

```bash
./setup/1_install_dependencies.sh    # Create venvs
./setup/2_download_checkpoints.sh    # Download weights
./setup/3a_validate_opensource.sh    # Validate open-source
./setup/3b_validate_commercial.sh    # Validate commercial
```

### Test Models

```bash
# Test open-source only
./setup/4_test_models.sh --opensource

# Test commercial only (requires API keys)
./setup/4_test_models.sh --commercial

# Test specific model
./setup/4_test_models.sh --model ltx-video

# Test all available
./setup/4_test_models.sh
```

### Check Status

```bash
./setup/CHECK_STATUS.sh
```

## Running Inference

### All Models (Parallel)

```bash
./run_all_models.sh --parallel
```

### Single Model

```bash
source envs/venv_main/bin/activate
python examples/generate_videos.py --model ltx-video --all-tasks
```

## Troubleshooting

### Clean Restart

```bash
rm -rf envs/
rm -rf weights/
./setup/RUN_SETUP.sh
```

### Skip Steps

```bash
./setup/RUN_SETUP.sh --skip-download    # Skip checkpoint download
./setup/RUN_SETUP.sh --skip-validate    # Skip validation
./setup/RUN_SETUP.sh --yes              # Skip confirmation
```

### GPU Issues

```bash
nvidia-smi                    # Check GPU status
pkill -f python               # Kill stuck processes
```

## Directory Structure

```
VMEvalKit/
├── envs/                     # Virtual environments (~11GB)
│   ├── venv_main/
│   ├── venv_hunyuan/
│   ├── venv_dynamicrafter/
│   └── venv_videocrafter/
├── weights/                  # Centralized model weights (~50GB)
│   ├── dynamicrafter/
│   │   ├── dynamicrafter_256_v1/
│   │   ├── dynamicrafter_512_v1/
│   │   └── dynamicrafter_1024_v1/
│   ├── videocrafter/
│   │   └── base_512_v2/
│   ├── wan/
│   │   └── Wan2.2-I2V-A14B/
│   ├── morphic/
│   ├── hunyuan/
│   ├── ltx-video/
│   └── svd/
├── submodules/               # Model source code (no weights)
│   ├── DynamiCrafter/
│   ├── VideoCrafter/
│   ├── HunyuanVideo-I2V/
│   ├── LTX-Video/
│   └── morphic-frames-to-video/
├── tests/
│   ├── assets/               # Test questions
│   └── outputs/              # Test results
└── setup/                    # Setup scripts
```

**Total disk usage: ~60GB** (11GB venvs + 50GB weights)
