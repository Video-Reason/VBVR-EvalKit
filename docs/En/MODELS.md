
# Supported Models

VMEvalKit provides unified access to **33 video generation models** across **13 provider families**.

## Commercial APIs (19 models)

### Luma Dream Machine (2 models)
**API Key:** `LUMA_API_KEY`
- `luma-ray-2` - Latest model with best quality
- `luma-ray-flash-2` - Faster generation

### Google Veo (6 models)
**API Key:** `GEMINI_API_KEY`
- `veo-2` - GA model for text+image→video
- `veo-2.0-generate` - GA model for text+image→video
- `veo-3.0-generate` - Advanced video generation model
- `veo-3.0-fast-generate` - Faster generation model
- `veo-3.1-generate` - Latest model with native 1080p and audio (preview)
- `veo-3.1-fast` - Faster variant of Veo 3.1 (preview)


### Kling AI (5 models)
**API Key:** `KLING_API_KEY`
- `kling-v2-6` - Latest Kling model with best quality
- `kling-v2-5-turbo` - Fast generation model
- `kling-v2-1-master` - High quality model
- `kling-v2-master` - Balanced quality and speed
- `kling-v1-6` - Improved original model

### Runway ML (4 models)
**API Key:** `RUNWAYML_API_SECRET`
- `runway-gen45` - World's top-rated video model (5s or 10s)
- `runway-gen4-turbo` - Fast high-quality generation (5s or 10s)
- `runway-gen4-aleph` - Premium quality (5s)
- `runway-gen3a-turbo` - Proven performance (5s or 10s)

### OpenAI Sora (2 models)
**API Key:** `OPENAI_API_KEY`
- `openai-sora-2` - High-quality video generation (4s/8s/12s)
- `openai-sora-2-pro` - Enhanced model with more resolution options

## Open-Source Models (14 models)

### LTX-Video (3 models)
**VRAM:** 16-40GB | **Setup:** `bash setup/install_model.sh ltx-video`
- `ltx-video` - High-quality image-to-video generation (704x480, 24fps)
- `ltx-video-13b-distilled` - Distilled version with 13B parameters
- `LTX-2` - 19B FP8 text/image-to-video with audio generation (~40GB VRAM)

### HunyuanVideo (1 model)
**VRAM:** 24GB+ | **Setup:** `bash setup/install_model.sh hunyuan-video-i2v`
- `hunyuan-video-i2v` - High-quality image-to-video up to 720p

### Morphic (1 model)
**VRAM:** 20GB+ | **Setup:** `bash setup/install_model.sh morphic-frames-to-video`
- `morphic-frames-to-video` - High-quality interpolation using Wan2.2

### Stable Video Diffusion (1 model)
**VRAM:** 20GB | **Setup:** `bash setup/install_model.sh svd`
- `svd` - High-quality image-to-video generation

### WAN (Wan-AI) (4 models)
**VRAM:** 48GB+ | **Setup:** `bash setup/install_model.sh wan-2.2-ti2v-5b`
- `wan-2.1-i2v-480p` - Image to Video generation at 480p resolution
- `wan-2.1-i2v-720p` - Image to Video generation at 720p resolution
- `wan-2.2-i2v-a14b` - Image to Video generation with 14B parameters
- `wan-2.2-ti2v-5b` - Text + Image to Video generation with 5B parameters

### CogVideoX (2 models)
**VRAM:** 20GB+ | **Setup:** `bash setup/install_model.sh cogvideox-5b-i2v`
- `cogvideox-5b-i2v` - 6s image+text to video (720x480)
- `cogvideox1.5-5b-i2v` - 10s image+text to video (1360x768)

### SANA-Video (1 model)
**VRAM:** 16GB+ | **Setup:** `bash setup/install_model.sh sana-video-2b-480p`
- `sana-video-2b-480p` - Efficient text+image to video (480x832)

### Sana (1 model)
**VRAM:** 16GB+ | **Setup:** `bash setup/install_model.sh sana`
- `sana` - Image-to-Video generation with motion control


## Usage

### List Available Models
```bash
python examples/generate_videos.py --list-models
```

### Quick Start Examples

#### Commercial APIs (Instant Setup)
```bash
# Luma Dream Machine - Best quality
python examples/generate_videos.py --questions-dir ./questions --model luma-ray-2

# Google Veo 3.1 - Latest with 1080p + audio
python examples/generate_videos.py --questions-dir ./questions --model veo-3.1-generate

# Kling AI 2.6 - Latest Kling with best quality
python examples/generate_videos.py --questions-dir ./questions --model kling-v2-6

# Runway Gen-4.5 - World's top-rated video model
python examples/generate_videos.py --questions-dir ./questions --model runway-gen45

# OpenAI Sora 2 - High-quality generation
python examples/generate_videos.py --questions-dir ./questions --model openai-sora-2
```

#### Open-Source Models (Requires Installation)
```bash
# LTX-Video - Lightweight, good quality
bash setup/install_model.sh ltx-video
python examples/generate_videos.py --questions-dir ./questions --model ltx-video

# Stable Video Diffusion - Proven model
bash setup/install_model.sh svd
python examples/generate_videos.py --questions-dir ./questions --model svd

# HunyuanVideo - High-quality up to 720p
bash setup/install_model.sh hunyuan-video-i2v
python examples/generate_videos.py --questions-dir ./questions --model hunyuan-video-i2v

# CogVideoX - Long-form generation
bash setup/install_model.sh cogvideox-5b-i2v
python examples/generate_videos.py --questions-dir ./questions --model cogvideox-5b-i2v
```
