
## 🎬 Supported Models

VMEvalKit provides unified access to **40 video generation models** across **11 provider families**:

For commercial APIs, we support Luma Dream Machine, Google Veo, Google Veo 3.1, WaveSpeed WAN 2.1, WaveSpeed WAN 2.2, Runway ML, OpenAI Sora.
For open-source models, we support HunyuanVideo, VideoCrafter, DynamiCrafter, Stable Video Diffusion, Morphic, LTX-Video.

### Commercial APIs (32 models)

| Provider | Models | Key Features | API Required |
|----------|---------|-------------|--------------|
| **Luma Dream Machine** | 2 | `luma-ray-2`, `luma-ray-flash-2` | `LUMA_API_KEY` |
| **Google Veo** | 3 | `veo-2.0-generate`, `veo-3.0-generate`, `veo-3.0-fast-generate` | GCP credentials |
| **Google Veo 3.1** | 4 | Native 1080p, audio generation (via WaveSpeed) | `WAVESPEED_API_KEY` |
| **WaveSpeed WAN 2.1** | 8 | 480p/720p variants with LoRA and ultra-fast options | `WAVESPEED_API_KEY` |
| **WaveSpeed WAN 2.2** | 10 | Enhanced 5B models, improved quality | `WAVESPEED_API_KEY` |
| **Runway ML** | 3 | Gen-3A Turbo, Gen-4 Turbo, Gen-4 Aleph | `RUNWAYML_API_SECRET` |
| **OpenAI Sora** | 2 | Sora-2, Sora-2-Pro (4s/8s/12s durations) | `OPENAI_API_KEY` |

### Open-Source Models

| Provider | Models | Key Features | Hardware Requirements |
|----------|---------|-------------|----------------------|
| **HunyuanVideo** | 1 | High-quality 720p I2V | GPU with 24GB+ VRAM |
| **VideoCrafter** | 1 | Text-guided video synthesis | GPU with 16GB+ VRAM |
| **DynamiCrafter** | 3 | 256p/512p/1024p, image animation | GPU with 12-24GB VRAM |
| **Stable Video Diffusion** | 1 | Video generation | GPU with 16GB+ VRAM |
| **Morphic** | 1 | Video generation | GPU with 16GB+ VRAM |
| **LTX-Video** | 1 | Video generation | GPU with 16GB+ VRAM |





## Installation

Install the required dependencies for open-source models:

```bash
uv pip install transformers diffusers # for diffusers support
```


## svd

It takes  20GB vram.

```bash
uv run examples/generate_videos.py --model svd --task chess maze
```

## WAN (Wan-AI)

WAN 2.1 FLF2V (First-Last Frame to Video) is an open-source image-to-video generation model using diffusers. 

The model will be automatically downloaded from HuggingFace when first used. No additional installation steps are required.

### Usage

```bash
# Generate videos using WAN model
uv run examples/generate_videos.py --model wan --task chess maze # requires more than 48GB vram.

# Or use the full model name
uv run examples/generate_videos.py --model wan-2.1-flf2v-720p --task chess maze
```

### Model Details

- **Model ID**: `Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers`
- **Resolution**: Up to 720p (automatically resized based on input image aspect ratio)
- **FPS**: 16 fps

## LTX-Video

refer https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled 

LTX-Video is the first DiT-based video generation model capable of generating high-quality videos in real-time. It produces 30 FPS videos at a 1216×704 resolution faster than they can be watched.
### Usage

```bash
uv pip install sentencepiece accelerate #  otherwise has error. refer https://huggingface.co/Lightricks/LTX-Video/discussions/96
# Generate videos using LTX-Video model
uv run examples/generate_videos.py --model ltx-video --task chess maze
```