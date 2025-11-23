## Installation

Install the required dependencies for open-source models:

```bash
uv pip install transformers diffusers
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
- **Type**: Image-to-video generation

## LTX-Video

refer https://huggingface.co/Lightricks/LTX-Video-0.9.8-13B-distilled 

LTX-Video is the first DiT-based video generation model capable of generating high-quality videos in real-time. It produces 30 FPS videos at a 1216×704 resolution faster than they can be watched.
### Usage

```bash
uv pip install sentencepiece accelerate #  otherwise has error. refer https://huggingface.co/Lightricks/LTX-Video/discussions/96
# Generate videos using LTX-Video model
uv run examples/generate_videos.py --model ltx-video --task chess maze
```


###  Opensource Evaluator


refer https://huggingface.co/OpenGVLab/InternVL3-8B

```bash
uv pip install lmdeploy timm peft>=0.17.0 openai
CUDA_VISIBLE_DEVICES=2 lmdeploy serve api_server OpenGVLab/InternVL3-8B --chat-template internvl2_5 --server-port 23333 --tp 1 # takes 30GB vram.

# in another terminal
uv run examples/score_videos.py internvl
```

