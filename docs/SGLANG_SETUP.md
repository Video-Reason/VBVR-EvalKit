# SGLang Setup and Usage Guide

This guide explains how to set up and use SGLang for video generation in VMEvalKit.

## Overview

SGLang provides efficient inference for video diffusion models including:
- **HunyuanVideo-I2V** (Tencent)
- **WAN 2.1/2.2** (Wan-AI)  
- **FastWan** models (Hao AI Lab @ UCSD)

SGLang Issue #12850 has been fixed (closed on 2025-11-09), so the implementation is ready for testing.

## Installation

### Option 1: Install from Source (Recommended)

```bash
# Clone SGLang repository
git clone https://github.com/sgl-project/sglang.git
cd sglang/python

# Install with diffusion support
pip install -e ".[diffusion]"
```

### Option 2: Direct pip install

```bash
pip install -e "python[diffusion]"
```

**Note**: `pip install "sglang[all]"` may not work correctly for diffusion models. Use the source installation method above.

## Usage Modes

### Mode 1: Persistent Server (Recommended)

This approach avoids reloading the model on every inference, providing much better performance for multiple requests.

#### Start the Server

```bash
# Example: Start WAN 2.1 server with 4 GPUs
sglang serve \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --text-encoder-cpu-offload \
  --pin-cpu-memory \
  --num-gpus 4 \
  --ulysses-degree=2 \
  --ring-degree=2
```

Server Arguments:
- `--model-path`: HuggingFace model path
- `--num-gpus`: Number of GPUs to use
- `--ulysses-degree`: Ulysses attention parallelism degree
- `--ring-degree`: Ring attention parallelism degree
- `--text-encoder-cpu-offload`: Offload text encoder to CPU
- `--pin-cpu-memory`: Pin CPU memory for faster transfers

#### Use the API

```python
from vmevalkit.runner import inference

# The wrapper will automatically detect and use the running server
result = inference(
    model="sglang-wan-2.1",
    task="example_i2v",
    use_persistent_server=True,  # Default
    sglang_server_url="http://localhost:30000"  # Default
)
```

### Mode 2: Direct Loading

This mode loads the model for each inference. Simpler but slower for multiple requests.

```python
result = inference(
    model="sglang-wan-2.1",
    task="example_i2v",
    use_persistent_server=False  # Disable server mode
)
```

## Supported Models

### HunyuanVideo
```python
model="sglang-hunyuan-video-i2v"
```

### WAN Models
```python
model="sglang-wan-2.1"      # WAN 2.1 FLF2V 14B 720P
model="sglang-wan-2.2"      # WAN 2.2 I2V A14B
```

### FastWan Models (by Hao AI Lab @ UCSD)
```python
model="sglang-fastwan"         # 480P, 1.3B params (default)
model="sglang-fastwan-1.3b"    # 480P, 1.3B params (explicit)
model="sglang-fastwan-14b"     # 720P, 14B params (preview)
model="sglang-fastwan-2.2-5b"  # 720P, 5B params
```

## Configuration Parameters

### Wrapper Parameters

```python
SGLangWrapper(
    model="sglang-wan-2.1",
    output_dir="./data/outputs",
    sglang_server_url="http://localhost:30000",  # Server URL
    use_persistent_server=True,                   # Use server if available
    use_docker=True,                              # Use Docker (optional)
    num_gpus=4                                    # GPUs for direct mode
)
```

### Generation Parameters

```python
wrapper.generate(
    image_path="path/to/image.jpg",
    text_prompt="A beautiful sunset",
    duration=8.0,              # Video duration in seconds
    height=720,                # Video height
    width=1280,                # Video width
    fps=25,                    # Frames per second
    seed=42,                   # Random seed for reproducibility
    output_filename="output.mp4"
)
```

## Performance Tips

1. **Use Persistent Server**: Start `sglang serve` once and reuse it for multiple inferences
2. **Multi-GPU**: Use multiple GPUs with `--num-gpus` for faster generation
3. **Memory Optimization**: Use `--text-encoder-cpu-offload` and `--pin-cpu-memory`
4. **Parallelism**: Adjust `--ulysses-degree` and `--ring-degree` based on your GPU setup

## Troubleshooting

### ModuleNotFoundError: No module named 'sglang.multimodal_gen'

This means SGLang was not installed correctly. Use the source installation method:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang/python
pip install -e ".[diffusion]"
```

### Server Not Responding

Check if the server is running:
```bash
curl http://localhost:30000/health
```

If not, start the server as shown in Mode 1 above.

### Out of Memory

Try these solutions:
1. Use `--text-encoder-cpu-offload` to offload text encoder
2. Reduce `--num-gpus` if you don't have enough VRAM
3. Use a smaller model (e.g., `fastwan-1.3b` instead of `fastwan-14b`)

## Example Usage

### Complete Example with Persistent Server

```bash
# Terminal 1: Start server
sglang serve \
  --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
  --num-gpus 4 \
  --ulysses-degree=2 \
  --ring-degree=2
```

```python
# Terminal 2: Run inference
from vmevalkit.runner import inference

result = inference(
    model="sglang-wan-2.1",
    task="example_i2v",
    use_persistent_server=True
)

print(f"Video saved to: {result['video_path']}")
```

## References

- [SGLang Blog Post](https://lmsys.org/blog/2025-11-07-sglang-diffusion/)
- [SGLang CLI Documentation](https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cli.md)
- [SGLang GitHub](https://github.com/sgl-project/sglang)
- [FastWan Models](https://hao-ai-lab.github.io/blogs/fastvideo_post_training/)

