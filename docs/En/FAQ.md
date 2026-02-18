# Frequently Asked Questions (FAQ)

## Installation

### Q: After `pip install -e .`, running an open-source model throws `ModuleNotFoundError: No module named 'diffusers'`

**Cause:** Open-source model dependencies (torch, diffusers, transformers, etc.) are not included in VMEvalKit's core dependencies and must be installed separately.

**Solutions:**

Option 1: Install model dependencies in the main venv (recommended for quick testing):
```bash
source venv/bin/activate
pip install diffusers transformers accelerate torch torchvision
```

Option 2: Use the model installation script (creates an isolated venv):
```bash
bash setup/install_model.sh --model svd
```

> Note: The installation script creates an isolated venv under `envs/{model-name}/`, but the current inference script imports model dependencies directly in the main process, so these packages also need to be available in the main venv.

---

### Q: Installation script errors with `No matching distribution found for torchvision==0.20.1`

**Cause:** Older pinned versions of torch/torchvision are unavailable for Python 3.13. Pinned version numbers may not work with newer Python versions.

**Solution:** Edit `setup/models/{model}/setup.sh` to remove version pins:
```bash
# Before (may be incompatible)
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# After (installs compatible versions automatically)
pip install -q torch torchvision torchaudio
```

---

### Q: `nvidia-smi` reports `Driver/library version mismatch` or CUDA is unavailable

**Cause:** NVIDIA driver version doesn't match the CUDA library version.

**Solutions:**
1. Models like SVD support CPU fallback and will automatically use `float32` on CPU (slower but functional)
2. For GPU acceleration, update the NVIDIA driver or install a matching CUDA version:
   ```bash
   # Check current driver version
   cat /proc/driver/nvidia/version
   # Install matching CUDA toolkit
   ```

---

## Inference

### Q: Tasks show `Skipped (existing)` when running `generate_videos.py`

**Cause:** A `{task_id}.mp4` already exists in the output directory; completed tasks are skipped by default.

**Solutions:**
```bash
# Option 1: Use --override to clear output directory and re-run
python examples/generate_videos.py --questions-dir ./questions --model svd --override

# Option 2: Manually delete the corresponding mp4 file
rm outputs/svd/test_task/test_0000.mp4
```

---

### Q: `--list-models` errors with `the following arguments are required: --model`

**Cause:** `--model` is a required argument, even when using `--list-models`.

**Solution:** List models directly via Python:
```bash
python -c "
from vmevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS, MODEL_FAMILIES
for f, ms in MODEL_FAMILIES.items():
    print(f'{f} ({len(ms)}):')
    for m in ms: print(f'  {m}')
print(f'\nTotal: {len(AVAILABLE_MODELS)} models')
"
```

---

### Q: Inference is extremely slow in CPU mode

**Cause:** Open-source models use `float32` arithmetic on CPU, which is computationally expensive.

**Reference timing (SVD, CPU, single image):** ~2 minutes per task

**Recommendations:**
- Use a GPU environment (16GB+ VRAM recommended)
- Or use commercial API models (no GPU required):
  ```bash
  python examples/generate_videos.py --questions-dir ./questions --model luma-ray-2
  ```

---

## Evaluation

### Q: How to choose an evaluation method?

| Method | Config Value | Dependencies | Use Case |
|--------|-------------|--------------|----------|
| Human eval | `"human"` | Gradio | Small-scale precise evaluation |
| GPT-4O | `"gpt4o"` | `OPENAI_API_KEY` | Large-scale automated evaluation |
| InternVL | `"internvl"` | Local deployment (30GB VRAM) | Automated evaluation without API budget |
| Qwen3-VL | `"qwen"` | Local deployment | Automated evaluation without API budget |
| Multi-frame | `"multiframe_gpt4o"` etc. | Same as above | Higher accuracy evaluation |

---

## Environment Variables

### Q: Which API keys are needed?

| Model Family | Environment Variable | How to Obtain |
|-------------|---------------------|---------------|
| Luma | `LUMA_API_KEY` | Luma AI website |
| Google Veo | `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| Kling AI | `KLING_API_KEY` | Kling AI website |
| Runway | `RUNWAYML_API_SECRET` | Runway ML website |
| OpenAI Sora | `OPENAI_API_KEY` | OpenAI platform |
| GPT-4O eval | `OPENAI_API_KEY` | Same as above |

```bash
cp env.template .env
# Edit .env and fill in the corresponding keys
```
