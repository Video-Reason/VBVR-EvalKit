# VMEvalKit Inference Module

## 🚀 Quick Start

```bash
# 1. Set up questions directory (only first_frame.png and prompt.txt required)
# questions/chess_task/chess_0000/{first_frame.png, prompt.txt}

# 2. Generate videos (runs all discovered tasks)
python examples/generate_videos.py --questions-dir ./questions --output-dir ./outputs --model svd

# 3. Run with specific models  
python examples/generate_videos.py --questions-dir ./questions --output-dir ./outputs --model luma-ray-2
```

## 📚 Core Concepts

### Task Pairs: The Evaluation Unit

VMEvalKit evaluates video models' reasoning capabilities through **Task Pairs** - carefully designed visual reasoning problems:

| Component | File | Purpose | Required |
|-----------|------|---------|----------|
| 📸 **Initial State** | `first_frame.png` | Problem/puzzle to solve | ✅ Required |
| 📝 **Text Prompt** | `prompt.txt` | Natural language instructions | ✅ Required |
| 🎯 **Final State** | `final_frame.png` | Solution/goal reference | ⚪ Optional |
| 🎬 **Ground Truth** | `ground_truth.mp4` | Reference video | ⚪ Optional |

**Directory Structure:**
```
questions/
├── chess_task/
│   ├── chess_0000/
│   │   ├── first_frame.png      # Initial state (required)
│   │   ├── prompt.txt           # Instructions (required)
│   │   ├── final_frame.png      # Goal state (optional)
│   │   └── ground_truth.mp4     # Reference (optional)
│   └── chess_0001/...
├── maze_task/...
└── sudoku_task/...
```

Models receive the initial state + prompt and must generate videos demonstrating the reasoning process to reach the final state.

## 🏗️ Architecture

VMEvalKit uses a **modular architecture** with dynamic loading:

- **MODEL_CATALOG**: Registry of 33 models across 13 families
- **Dynamic Loading**: Models loaded on-demand via importlib
- **Unified Interface**: All models inherit from `ModelWrapper`
- **Two Categories**:
  - **Commercial APIs**: Instant setup with API keys (Luma, Veo, Kling, Sora, Runway)
  - **Open-Source**: Local installation required (LTX-Video, LTX-2, HunyuanVideo, SVD)

## 📂 Output Structure

Outputs are organized hierarchically: `model/domain_task/task_id/run_id/`

```
outputs/
├── luma-ray-2/
│   └── chess_task/
│       └── chess_0000/
│           └── luma-ray-2_chess_0000_20250103_143025/
│               ├── video/generated_video.mp4
│               ├── question/{first_frame.png, prompt.txt, final_frame.png}
│               └── metadata.json  # Generated: run info, model, duration, status
```



## 💻 Python API

```python
from vmevalkit.runner.inference import InferenceRunner

runner = InferenceRunner(output_dir="./outputs")
result = runner.run(
    model_name="luma-ray-2",
    image_path="questions/chess_task/chess_0000/first_frame.png",
    text_prompt="Find the checkmate move"
)
print(f"Generated: {result['video_path']}")
```


## ⚙️ Configuration

### API Keys
```bash
cp env.template .env
# Edit .env with your API keys:
LUMA_API_KEY=your_key_here
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
KLING_API_KEY=your_kling_key
RUNWAYML_API_SECRET=your_runway_secret
```


# Open-Source Model Test Results

Task: `shape_scaling_00000000` (1 task)

Env: RTX A6000

## Tested & Succeeded

| Model | Time | Notes |
|---|---|---|
| svd | - | Previous results available |
| ltx-video | 1m31s | |
| ltx-video-13b-distilled | 5m38s | |
| wan-2.2-ti2v-5b | 9m19s | |
| sana-video-2b-480p | ~12s | Succeeded after fix |
| cogvideox1.5-5b-i2v | 5m22s | Reinstalled venv + forced native resolution/frames |

## TODO (Need Re-run)

| Model | Reason |
|---|---|
| wan-2.1-i2v-480p | Process ran but outputs were cleared, need re-run to confirm |
| wan-2.1-i2v-720p | Same as above |
| wan-2.2-i2v-a14b | Same as above |

## Failed - Environment/Dependency Issues

| Model | Error | Fix |
|---|---|---|
| cogvideox-5b-i2v | venv missing torch | Reinstall venv: `bash setup/install_model.sh --model cogvideox-5b-i2v` |

## Failed - Missing Weights/Installation

| Model | Error | Fix |
|---|---|---|
| LTX-2 | `RuntimeError: LTX-2 is not installed` | Run setup script |
| hunyuan-video-i2v | Missing CLIP text encoder | `huggingface-cli download openai/clip-vit-large-patch14` |
| morphic-frames-to-video | Missing Wan2.2 + morphic LoRA weights | `huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./weights/wan/Wan2.2-I2V-A14B` + `huggingface-cli download morphic/Wan2.2-frames-to-video --local-dir ./weights/morphic` |