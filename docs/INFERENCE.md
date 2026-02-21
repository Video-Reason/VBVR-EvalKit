# VBVR-EvalKit Inference Module

## Quick Start

```bash
# 1. Set up questions directory (only first_frame.png and prompt.txt required)
# questions/chess_task/chess_0000/{first_frame.png, prompt.txt}

# 2. Generate videos (runs all discovered tasks)
python examples/generate_videos.py --questions-dir ./questions --output-dir ./outputs --model svd

# 3. Run with specific models  
python examples/generate_videos.py --questions-dir ./questions --output-dir ./outputs --model luma-ray-2
```

## Core Concepts

### Task Pairs: The Evaluation Unit

VBVR-EvalKit evaluates video models' reasoning capabilities through **Task Pairs** - carefully designed visual reasoning problems:

| Component | File | Purpose | Required |
|-----------|------|---------|----------|
| **Initial State** | `first_frame.png` | Problem/puzzle to solve | Required |
| **Text Prompt** | `prompt.txt` | Natural language instructions | Required |
| **Final State** | `final_frame.png` | Solution/goal reference | Optional |
| **Ground Truth** | `ground_truth.mp4` | Reference video | Optional |

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

## Architecture

VBVR-EvalKit uses a **modular architecture** with dynamic loading:

- **MODEL_CATALOG**: Registry of 37 models across 15 families
- **Dynamic Loading**: Models loaded on-demand via importlib
- **Unified Interface**: All models inherit from `ModelWrapper`
- **Two Categories**:
  - **Commercial APIs**: Instant setup with API keys (Luma, Veo, Kling, Sora, Runway)
  - **Open-Source**: Local installation required (LTX-Video, LTX-2, HunyuanVideo, DynamiCrafter, SVD)

## Output Structure

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

## Python API

```python
from vbvrevalkit.runner.inference import InferenceRunner

runner = InferenceRunner(output_dir="./outputs")
result = runner.run(
    model_name="luma-ray-2",
    image_path="questions/chess_task/chess_0000/first_frame.png",
    text_prompt="Find the checkmate move"
)
print(f"Generated: {result['video_path']}")
```

## Configuration

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