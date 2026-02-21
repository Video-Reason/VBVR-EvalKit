# VBVR-EvalKit

**Unified inference and evaluation framework for 37 video generation models.**

## Features

- **37 Models**: Unified interface for commercial APIs (Luma, Veo, Kling, Sora, Runway) + open-source (LTX-Video, LTX-2, HunyuanVideo, SVD, WAN, CogVideoX, etc.)
- **VBVR-Bench Evaluation**: 100+ task-specific rule-based evaluators producing deterministic, reproducible 0-1 scores — no API calls needed
- **Cloud Integration**: S3 + HuggingFace Hub support

## Data Format

Organize your questions outside VBVR-EvalKit with the following structure:

```
questions/
└── {domain}_task/                    # task folder (e.g., chess_task, matching_object_task)
    ├── {domain}_0000/                # individual question folder
    │   ├── first_frame.png           # required: input image for video generation
    │   ├── prompt.txt                # required: text prompt describing the video
    │   ├── final_frame.png           # optional: expected final frame for evaluation
    │   └── ground_truth.mp4          # optional: reference video for evaluation
    ├── {domain}_0001/
    │   └── ...
    └── {domain}_0002/
        └── ...
```

**Example** with domain `chess`:
```
questions/
└── chess_task/
    ├── chess_0000/
    │   ├── first_frame.png
    │   ├── prompt.txt
    │   ├── final_frame.png
    │   └── ground_truth.mp4
    ├── chess_0001/
    │   └── ...
    └── chess_0002/
        └── ...
```

**Naming Convention:**
- **Task folder**: `{domain}_task` (e.g., `chess_task`, `matching_object_task`)
- **Question folders**: `{domain}_{i:04d}` where `i` is zero-padded (e.g., `chess_0000`, `chess_0064`). Padding automatically expands beyond 4 digits when needed.

## Quick Start

```bash
# 1. Install
git clone https://github.com/Video-Reason/VBVR-EvalKit.git
cd VBVR-EvalKit

python -m venv venv
source venv/bin/activate

pip install -e .

# 2. Setup models
bash setup/install_model.sh --model svd --validate

# 3. Run inference
python examples/generate_videos.py --questions-dir setup/test_assets/ --output-dir ./outputs --model svd

# 4. Run evaluation (VBVR-Bench)
python examples/score_videos.py --inference-dir ./outputs
python examples/score_videos.py --inference-dir ./outputs --full-score --device cuda
```

## Evaluation: VBVR-Bench

VBVR-EvalKit uses **VBVR-Bench** for evaluation — a rule-based system with 100+ task-specific evaluators. No API calls, fully deterministic, and reproducible.

### Scoring Dimensions

| Dimension | Weight | Description |
|-----------|--------|-------------|
| `first_frame_consistency` | 15% | First frame alignment with ground truth |
| `final_frame_accuracy` | 35% | Final frame correctness |
| `temporal_smoothness` | 15% | Temporal coherence between frames |
| `visual_quality` | 10% | Visual fidelity (sharpness, noise) |
| `task_specific` | 25% | Task-specific reasoning logic |

Default mode returns only the `task_specific` dimension score, focusing on reasoning correctness. Use `--full-score` for all 5 dimensions.

### Usage

```bash
# Basic evaluation
python examples/score_videos.py --inference-dir ./outputs

# Full 5-dimension weighted score
python examples/score_videos.py --inference-dir ./outputs --full-score

# With GT data and device selection
python examples/score_videos.py --inference-dir ./outputs --gt-base-path /path/to/gt --device cuda

# Via the runner module
python -m vbvrevalkit.runner.score --inference-dir ./outputs
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--inference-dir, -i` | (required) | Directory containing inference outputs |
| `--eval-output-dir, -o` | `./evaluations/rubrics` | Directory to save evaluation results |
| `--gt-base-path, -g` | None | Path to VBVR-Bench GT data (optional) |
| `--device` | `cuda` | Computation device (`cuda` or `cpu`) |
| `--full-score` | off | Use full 5-dimension weighted score |

### Output

Per-sample JSON files (`VBVRBenchEvaluator.json`) and a summary file (`VBVRBenchEvaluator_summary.json`) with breakdowns by model, category (6 categories), and split (In_Domain / Out_of_Domain). Supports resume — previously evaluated tasks are not re-evaluated.

## API Keys

Set in `.env` file:
```bash
cp env.template .env
# Edit .env with your API keys:
# LUMA_API_KEY=...
# OPENAI_API_KEY=...
# GEMINI_API_KEY=...
# KLING_API_KEY=...
# RUNWAYML_API_SECRET=...
```

## Adding Models

```python
# Inherit from ModelWrapper
from vbvrevalkit.models.base import ModelWrapper

class MyModelWrapper(ModelWrapper):
    def generate(self, image_path, text_prompt, **kwargs):
        # Your inference logic
        return {"success": True, "video_path": "...", ...}
```

Register in `vbvrevalkit/runner/MODEL_CATALOG.py`:
```python
"my-model": {
    "wrapper_module": "vbvrevalkit.models.my_model_inference",
    "wrapper_class": "MyModelWrapper", 
    "family": "MyCompany"
}
```

## License

Apache 2.0