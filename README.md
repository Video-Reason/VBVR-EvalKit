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

VBVR-EvalKit uses **VBVR-Bench** for evaluation — 100+ task-specific rule-based evaluators that produce deterministic 0-1 scores. No API calls needed.

Each evaluator is matched by the **generator name** in the directory path (e.g., `O-9_shape_scaling_data-generator` maps to the shape-scaling evaluator). The evaluator reads the generated video and reference data, then scores it.

### How It Works

**Step 1: Generate questions** using a [VBVR-DataFactory](https://github.com/VBVR-DataFactory) data-generator:
```bash
git clone https://github.com/VBVR-DataFactory/O-9_shape_scaling_data-generator.git
cd O-9_shape_scaling_data-generator && pip install -r requirements.txt
python examples/generate.py --num-samples 10 --seed 42 --output /path/to/questions
```

**Step 2: Run inference** to generate videos:
```bash
python examples/generate_videos.py --questions-dir ./questions --output-dir ./outputs --model svd
```

**Step 3: Organize for evaluation.** The evaluator expects this directory structure:
```
outputs_rubrics/
└── {model_name}/
    └── {generator_name}/                  # e.g., O-9_shape_scaling_data-generator
        └── {task_type}/                   # e.g., shape_scaling_task
            └── {task_id}/                 # e.g., shape_scaling_00000000
                └── {run_id}/             # any name (e.g., "default")
                    ├── video/
                    │   └── output.mp4     # model-generated video
                    └── question/
                        ├── first_frame.png   # reference first frame
                        ├── final_frame.png   # reference final frame
                        ├── prompt.txt        # text prompt
                        └── ground_truth.mp4  # GT video (optional)
```

The key is that the **generator name** directory (e.g., `O-9_shape_scaling_data-generator`) must match a VBVR-Bench task name so the evaluator knows which rule-based evaluator to use. Each sample needs both the generated `video/` and the reference `question/` files.

**Step 4: Run evaluation:**
```bash
python examples/score_videos.py --inference-dir ./outputs_rubrics
```

### Scoring

By default, only the `task_specific` score is returned (reasoning correctness). Use `--full-score` for the weighted combination of all 5 dimensions:

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| `task_specific` | 25% | Task-specific reasoning logic |
| `final_frame_accuracy` | 35% | Does the final frame match the expected result? |
| `first_frame_consistency` | 15% | Does the first frame match the input image? |
| `temporal_smoothness` | 15% | Are frame transitions smooth? |
| `visual_quality` | 10% | Sharpness and noise levels |

### CLI Reference

```bash
python examples/score_videos.py --inference-dir ./outputs_rubrics                     # basic
python examples/score_videos.py --inference-dir ./outputs_rubrics --full-score         # all 5 dimensions
python examples/score_videos.py --inference-dir ./outputs_rubrics --device cpu         # CPU mode
python examples/score_videos.py --inference-dir ./outputs_rubrics --gt-base-path /path # external GT data
python examples/score_videos.py --inference-dir ./outputs_rubrics -o ./my_evals        # custom output dir
```

| Flag | Default | Description |
|------|---------|-------------|
| `-i, --inference-dir` | (required) | Root directory with the structure above |
| `-o, --eval-output-dir` | `./evaluations/rubrics` | Where to write result JSONs |
| `-g, --gt-base-path` | None | External GT data path (optional) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--full-score` | off | Score all 5 dimensions instead of just `task_specific` |

### Output

The evaluator writes two types of files:

**Per-sample** (`VBVRBenchEvaluator.json`) — one per task, with score, dimensions, and task-specific details.

**Summary** (`VBVRBenchEvaluator_summary.json`) — aggregated stats broken down by:
- Model (mean, median, std per model)
- Category (6 categories: Abstraction, Categorization, Navigation, Perception, Physics, Transformation)
- Split (In_Domain / Out_of_Domain, 50 tasks each)

Evaluation is **resumable** — re-running the same command skips already-scored tasks.

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