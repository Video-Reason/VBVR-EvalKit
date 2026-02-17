# VMEvalKit Scoring

Comprehensive scoring methods for assessing video generation models' reasoning capabilities.

## Available Evaluators

### Human Evaluation
Interactive Gradio interface for human scoring.

```bash
python examples/score_videos.py --eval-config eval_config.json
# Set "method": "human" in config
```

### GPT-4O Evaluation
Automated scoring using OpenAI's GPT-4O vision model.

```bash
# Requires OPENAI_API_KEY
python examples/score_videos.py --eval-config eval_config.json  
# Set "method": "gpt4o" in config
```

### InternVL Evaluation
Open-source VLM evaluation (requires 30GB VRAM).

```bash
# Start InternVL server
bash script/lmdeploy_server.sh

# Run evaluation
python examples/score_videos.py --eval-config eval_config.json
# Set "method": "internvl" in config
```

### Qwen3-VL Evaluation
Open-source VLM evaluation using Qwen3-VL served via OpenAI-compatible API.

```bash
# Start Qwen3-VL server (e.g., via vLLM or SGLang)
# Set QWEN_API_KEY and QWEN_API_BASE in .env

# Run evaluation
python examples/score_videos.py --eval-config eval_config.json
# Set "method": "qwen" in config
```

### Multi-Frame Evaluation
Advanced evaluation using multiple video frames with consistency analysis and voting.

```bash
# Multi-frame GPT-4O, InternVL, or Qwen3-VL
# Set "method": "multiframe_gpt4o", "multiframe_internvl", or "multiframe_qwen" in config
```

### VBVR-Bench Rule-Based Evaluation (Rubrics)
Deterministic, fully reproducible evaluation using VBVR-Bench's 100 task-specific evaluators. No API calls needed.

```bash
# Basic usage
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs

# With GT data and device selection
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs --gt-base-path /path/to/gt --device cuda

# Full 5-dimension weighted score (default: task_specific only)
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs --full-score

# Custom output directory
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs --eval-output-dir ./evaluations/rubrics
```

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--inference-dir, -i` | (required) | Directory containing inference outputs |
| `--eval-output-dir, -o` | `./evaluations/rubrics` | Directory to save evaluation results |
| `--gt-base-path, -g` | None | Path to VBVR-Bench GT data (optional, for ground_truth.mp4) |
| `--device` | `cuda` | Computation device (`cuda` or `cpu`) |
| `--full-score` | off | Use full 5-dimension weighted score instead of task_specific only |

## Scoring Scale

### VLM Scoring (1-5 Scale)

**1-5 Scale** converted to **Binary** for analysis:
- **Success**: Scores 4-5 (mostly/completely correct)
- **Failure**: Scores 1-3 (incorrect/partially correct)

### Rubrics Scoring (0-1 Continuous)

VBVR-Bench evaluators produce 0-1 continuous scores across 5 dimensions:
- `first_frame_consistency` (0.15): First frame alignment
- `final_frame_accuracy` (0.35): Final frame correctness
- `temporal_smoothness` (0.15): Temporal coherence
- `visual_quality` (0.10): Visual fidelity
- `task_specific` (0.25): Task-specific reasoning logic

Default mode (without `--full-score`) returns only the `task_specific` dimension score, focusing on reasoning correctness.

## Configuration

### VLM Evaluation Config (eval_config.json)

Create `eval_config.json` to configure VLM evaluation:

```json
{
  "method": "gpt4o",
  "inference_dir": "./outputs",
  "eval_output_dir": "./evaluations",
  "temperature": 0.0,
  "multiframe": {
    "n_frames": 5,
    "strategy": "hybrid",
    "voting": "weighted_majority"
  }
}
```

### Rubrics Evaluation Config

Rubrics evaluation is configured via CLI arguments — no `eval_config.json` needed:

```bash
python -m vmevalkit.runner.score rubrics \
  --inference-dir ./outputs \
  --eval-output-dir ./evaluations/rubrics \
  --gt-base-path /path/to/vbvr-bench-gt \
  --device cuda
```

## Usage

```bash
# VLM evaluation (via config file)
python examples/score_videos.py --eval-config eval_config.json

# Test multi-frame pipeline (no API calls)
python examples/score_videos.py --test-multiframe --video path/to/video.mp4

# Rubrics evaluation (direct CLI)
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs
```

## Data Directory Requirements

### VLM Evaluation

Inference outputs only need generated videos in the `video/` subdirectory.

### Rubrics Evaluation

The rubrics evaluator requires reference data in the `question/` subdirectory:

```
outputs/
└── {model_name}/
    └── {generator_name}/          # VBVR-Bench task name (e.g., G-3_stable_sort_data-generator)
        └── {task_type}/
            └── {task_id}/
                └── {run_id}/
                    ├── video/
                    │   └── output.mp4       # Generated video
                    └── question/
                        ├── first_frame.png  # Reference first frame
                        ├── final_frame.png  # Reference final frame
                        ├── prompt.txt       # Text prompt
                        └── ground_truth.mp4 # GT video (optional)
```

## Output

### VLM Evaluation Output

Evaluations are saved to `eval_output_dir` with structured JSON files containing scores, metadata, and explanations. Results support resume capability and statistical analysis.

### Rubrics Evaluation Output

Each sample is saved as a separate `VBVRBenchEvaluator.json`. After evaluation completes, a summary file `VBVRBenchEvaluator_summary.json` is generated, containing:
- **Global statistics**: Mean, median, std across all models
- **Model statistics**: Per-model score summaries
- **By category**: Scores for 6 categories (Abstraction, Categorization, Navigation, Perception, Physics, Transformation)
- **By split**: In_Domain / Out_of_Domain scores

Supports resume — previously evaluated tasks are not re-evaluated.
