# VBVR-EvalKit Scoring

Rule-based evaluation for assessing video generation models' reasoning capabilities using VBVR-Bench.

## VBVR-Bench Evaluation

VBVR-Bench provides 100+ task-specific evaluators that produce deterministic, fully reproducible 0-1 continuous scores. No API calls needed.

### Usage

```bash
# Basic evaluation
python examples/score_videos.py --inference-dir ./outputs

# With GT data and device selection
python examples/score_videos.py --inference-dir ./outputs --gt-base-path /path/to/gt --device cuda

# Full 5-dimension weighted score (default: task_specific only)
python examples/score_videos.py --inference-dir ./outputs --full-score

# Custom output directory
python examples/score_videos.py --inference-dir ./outputs --eval-output-dir ./evaluations/rubrics

# Via the runner module
python -m vbvrevalkit.runner.score --inference-dir ./outputs
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--inference-dir, -i` | (required) | Directory containing inference outputs |
| `--eval-output-dir, -o` | `./evaluations/rubrics` | Directory to save evaluation results |
| `--gt-base-path, -g` | None | Path to VBVR-Bench GT data (optional, for ground_truth.mp4) |
| `--device` | `cuda` | Computation device (`cuda` or `cpu`) |
| `--full-score` | off | Use full 5-dimension weighted score instead of task_specific only |

## Scoring Scale

### 0-1 Continuous Score

VBVR-Bench evaluators produce 0-1 continuous scores across 5 dimensions:

| Dimension | Weight | Description |
|-----------|--------|-------------|
| `first_frame_consistency` | 0.15 | First frame alignment with ground truth |
| `final_frame_accuracy` | 0.35 | Final frame correctness |
| `temporal_smoothness` | 0.15 | Temporal coherence between frames |
| `visual_quality` | 0.10 | Visual fidelity (sharpness, noise) |
| `task_specific` | 0.25 | Task-specific reasoning logic |

Default mode (without `--full-score`) returns only the `task_specific` dimension score, focusing on reasoning correctness.

## Data Directory Requirements

The evaluator requires reference data in the `question/` subdirectory:

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

Each sample is saved as a separate `VBVRBenchEvaluator.json`. After evaluation completes, a summary file `VBVRBenchEvaluator_summary.json` is generated, containing:
- **Global statistics**: Mean, median, std across all models
- **Model statistics**: Per-model score summaries
- **By category**: Scores for 6 categories (Abstraction, Categorization, Navigation, Perception, Physics, Transformation)
- **By split**: In_Domain / Out_of_Domain scores

Supports resume — previously evaluated tasks are not re-evaluated.
