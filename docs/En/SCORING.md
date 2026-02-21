# VBVR-EvalKit Scoring

VBVR-Bench rule-based evaluation for assessing video generation models' reasoning capabilities. 100+ task-specific evaluators, deterministic 0-1 scores, no API calls.

## How It Works

Each VBVR-Bench evaluator is matched by the **generator name** in the directory path. For example, `O-9_shape_scaling_data-generator` maps to the shape-scaling evaluator. The evaluator reads the generated video and reference data, then scores it.

### End-to-End Workflow

**1. Generate questions** using a [VBVR-DataFactory](https://github.com/VBVR-DataFactory) data-generator:
```bash
git clone https://github.com/VBVR-DataFactory/O-9_shape_scaling_data-generator.git
cd O-9_shape_scaling_data-generator && pip install -r requirements.txt
python examples/generate.py --num-samples 10 --seed 42 --output /path/to/questions
```

**2. Run inference** to generate videos:
```bash
python examples/generate_videos.py --questions-dir ./questions --output-dir ./outputs --model svd
```

**3. Organize for evaluation.** The evaluator expects this directory structure:
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

The **generator name** directory (e.g., `O-9_shape_scaling_data-generator`) must match a VBVR-Bench task name so the correct rule-based evaluator is selected. Each sample needs both the generated `video/` and the reference `question/` files.

**4. Run evaluation:**
```bash
python examples/score_videos.py --inference-dir ./outputs_rubrics
```

### Batch Processing

Multiple generators can be placed under the same root. The evaluator walks all of them automatically:

```
outputs_rubrics/
└── svd/
    ├── G-3_stable_sort_data-generator/
    │   └── stable_sort_task/...
    ├── O-9_shape_scaling_data-generator/
    │   └── shape_scaling_task/...
    └── G-15_maze_solving_data-generator/
        └── maze_solving_task/...
```

```bash
python examples/score_videos.py --inference-dir ./outputs_rubrics
```

## Scoring

By default, only the `task_specific` score is returned (reasoning correctness). Use `--full-score` for the weighted combination of all 5 dimensions:

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| `task_specific` | 25% | Task-specific reasoning logic |
| `final_frame_accuracy` | 35% | Does the final frame match the expected result? |
| `first_frame_consistency` | 15% | Does the first frame match the input image? |
| `temporal_smoothness` | 15% | Are frame transitions smooth? |
| `visual_quality` | 10% | Sharpness and noise levels |

## CLI Reference

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

You can also use the runner module directly:
```bash
python -m vbvrevalkit.runner.score --inference-dir ./outputs_rubrics
```

## Output

The evaluator writes two types of files:

**Per-sample** (`VBVRBenchEvaluator.json`):
```json
{
  "metadata": {
    "evaluator": "VBVRBenchEvaluator",
    "model_name": "svd",
    "task_type": "O-9_shape_scaling_data-generator/shape_scaling_task",
    "task_id": "shape_scaling_00000000"
  },
  "result": {
    "score": 0.8667,
    "dimensions": { "task_specific": 0.8667 },
    "details": {
      "task_specific_details": {
        "element_preservation": 0.6667,
        "scaling_ratio": 1.0,
        "shape_type_matching": 1.0,
        "position_correctness": 1.0
      }
    },
    "evaluation_type": "rubrics",
    "vbvr_task_name": "O-9_shape_scaling_data-generator"
  }
}
```

**Summary** (`VBVRBenchEvaluator_summary.json`):
```json
{
  "global_statistics": {
    "total_models": 1,
    "total_samples": 10,
    "mean_score": 0.8667
  },
  "models": {
    "svd": {
      "model_statistics": { "mean_score": 0.8667, "total_samples": 10 },
      "by_category": { "Transformation": { "mean_score": 0.8667 } },
      "by_split": { "Out_of_Domain": { "mean_score": 0.8667 } }
    }
  }
}
```

Breakdowns include:
- **Model**: Mean, median, std per model
- **Category**: 6 categories (Abstraction, Categorization, Navigation, Perception, Physics, Transformation)
- **Split**: In_Domain / Out_of_Domain (50 tasks each)

Evaluation is **resumable** — re-running the same command skips already-scored tasks.
