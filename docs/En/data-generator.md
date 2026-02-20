# Data Generation → Inference → Evaluation End-to-End Workflow

This document walks through the complete pipeline — from generating questions, running model inference, to rubrics evaluation — using `O-9_shape_scaling_data-generator` (shape scaling analogy reasoning) as an example. All 100 VBVR-Bench data-generators follow the same pattern.

## Prerequisites

```bash
# 1. Install VMEvalKit
cd /path/to/VMEvalKit
pip install -e .

# 2. Install at least one video generation model (SVD as example)
bash setup/install_model.sh --model svd --validate

# 3. Clone a data-generator (O-9 as example)
git clone https://github.com/VBVR-DataFactory/O-9_shape_scaling_data-generator.git
cd O-9_shape_scaling_data-generator
pip install -r requirements.txt
```

## Full Pipeline

### Step 1: Generate Questions

Use the data-generator to produce task samples. Each sample contains: first frame (`first_frame.png`), final frame (`final_frame.png`), prompt (`prompt.txt`), and reference video (`ground_truth.mp4`).

```bash
cd /path/to/O-9_shape_scaling_data-generator

# Generate 1 sample (for testing)
python examples/generate.py --num-samples 1 --seed 42 --output /path/to/VMEvalKit/questions

# Generate 100 samples (for full evaluation)
python examples/generate.py --num-samples 100 --seed 42 --output /path/to/VMEvalKit/questions
```

Output structure:

```
VMEvalKit/questions/
└── shape_scaling_task/
    └── shape_scaling_00000000/
        ├── first_frame.png       # Initial state (analogy A:B :: C:?)
        ├── final_frame.png       # Goal state (correct answer for ?)
        ├── prompt.txt            # Task description
        ├── ground_truth.mp4      # Reference video (16fps, ~3.8s)
        └── metadata.json         # Generation metadata
```

### Step 2: Run Model Inference

Use VMEvalKit's inference pipeline to generate videos.

```bash
cd /path/to/VMEvalKit

# Generate with SVD
python examples/generate_videos.py \
  --questions-dir ./questions \
  --output-dir ./outputs \
  --model svd

# Or use other models
python examples/generate_videos.py \
  --questions-dir ./questions \
  --output-dir ./outputs \
  --model luma-ray-2
```

Inference output is a flat structure:

```
outputs/svd/shape_scaling_task/shape_scaling_00000000.mp4
```

### Step 3: Organize Directory Structure

The rubrics evaluator requires VMEvalKit's run directory structure (with `video/` and `question/` subdirectories). Reorganize the inference output:

```bash
# Set variables
MODEL=svd
GENERATOR=O-9_shape_scaling_data-generator
TASK_TYPE=shape_scaling_task
TASK_ID=shape_scaling_00000000
QUESTIONS_DIR=./questions/${TASK_TYPE}/${TASK_ID}

# Create directory structure
OUTPUT_DIR=./outputs_rubrics/${MODEL}/${GENERATOR}/${TASK_TYPE}/${TASK_ID}/default
mkdir -p ${OUTPUT_DIR}/video
mkdir -p ${OUTPUT_DIR}/question

# Copy generated video
cp ./outputs/${MODEL}/${TASK_TYPE}/${TASK_ID}.mp4 ${OUTPUT_DIR}/video/output.mp4

# Copy question files (evaluator needs reference data)
cp ${QUESTIONS_DIR}/first_frame.png  ${OUTPUT_DIR}/question/
cp ${QUESTIONS_DIR}/final_frame.png  ${OUTPUT_DIR}/question/
cp ${QUESTIONS_DIR}/prompt.txt       ${OUTPUT_DIR}/question/
cp ${QUESTIONS_DIR}/ground_truth.mp4 ${OUTPUT_DIR}/question/   # optional
```

Final directory structure:

```
outputs_rubrics/
└── svd/
    └── O-9_shape_scaling_data-generator/    # generator name (must match VBVR-Bench task name)
        └── shape_scaling_task/
            └── shape_scaling_00000000/
                └── default/                 # run_id (any name)
                    ├── video/
                    │   └── output.mp4       # Model-generated video
                    └── question/
                        ├── first_frame.png  # Reference first frame
                        ├── final_frame.png  # Reference final frame
                        ├── prompt.txt       # Prompt text
                        └── ground_truth.mp4 # GT video
```

> **Key**: The top-level directory name must be the VBVR-Bench task name (e.g., `O-9_shape_scaling_data-generator`), following the format `{uppercase letter}-{number}_{description}_data-generator`. The evaluator uses this name to match the corresponding rule-based evaluator.

### Step 4: Run Rubrics Evaluation

```bash
cd /path/to/VMEvalKit

# Run evaluation
python -m vmevalkit.runner.score rubrics \
  --inference-dir ./outputs_rubrics \
  --eval-output-dir ./evaluations/rubrics \
  --device cuda
```

### Step 5: View Results

Two types of output files are generated:

**Per-sample result** (`VBVRBenchEvaluator.json`):

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

**Summary file** (`VBVRBenchEvaluator_summary.json`):

```json
{
  "global_statistics": {
    "total_models": 1,
    "total_samples": 1,
    "mean_score": 0.8667
  },
  "models": {
    "svd": {
      "model_statistics": { "mean_score": 0.8667, "total_samples": 1 },
      "by_category": { "Transformation": { "mean_score": 0.8667 } },
      "by_split": { "Out_of_Domain": { "mean_score": 0.8667 } }
    }
  }
}
```

## Batch Processing Multiple Data-Generators

For multiple data-generators, repeat the above steps. The directory structure supports placing multiple generators under the same `outputs_rubrics/`:

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

The evaluation command stays the same — the evaluator automatically walks all generators and matches them to the corresponding rule-based evaluators:

```bash
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs_rubrics
```

The summary file automatically includes score breakdowns by category (6 categories) and by split (In_Domain / Out_of_Domain).

## Parameter Reference

### Data-Generator Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--num-samples` | Number of samples to generate | `100` |
| `--seed` | Random seed for reproducibility | `42` |
| `--output` | Output directory | `./questions` |

### Inference Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--model` | Model name(s) | `svd`, `luma-ray-2` |
| `--questions-dir` | Questions directory | `./questions` |
| `--output-dir` | Output directory | `./outputs` |
| `--domains` | Process only specified domains | `shape_scaling_task` |

### Rubrics Evaluation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--inference-dir, -i` | (required) | Inference output directory (VMEvalKit structure required) |
| `--eval-output-dir, -o` | `./evaluations/rubrics` | Directory to save evaluation results |
| `--gt-base-path, -g` | None | Path to VBVR-Bench GT data (optional) |
| `--device` | `cuda` | Computation device (`cuda` / `cpu`) |
| `--full-score` | off | Use full 5-dimension weighted score instead of task_specific only |
