# Inference Module

This module handles **PURE INFERENCE** - no evaluation logic.

## Simple Flow
```
text + image → video model → output video
```

## Usage

### Single Inference
```python
from vmevalkit.inference import InferenceRunner

runner = InferenceRunner(output_dir="./outputs")

# Run with direct inputs
result = runner.run(
    model_name="luma-dream-machine",
    image_path="path/to/image.png",
    text_prompt="Move the star to the circle",
    duration=5.0
)

# Or run from a task
result = runner.run_from_task(
    model_name="luma-dream-machine",
    task_data={
        "prompt": "Move the star to the circle",
        "first_image_path": "path/to/image.png"
    }
)
```

### Batch Inference
```python
from vmevalkit.inference import BatchInferenceRunner

batch_runner = BatchInferenceRunner(output_dir="./outputs")

# Run on entire dataset
results = batch_runner.run_dataset(
    model_name="luma-dream-machine",
    dataset_path="data/maze_tasks/irregular_tasks.json"
)

# Compare multiple models
comparison = batch_runner.run_models_comparison(
    model_names=["luma-dream-machine", "google-veo-001"],
    dataset_path="data/maze_tasks/irregular_tasks.json"
)
```

## Command Line

### Single inference
```bash
# From dataset task
vmevalkit inference luma-dream-machine \
    --task-file data/maze_tasks/irregular_tasks.json \
    --task-id irregular_0000

# Direct image + prompt
vmevalkit inference luma-dream-machine \
    --image path/to/image.png \
    --prompt "Move the green dot to the red flag"
```

### Batch inference
```bash
# Single model on dataset
vmevalkit batch luma-dream-machine \
    --dataset data/maze_tasks/irregular_tasks.json

# Multiple models comparison
vmevalkit batch luma-dream-machine google-veo-001 \
    --dataset data/maze_tasks/irregular_tasks.json \
    --max-tasks 5
```

## Output Structure
```
outputs/
├── luma_<generation_id>.mp4      # Generated videos
├── inference_runs.json           # Log of all runs
└── batch_results/
    ├── batch_<timestamp>.json    # Batch run results
    └── comparison_<timestamp>.json # Model comparison results
```

## No Evaluation
This module does NOT:
- Calculate scores
- Evaluate correctness
- Judge quality
- Compare to ground truth

It ONLY:
- Takes text + image
- Calls video model
- Saves output video
- Logs the run
