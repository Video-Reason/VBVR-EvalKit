# VMEvalKit Usage Guide

This guide shows how to use VMEvalKit's inference module for video generation.

## Installation

```bash
# From source
git clone https://github.com/yourusername/VMEvalKit.git
cd VMEvalKit
pip install -e .

# Or direct pip install (when available)
pip install vmevalkit
```

## Python API Usage

### Single Inference

```python
from vmevalkit import InferenceRunner

# Initialize runner
runner = InferenceRunner(output_dir="./outputs")

# Method 1: Direct inference with image and prompt
result = runner.run(
    model_name="luma-dream-machine",
    image_path="path/to/maze.png",
    text_prompt="Show the solution path through this maze",
    duration=5.0,
    resolution=(512, 512)
)

print(f"Video saved to: {result['video_path']}")
```

### From Task File

```python
# Method 2: Run from a task definition
task = {
    "id": "maze_001",
    "prompt": "Solve this maze from start to finish",
    "first_image_path": "data/mazes/maze_001.png"
}

result = runner.run_from_task(
    model_name="luma-dream-machine",
    task_data=task
)
```

### Batch Inference

```python
from vmevalkit import BatchInferenceRunner

# Initialize batch runner
batch_runner = BatchInferenceRunner(
    output_dir="./outputs",
    max_workers=4  # For parallel processing
)

# Run on entire dataset
results = batch_runner.run_dataset(
    model_name="luma-dream-machine",
    dataset_path="data/maze_tasks.json",
    max_tasks=10,  # Limit number of tasks
    duration=5.0,
    resolution=(512, 512)
)

print(f"Processed {results['successful']} videos successfully")
```

### Model Comparison

```python
# Compare multiple models on the same dataset
comparison = batch_runner.run_models_comparison(
    model_names=["luma-dream-machine", "google-veo-001"],
    dataset_path="data/maze_tasks.json",
    duration=5.0
)
```

## Command Line Usage

VMEvalKit provides a CLI through the `vmevalkit` command:

### Single Inference

```bash
# From a task file
vmevalkit inference luma-dream-machine \
    --task-file data/maze_task.json \
    --duration 5.0 \
    --resolution 512x512

# Direct image + prompt
vmevalkit inference luma-dream-machine \
    --image path/to/maze.png \
    --prompt "Solve this maze step by step" \
    --output-dir ./my_outputs
```

### Task Selection from Dataset

```bash
# Run specific task by ID
vmevalkit inference luma-dream-machine \
    --task-file data/maze_tasks.json \
    --task-id irregular_0001

# Run task by index
vmevalkit inference luma-dream-machine \
    --task-file data/maze_tasks.json \
    --task-index 5
```

### Batch Processing

```bash
# Process entire dataset
vmevalkit batch luma-dream-machine \
    --dataset data/maze_tasks.json \
    --workers 4

# Process specific tasks
vmevalkit batch luma-dream-machine \
    --dataset data/maze_tasks.json \
    --task-ids task_001 task_002 task_003

# Limit number of tasks
vmevalkit batch luma-dream-machine \
    --dataset data/maze_tasks.json \
    --max-tasks 5
```

### Multi-Model Comparison

```bash
# Compare multiple models
vmevalkit batch luma-dream-machine google-veo-001 runway-gen3 \
    --dataset data/maze_tasks.json \
    --duration 5.0 \
    --resolution 1024x1024
```

## Environment Configuration

Create a `.env` file with your API keys:

```bash
# Luma Dream Machine
LUMA_API_KEY=your_luma_key

# Google Veo
GOOGLE_VEO_API_KEY=your_veo_key

# AWS (for S3 image hosting)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_DEFAULT_REGION=us-east-2
S3_BUCKET=your-bucket-name
```

## Task File Format

### Single Task
```json
{
    "id": "maze_001",
    "prompt": "Navigate through the maze from the green start to red exit",
    "first_image_path": "data/mazes/maze_001.png"
}
```

### Dataset Format
```json
{
    "dataset": "maze_reasoning",
    "version": "1.0",
    "pairs": [
        {
            "id": "irregular_0001",
            "prompt": "Show the solution path through this maze",
            "first_image_path": "data/mazes/irregular_0001.png",
            "solution_path": "optional/path/to/solution.png"
        },
        {
            "id": "irregular_0002",
            "prompt": "Trace the path from start to finish",
            "first_image_path": "data/mazes/irregular_0002.png"
        }
    ]
}
```

## Output Structure

```
outputs/
├── luma_<generation_id>.mp4        # Generated videos
├── inference_runs.json             # Log of all inference runs
└── batch_results/
    ├── batch_luma_20240115_120000.json     # Batch run results
    └── comparison_20240115_130000.json      # Model comparison results
```

## Model Support

Currently supported models:
- `luma-dream-machine` - Luma AI Dream Machine
- `google-veo-001` - Google Veo (coming soon)
- `runway-gen3` - Runway Gen-3 (coming soon)

Check available models:
```python
from vmevalkit import ModelRegistry
print(ModelRegistry.list_models())
```

## Advanced Usage

### Custom Generation Parameters

```python
result = runner.run(
    model_name="luma-dream-machine",
    image_path="maze.png",
    text_prompt="Solve the maze",
    duration=7.0,           # Longer video
    fps=30,                 # Higher frame rate
    resolution=(1024, 1024), # Square HD
    enhance_prompt=False,    # Don't modify prompt
    loop_video=False        # Don't loop
)
```

### Error Handling

```python
try:
    result = runner.run(
        model_name="luma-dream-machine",
        image_path="maze.png",
        text_prompt="Solve the maze"
    )
    
    if result["status"] == "success":
        print(f"Success! Video: {result['video_path']}")
    else:
        print(f"Failed: {result['error']}")
        
except Exception as e:
    print(f"Error during inference: {e}")
```

### Parallel Batch Processing

```python
# Use multiple workers for faster processing
batch_runner = BatchInferenceRunner(
    output_dir="./outputs",
    max_workers=8  # Process 8 videos in parallel
)

# Note: Be mindful of API rate limits
```

## Tips

1. **Image Hosting**: Some models (like Luma) require images to be accessible via HTTP. VMEvalKit handles this automatically using S3.

2. **Rate Limits**: Most APIs have rate limits. The batch runner includes automatic delays between requests.

3. **Resolution**: Use appropriate resolution for your task. Mazes work well at 512x512, while other tasks may need higher resolution.

4. **Debugging**: Set verbose mode for detailed output:
   ```python
   model = ModelRegistry.load_model("luma-dream-machine", verbose=True)
   ```

5. **Task IDs**: Use meaningful task IDs for easier tracking and debugging.

## Need Help?

- Check the [examples/](examples/) directory for complete examples
- See model-specific documentation in [vmevalkit/api_clients/](vmevalkit/api_clients/)
- Report issues on GitHub
