# VMEvalKit Inference Module

## 🚀 Quick Start

```python
python run.py configs/demo.yaml
```

## 📚 Core Concepts

### Task Pairs: The Evaluation Unit

VMEvalKit evaluates video models' reasoning capabilities through **Task Pairs** - carefully designed visual reasoning problems:

| Component | File | Purpose | Sent to Model |
|-----------|------|---------|---------------|
| 📸 **Initial State** | `first_frame.png` | Problem/puzzle to solve | ✅ Yes |
| 🎯 **Final State** | `final_frame.png` | Solution/goal reference | ❌ No |
| 📝 **Text Prompt** | `prompt.txt` | Natural language instructions | ✅ Yes |
| 📊 **Metadata** | `question_metadata.json` | Difficulty, parameters, ground truth | ❌ No |

**Directory Structure:**
```
data/questions/
├── chess_task/
│   ├── chess_0000/
│   │   ├── first_frame.png      # Chess position
│   │   ├── final_frame.png      # After checkmate
│   │   ├── prompt.txt           # "Find checkmate in one move"
│   │   └── question_metadata.json
│   └── chess_0001/...
├── maze_task/...
├── raven_task/...
├── rotation_task/...
└── sudoku_task/...
```

Models receive the initial state + prompt and must generate videos demonstrating the reasoning process to reach the final state.

## 🏗️ Architecture

### System Design

VMEvalKit uses a **three-layer modular architecture** that cleanly supports both commercial (closed-source) APIs and open-source video models—enabling seamless scaling, easy model addition, and clear separation of concerns.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               InferenceRunner                              │
│        Top-level orchestrator: manages workflow, batching, and output      │
└───────────────────────┬─────────────────────────────────────────────────────┘
                        │      Dynamic Model Loading (importlib)              
                        ▼                                                    
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODEL_CATALOG                                 │
│  Unified model registry:                                                   │
│    - Lists all available models (both API and open-source)                 │
│    - Records provider family, wrapper paths, model meta-info               │
│    - No imports of implementations (pure config)                           │
└───────────────────────┬─────────────────────────────────────────────────────┘
                        │      importlib.import_module() dynamically loads   
                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Model Implementations (Two Flavors)                   │
│ ┌────────────────────────────┬────────────────────────────────────────────┐ │
│ │          Commercial Models             │      Open-Source Models        │ │
│ │       (Closed Source Services)         │    (Local Implementations)     │ │
│ ├────────────────────────────┼────────────────────────────────────────────┤ │
│ │ LumaWrapper  +  LumaService           │ LTXVideoWrapper  +  LTXService  │ │
│ │ VeoWrapper   +  VeoService            │ HunyuanWrapper   +  HunyuanSvc  │ │
│ │ RunwayWrapper+  RunwayService         │ VideoCrafterWrapper+VCService   │ │
│ │ ...                                   │ DynamiCrafterWrapper+DynService │ │
│ │                                       │ MorphicWrapper   +  MorphicSvc  │ │
│ └────────────────────────────┴────────────────────────────────────────────┘ │
│   - Each Wrapper implements unified VMEvalKit interface                     │
│   - API Services handle endpoints, retries, S3-upload (when needed)         │
│   - Open-source backends directly invoke local model code                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Key Points:**

- **MODEL_CATALOG** lists both API-based (closed-source) and open-source models in one place. Each model specifies its provider, class paths, and type (`"api"` or `"open_source"`).
- **Dynamic loading** means only the requested model's code is ever imported—no slow startup for unused models.
- **Wrappers** for APIs and open-source models both inherit from `ModelWrapper` (or equivalent) and expose a common `.generate()` interface. API wrappers talk to services handling REST calls (with retry logic, S3 upload, etc), while open-source wrappers call local PyTorch/Tensorflow code.

## 📂 Structured Output System

### Output Directory Hierarchy

VMEvalKit creates a **multi-level directory structure** for organized experiment management:

```
data/outputs/
└── {experiment_name}/               # Experiment (e.g., "pilot_experiment")
    └── {model_name}/                # Model (e.g., "luma-ray-2")
        └── {domain}_task/           # Task domain (e.g., "maze_task")
            └── {task_id}/           # Individual task (e.g., "maze_0000")
                └── {run_id}/        # Unique run identifier
                    ├── video/
                    │   └── generated_video.mp4
                    ├── question/
                    │   ├── first_frame.png
                    │   ├── final_frame.png  
                    │   ├── prompt.txt
                    │   └── question_metadata.json
                    └── metadata.json

# Real example:
data/outputs/
└── pilot_experiment/
    └── luma-ray-2/
        └── maze_task/
            └── maze_0000/
                └── luma-ray-2_maze_0000_20250103_143025/
                    ├── video/
                    │   └── generated_video.mp4
                    ├── question/
                    │   ├── first_frame.png
                    │   ├── final_frame.png
                    │   ├── prompt.txt
                    │   └── question_metadata.json
                    └── metadata.json
```

### Metadata Structure

The `metadata.json` file contains comprehensive inference information:

```json
{
  "inference": {
    "run_id": "luma-ray-2_maze_0000_20250103_143025",
    "model": "luma-ray-2",
    "timestamp": "2025-01-03T14:30:25.123456",
    "status": "success",
    "duration_seconds": 45.2,
    "error": null
  },
  "input": {
    "prompt": "Navigate the green dot through the maze...",
    "image_path": "data/questions/maze_task/maze_0000/first_frame.png",
    "question_id": "maze_0000",
    "task_category": "maze"
  },
  "output": {
    "video_path": "data/outputs/pilot_experiment/luma-ray-2/maze_task/maze_0000/luma-ray-2_maze_0000_20250103_143025/video/generated_video.mp4",
    "generation_id": "abc123-def456",  # Provider-specific ID
    "video_url": "https://..."         # If using cloud storage
  },
  "paths": {
    "inference_dir": "data/outputs/pilot_experiment/luma-ray-2/maze_task/maze_0000/luma-ray-2_maze_0000_20250103_143025",
    "video_dir": "data/outputs/pilot_experiment/luma-ray-2/maze_task/maze_0000/luma-ray-2_maze_0000_20250103_143025/video",
    "question_dir": "data/outputs/pilot_experiment/luma-ray-2/maze_task/maze_0000/luma-ray-2_maze_0000_20250103_143025/question"
  },
  "question_data": {
    // Complete original question metadata
    "id": "maze_0000",
    "domain": "maze",
    "difficulty": "medium",
    "maze_size": [10, 10],
    "solution_length": 23,
    // ... more task-specific data
  }
}
```



## Running Experiments


## 💻 Python API


### Batch Processing

```python
# Process a dataset of tasks
tasks = [
    {"id": "chess_0001", "image": "chess_0001/first_frame.png", "prompt": "Find checkmate"},
    {"id": "maze_0002", "image": "maze_0002/first_frame.png", "prompt": "Solve the maze"},
]

results = []
for task in tasks:
    try:
    result = runner.run(
        model_name="veo-3.0-generate",
        image_path=task["image"],
        text_prompt=task["prompt"],
            question_data={"id": task["id"], "domain": task.get("domain")}
        )
        results.append(result)
        
        if result.get("status") == "failed":
            print(f"❌ Failed {task['id']}: {result.get('error')}")
        else:
            print(f"✅ Completed {task['id']}: {result['video_path']}")
    except Exception as e:
        print(f"❌ Error processing {task['id']}: {e}")

# Summary statistics
successful = sum(1 for r in results if r.get("status") != "failed")
print(f"Completed {successful}/{len(tasks)} tasks successfully")
```


### API Keys

Set up API keys in `.env` file:

```bash
# Commercial APIs
LUMA_API_KEY=your_key_here
WAVESPEED_API_KEY=your_wavespeed_api_key
RUNWAYML_API_SECRET=your_runway_secret
OPENAI_API_KEY=your_openai_key

# AWS for S3 storage (optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=vmevalkit
AWS_DEFAULT_REGION=us-east-2
```



### Common Error Types

| Error Type | Cause | Solution |
|------------|-------|----------|
| `FileNotFoundError` | Missing input image | Verify question dataset paths |
| `LumaAPIError` | API failures | Check API key and rate limits |
| `ValueError` | Unknown model name | Verify model name in MODEL_CATALOG |
| `ImportError` | Missing dependencies | Install required packages |
| SVG conversion | Some tasks use SVG | Auto-converts to PNG with cairosvg |

## 💡 Tips and Best Practices
### Quality Assurance

```python
# Validate outputs programmatically
import json
from pathlib import Path

def validate_inference(inference_dir):
    """Check if inference completed successfully."""
    metadata_file = Path(inference_dir) / "metadata.json"
    
    if not metadata_file.exists():
        return False, "No metadata file"
    
    with open(metadata_file) as f:
        metadata = json.load(f)
    
    if metadata["inference"]["status"] == "failed":
        return False, metadata["inference"]["error"]
    
    video_path = Path(inference_dir) / "video" / "generated_video.mp4"
    if not video_path.exists():
        return False, "No video file"
    
    return True, "Valid inference"
```

### S3 Integration

VMEvalKit includes **automatic S3 upload** for models requiring image URLs:

```python
# Configure S3 in .env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=vmevalkit

# S3ImageUploader handles uploads automatically
from vmevalkit.utils.s3_uploader import S3ImageUploader

uploader = S3ImageUploader()
image_url = uploader.upload_image("local_image.png")
# Returns: https://vmevalkit.s3.amazonaws.com/images/xxxxx.png
```

Models like Luma automatically use S3 when configured.

## 🛠️ Troubleshooting

### Common Issues and Solutions

**1. Model Not Found Error**
```
ValueError: Unknown model: model-name
```
**Solution:** Check available models in MODEL_CATALOG.py or use:
```python
from vmevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS
print(list(AVAILABLE_MODELS.keys()))
```

**2. SVG to PNG Conversion Issues**
```
PIL.UnidentifiedImageError: cannot identify image file
```
**Solution:** Install cairosvg for automatic conversion:
```bash
pip install cairosvg
```

**3. API Key Not Set**
```
KeyError: 'LUMA_API_KEY'
```
**Solution:** Set API keys in `.env` file:
```bash
cp env.template .env
# Edit .env with your keys
```

**4. GPU Memory Issues (Open-Source Models)**
```
torch.cuda.OutOfMemoryError
```
**Solution:** Use smaller models or reduce resolution:
```python
# Use 2B model instead of 13B
runner.run(model_name="ltx-video-2b-distilled", ...)

# Or reduce resolution
runner.run(..., height=256, width=256)
```

**5. Rate Limiting**
```
APIError: Rate limit exceeded
```
**Solution:** Add delays between requests or use retry logic:
```python
import time

for task in tasks:
    result = runner.run(...)
    time.sleep(5)  # Delay between API calls
```

### Debug Mode

Enable verbose output for debugging:
```python
# Some models support verbose mode
result = run_inference(
    model_name="luma-ray-2",
    image_path="test.png", 
    text_prompt="test",
    verbose=True  # Shows detailed progress
)
```

## 🔌 Extending the System

### Custom Output Processing

```python
from vmevalkit.runner.inference import InferenceRunner

class CustomRunner(InferenceRunner):
    def _save_metadata(self, inference_dir, result, question_data):
        # Add custom metadata fields
        super()._save_metadata(inference_dir, result, question_data)
        
        # Add your custom processing
        custom_file = inference_dir / "custom_analysis.json"
        with open(custom_file, 'w') as f:
            json.dump({"custom": "data"}, f)
```
