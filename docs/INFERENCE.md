# VMEvalKit Inference Module

A powerful, production-ready inference system for running video generation models at scale. Features unified API access to 40+ models, automatic error handling, intelligent resume capability, and structured output management for reproducible research.

## 🚀 Quick Start

```python
from vmevalkit.runner.inference import InferenceRunner

# Initialize runner - creates structured output directories
runner = InferenceRunner(output_dir="data/outputs")

# Generate video showing reasoning process
result = runner.run(
    model_name="luma-ray-2",
    image_path="data/questions/maze_task/maze_0000/first_frame.png",
    text_prompt="Navigate the green dot through the maze corridors to reach the red flag",
    question_data={"id": "maze_0000", "domain": "maze"}  # Optional metadata
)

# Each inference creates a self-contained output folder:
print(f"📁 Output folder: {result['inference_dir']}")
# Contains:
# ├── video/generated_video.mp4    # The generated video
# ├── question/                    # Input data archive
# │   ├── first_frame.png         # Input image
# │   ├── prompt.txt              # Text prompt
# │   └── question_metadata.json  # Task metadata
# └── metadata.json               # Complete inference record
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

## 🎬 Supported Models

VMEvalKit provides unified access to **40 video generation models** across **11 provider families**:

### Commercial APIs (32 models)

| Provider | Models | Key Features | API Required |
|----------|---------|-------------|--------------|
| **Luma Dream Machine** | 2 | `luma-ray-2`, `luma-ray-flash-2` | `LUMA_API_KEY` |
| **Google Veo** | 3 | `veo-2.0-generate`, `veo-3.0-generate`, `veo-3.0-fast-generate` | GCP credentials |
| **Google Veo 3.1** | 4 | Native 1080p, audio generation (via WaveSpeed) | `WAVESPEED_API_KEY` |
| **WaveSpeed WAN 2.1** | 8 | 480p/720p variants with LoRA and ultra-fast options | `WAVESPEED_API_KEY` |
| **WaveSpeed WAN 2.2** | 10 | Enhanced 5B models, improved quality | `WAVESPEED_API_KEY` |
| **Runway ML** | 3 | Gen-3A Turbo, Gen-4 Turbo, Gen-4 Aleph | `RUNWAY_API_SECRET` |
| **OpenAI Sora** | 2 | Sora-2, Sora-2-Pro (4s/8s/12s durations) | `OPENAI_API_KEY` |

### Open-Source Models (9 models)

| Provider | Models | Key Features | Hardware Requirements |
|----------|---------|-------------|----------------------|
| **LTX-Video** | 3 | 2B/13B variants, real-time generation | GPU with 16GB+ VRAM |
| **HunyuanVideo** | 1 | High-quality 720p I2V | GPU with 24GB+ VRAM |
| **VideoCrafter** | 1 | Text-guided video synthesis | GPU with 16GB+ VRAM |
| **DynamiCrafter** | 3 | 256p/512p/1024p, image animation | GPU with 12-24GB VRAM |
| **Morphic** | 1 | Frames-to-video interpolation using Wan2.2 | 8 GPUs (distributed), requires Wan2.2 weights |

**✨ Key Capabilities:**
- All models support **image + text → video** generation
- Unified interface through `ModelWrapper` base class
- Dynamic loading - models initialized only when needed
- Automatic retry logic for API failures
- S3 upload support for models requiring image URLs

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
- This organization makes it trivial to:
  - Add a new commercial model (just code wrapper/service, update catalog)
  - Integrate new open-source models (add wrapper, point catalog)
  - Avoid any circular dependencies or bloat at startup


### Component Breakdown

**1. MODEL_CATALOG.py** - Pure Registry
```python
# No imports of model implementations!
AVAILABLE_MODELS = {
    "luma-ray-2": {
        "wrapper_module": "vmevalkit.models.luma_inference",
        "wrapper_class": "LumaWrapper",
        "service_class": "LumaInference",
        "model": "ray-2",
        "family": "Luma Dream Machine"
    },
    # ... 39 more models
}
```

**2. Dynamic Loading System**
```python
def _load_model_wrapper(model_name: str) -> Type[ModelWrapper]:
    config = AVAILABLE_MODELS[model_name]
    # Dynamic import at runtime
    module = importlib.import_module(config["wrapper_module"])
    wrapper_class = getattr(module, config["wrapper_class"])
    return wrapper_class
```

**3. Wrapper/Service Pattern**
- **Service**: Handles API calls and model-specific logic
- **Wrapper**: Adapts service to unified VMEvalKit interface
```python
class LumaWrapper(ModelWrapper):  # Unified interface
    def __init__(self, model: str, output_dir: str, **kwargs):
        self.luma_service = LumaInference(model, **kwargs)  # Specific implementation
    
    def generate(self, image_path, text_prompt, **kwargs):
        return self.luma_service.generate(...)  # Delegates to service
```

### Key Advantages

| Feature | Benefit |
|---------|---------|
| **No Circular Dependencies** | MODEL_CATALOG has no imports of model code |
| **Lazy Loading** | Models loaded only when actually used |
| **Easy Extension** | Add models by updating catalog + adding implementation |
| **Consistent Interface** | All models expose same `generate()` method |
| **Separation of Concerns** | Registry, orchestration, and implementation are isolated |

## 📂 Structured Output System

### Output Directory Hierarchy

VMEvalKit creates a **multi-level directory structure** mirroring the question organization:

```
data/outputs/
├── {domain}_task/                    # Task domain (e.g., maze_task)
│   └── {task_id}/                   # Individual task (e.g., maze_0000)
│       └── {run_id}/                # Unique run identifier
│           ├── video/
│           │   └── generated_video.mp4
│           ├── question/
│           │   ├── first_frame.png
│           │   ├── final_frame.png  
│           │   ├── prompt.txt
│           │   └── question_metadata.json
│           └── metadata.json

# Real example:
data/outputs/
├── maze_task/
│   └── maze_0000/
│       └── luma-ray-2_maze_0000_20250103_143025/
│           ├── video/
│           │   └── generated_video.mp4
│           ├── question/
│           │   ├── first_frame.png
│           │   ├── final_frame.png
│           │   ├── prompt.txt
│           │   └── question_metadata.json
│           └── metadata.json
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
    "video_path": "data/outputs/maze_task/maze_0000/luma-ray-2_maze_0000_20250103_143025/video/generated_video.mp4",
    "generation_id": "abc123-def456",  # Provider-specific ID
    "video_url": "https://..."         # If using cloud storage
  },
  "paths": {
    "inference_dir": "data/outputs/maze_task/maze_0000/luma-ray-2_maze_0000_20250103_143025",
    "video_dir": "data/outputs/maze_task/maze_0000/luma-ray-2_maze_0000_20250103_143025/video",
    "question_dir": "data/outputs/maze_task/maze_0000/luma-ray-2_maze_0000_20250103_143025/question"
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

### Benefits of This Structure

| Aspect | Benefit |
|--------|---------|
| **Reproducibility** | All inputs and outputs preserved together |
| **Batch Analysis** | Easy to process results programmatically |
| **Resume Capability** | Directory presence indicates completion |
| **Version Control** | Each run has unique timestamp |
| **Evaluation Ready** | Structured for downstream evaluation pipelines |

## Running Experiments

### Basic Usage

Generate dataset and run experiments:

```bash
cd /Users/access/VMEvalKit
source venv/bin/activate

# Generate dataset (if needed)
python -m vmevalkit.runner.create_dataset --pairs-per-domain 15

# Run experiment (1 task per domain for testing)
python examples/experiment_2025-10-14.py

# Run all tasks
python examples/experiment_2025-10-14.py --all-tasks
```

### Automatic Resume

The experiment script includes automatic resume capability:

**Features:**
- 🔄 Sequential execution: one model at a time, one task at a time
- ✅ Automatic skip of completed tasks
- 🎯 Selective model execution
- 📁 Directory-based completion tracking

**Usage:**

```bash
# Run all tasks (automatically skips completed ones)
python examples/experiment_2025-10-14.py --all-tasks

# Run specific models only
python examples/experiment_2025-10-14.py --all-tasks --only-model veo-3.0-generate

# Run multiple specific models
python examples/experiment_2025-10-14.py --all-tasks --only-model veo-3.0-generate luma-ray-2
```

**Command Options:**

| Option | Description |
|--------|-------------|
| `--all-tasks` | Run all tasks instead of 1 per domain |
| `--only-model [MODEL ...]` | Run only specified models (others skipped) |

**How It Works:**
- Automatically detects existing output directories
- Skips tasks that already have successful inference results
- To retry failed tasks: manually delete their output directories
- No separate checkpoint files - uses directory presence for tracking

## 💻 Python API

### InferenceRunner Class

The main orchestration class for video generation:

```python
from vmevalkit.runner.inference import InferenceRunner

# Initialize runner
runner = InferenceRunner(
    output_dir="data/outputs"  # Base directory for all outputs
)

# Note: The runner creates a structured folder hierarchy:
# output_dir/
# ├── {domain}_task/
# │   └── {task_id}/
# │       └── {model}_{task}_{timestamp}/
# │           ├── video/
# │           ├── question/
# │           └── metadata.json
```

### Running Single Inference

```python
# Basic usage
result = runner.run(
    model_name="luma-ray-2",
    image_path="path/to/image.png",
    text_prompt="Your prompt here"
)

# With complete task metadata (recommended)
result = runner.run(
    model_name="luma-ray-2",
    image_path="data/questions/maze_task/maze_0000/first_frame.png",
    text_prompt="Navigate through the maze",
    run_id="custom_run_id",  # Optional: custom ID instead of auto-generated
    question_data={           # Optional: preserves full task context
        "id": "maze_0000",
        "domain": "maze",
        "difficulty": "medium",
        "final_image_path": "path/to/final_frame.png",
        # Any additional metadata
    }
)

# Result dictionary contains:
print(result.keys())
# - run_id: Unique identifier for this run
# - inference_dir: Path to output folder
# - video_path: Path to generated video (if successful)
# - status: "success" or "failed"
# - error: Error message (if failed)
# - timestamp: ISO format timestamp
# - model: Model name used
# - duration_seconds: Generation time
```

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

### Direct Model Invocation

For lower-level control, use the `run_inference` function:

```python
from vmevalkit.runner.inference import run_inference

result = run_inference(
    model_name="runway-gen4-turbo",
    image_path="image.png",
    text_prompt="Animate this scene",
    output_dir="custom_output",
    question_data={"custom": "metadata"},
    # Model-specific kwargs passed through
    duration=10,  # Runway supports 5s or 10s
    seed=42
)

## Model-Specific Configuration

### API Keys

Set up API keys in `.env` file:

```bash
# Commercial APIs
LUMA_API_KEY=your_key_here
WAVESPEED_API_KEY=your_wavespeed_api_key
RUNWAY_API_SECRET=your_runway_secret
OPENAI_API_KEY=your_openai_key

# AWS for S3 storage (optional)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=vmevalkit
AWS_DEFAULT_REGION=us-east-2
```

### Model-Specific Parameters

Models accept parameters via **kwargs** passed directly to the generate method:

```python
# Luma models - supports duration and aspect ratio
result = runner.run(
    model_name="luma-ray-2",
    image_path="image.png",
    text_prompt="prompt",
    duration=8,           # Video duration (seconds)
    aspect_ratio="16:9",  # or "1:1", "9:16"
    enhance_prompt=True,  # AI prompt enhancement
    loop=False           # Loop video
)

# Runway models - supports 5s or 10s duration
result = runner.run(
    model_name="runway-gen4-turbo",
    image_path="image.png",
    text_prompt="prompt",
    duration=10,  # 5 or 10 seconds only
    seed=42      # For reproducibility
)

# Open-source models - resolution and frame settings
result = runner.run(
    model_name="ltx-video-13b-distilled",
    image_path="image.png", 
    text_prompt="prompt",
    height=512,
    width=512,
    num_frames=16,  # Number of frames to generate
    seed=42
)

# Morphic - frames-to-video interpolation (requires final_image_path in question_data)
result = runner.run(
    model_name="morphic-frames-to-video",
    image_path="first_frame.png",
    text_prompt="prompt",
    question_data={
        "id": "task_001",
        "final_image_path": "final_frame.png"  # Required for Morphic
    },
    size="1280*720",      # Video size
    frame_num=81,         # Number of frames
    nproc_per_node=8      # Number of GPUs
)
```

## 📋 Model Catalog Management

### Listing Available Models

```python
from vmevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS, MODEL_FAMILIES

# List all available models
print(f"Total models: {len(AVAILABLE_MODELS)}")
for model_name in sorted(AVAILABLE_MODELS.keys()):
    config = AVAILABLE_MODELS[model_name]
    print(f"  {model_name}: {config['description']}")

# List models by family
for family_name, models in MODEL_FAMILIES.items():
    print(f"\n{family_name} ({len(models)} models):")
    for model_name in models:
        print(f"  - {model_name}")

# Get model configuration
model_config = AVAILABLE_MODELS["luma-ray-2"]
print(f"Module: {model_config['wrapper_module']}")
print(f"Class: {model_config['wrapper_class']}")
print(f"Family: {model_config['family']}")
```

### Model Selection Helper

```python
def find_models_by_capability(capability):
    """Find models matching a capability."""
    matching = []
    
    for name, config in AVAILABLE_MODELS.items():
        desc = config.get("description", "").lower()
        if capability.lower() in desc:
            matching.append((name, config["family"]))
    
    return matching

# Find all 720p capable models
high_res_models = find_models_by_capability("720p")
print(f"Models with 720p support: {len(high_res_models)}")
for model, family in high_res_models:
    print(f"  {model} ({family})")
```

## 🔧 Error Handling & Recovery

### Error Handling Architecture

VMEvalKit implements **multi-level error handling**:

1. **Model Level** - Individual wrappers handle API-specific errors
2. **Runner Level** - InferenceRunner catches and logs failures
3. **Experiment Level** - Batch scripts skip failed tasks

### Retry Logic

Some models implement automatic retry (e.g., Luma):
```python
# In LumaInference - uses tenacity library
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def _create_generation(self, image_url, text_prompt):
    # API call with automatic retry on failure
```

### Resume Capability

The system uses **directory-based tracking** for resume:

```python
# Check if task already completed
task_folder = output_dir / domain_task / task_id
existing_runs = list(task_folder.glob(f"{model_name}_{task_id}_*"))

if existing_runs:
    print(f"Skipping completed task: {task_id}")
    continue  # Skip to next task

# To retry a failed task:
# 1. Delete its output directory
import shutil
shutil.rmtree("data/outputs/maze_task/maze_0000/luma-ray-2_maze_0000_20250101_120000")
# 2. Re-run the experiment
```

### Error Information

Failed inferences create a metadata.json with error details:
```json
{
  "inference": {
    "status": "failed",
    "error": "API rate limit exceeded",
    "timestamp": "2025-01-01T12:00:00Z",
    "model": "luma-ray-2"
  },
  "paths": {
    "inference_dir": "data/outputs/maze_task/maze_0000/luma-ray-2_maze_0000_20250101_120000"
  }
}
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

### Performance Optimization

| Tip | Description | Impact |
|-----|-------------|--------|
| **Start Small** | Test with 1 task per domain first | Saves API costs, validates setup |
| **Use --only-model** | Test specific models during development | Faster iteration |
| **Batch by Model** | Process all tasks for one model before switching | Better rate limit management |
| **Monitor Costs** | Track API usage for commercial models | Budget control |
| **Leverage Resume** | Don't restart from scratch after failures | Time savings |

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

### Adding Custom Models

See [ADDING_MODELS.md](ADDING_MODELS.md) for the complete guide. Quick overview:

1. Add entry to MODEL_CATALOG.py
2. Create wrapper class inheriting from ModelWrapper
3. Implement generate() method
4. No changes needed in core files!

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

### Hooks and Callbacks

```python
# Process results after each inference
def process_result(result):
    if result["status"] == "success":
        # Your custom logic here
        print(f"Video duration: {result.get('duration_seconds')}")

# Use in batch processing
for task in tasks:
    result = runner.run(...)
    process_result(result)
```

## 📖 Related Documentation

| Guide | Description |
|-------|-------------|
| [ADDING_MODELS.md](ADDING_MODELS.md) | Complete guide to adding new video models |
| [EVALUATION.md](EVALUATION.md) | Human and AI evaluation pipelines |
| [WEB_DASHBOARD.md](WEB_DASHBOARD.md) | Interactive results visualization |
| [DATA_MANAGEMENT.md](DATA_MANAGEMENT.md) | Dataset organization and versioning |
| [ADDING_TASKS.md](ADDING_TASKS.md) | Creating new reasoning tasks |

## 📋 Quick Reference

### Essential Commands

```bash
# Generate dataset
python -m vmevalkit.runner.create_dataset --pairs-per-domain 15

# Run quick test (1 task per domain)
python examples/experiment_2025-10-14.py

# Run full experiment
python examples/experiment_2025-10-14.py --all-tasks

# Run specific models only
python examples/experiment_2025-10-14.py --all-tasks --only-model luma-ray-2 veo-3.0-generate

# List available models
python -c "from vmevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS; print(list(AVAILABLE_MODELS.keys()))"
```

### Key File Locations

```
VMEvalKit/
├── .env                              # API keys configuration
├── vmevalkit/
│   ├── runner/
│   │   ├── MODEL_CATALOG.py        # All model definitions
│   │   └── inference.py            # InferenceRunner class
│   └── models/                      # Model implementations
├── data/
│   ├── questions/                  # Input task pairs
│   └── outputs/                    # Generated videos
└── examples/
    └── experiment_2025-10-14.py    # Main experiment script
```

### API Quick Reference

```python
# Initialize
from vmevalkit.runner.inference import InferenceRunner
runner = InferenceRunner(output_dir="data/outputs")

# Run inference
result = runner.run(
    model_name="model-name",         # Required: from MODEL_CATALOG
    image_path="path/to/image.png",  # Required: input image
    text_prompt="prompt text",        # Required: instructions
    run_id="custom_id",              # Optional: custom run ID
    question_data={...},             # Optional: task metadata
    **kwargs                         # Model-specific parameters
)

# Check result
if result["status"] == "success":
    print(f"Video: {result['video_path']}")
else:
    print(f"Error: {result['error']}")
```

### Model Naming Convention

| Pattern | Example | Meaning |
|---------|---------|---------|
| `provider-model-variant` | `luma-ray-2` | Provider + model + version |
| `provider-model-resolution` | `veo-3.1-720p` | With resolution variant |
| `provider-family-version-feature` | `wavespeed-wan-2.2-i2v-720p-lora` | Full specification |

### Environment Variables

```bash
# Commercial APIs
LUMA_API_KEY=xxx
WAVESPEED_API_KEY=xxx
RUNWAY_API_SECRET=xxx
OPENAI_API_KEY=xxx

# Google Cloud (for Veo)
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json
GCP_PROJECT_ID=your-project-id

# AWS S3 (optional)
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
S3_BUCKET=vmevalkit
AWS_DEFAULT_REGION=us-east-2

# Morphic Frames-to-Video (open-source)
MORPHIC_WAN2_CKPT_DIR=./Wan2.2-I2V-A14B
MORPHIC_LORA_WEIGHTS_PATH=./morphic-frames-lora-weights/lora_interpolation_high_noise_final.safetensors
MORPHIC_NPROC_PER_NODE=8
```

### Success Metrics

- ✅ **Successful inference**: Video file exists in output directory
- 📁 **Resume detection**: Directory exists = task completed
- ❌ **Failed inference**: metadata.json contains error details
- ⏭️ **Skipped task**: Existing output directory found

---

*VMEvalKit Inference Module - Version 1.0*
