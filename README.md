# VMEvalKit 🎥🧠

**VMEvalKit** is a comprehensive evaluation framework for assessing reasoning capabilities in video generation models through cognitive and logical tasks.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🎯 Overview

VMEvalKit addresses a critical gap in evaluating modern video generation models beyond visual quality metrics. While existing benchmarks focus on photorealism and temporal consistency, VMEvalKit evaluates whether video models can demonstrate genuine reasoning capabilities by solving visual problems through generated video sequences.

### 🔴 Critical Requirement: Text + Image → Video

**VMEvalKit specifically requires models that accept BOTH:**
- 📸 **An input image** (the problem to solve: maze, chess board, etc.)
- 📝 **A text prompt** (instructions: "solve the maze", "show the next move", etc.)

Many models advertise "image-to-video" but only accept images WITHOUT text guidance. These are NOT suitable for reasoning evaluation as they cannot receive problem-solving instructions.

Our framework tests these specialized text+image→video models on fundamental reasoning tasks, requiring them to:
- Understand complex visual problems from the input image
- Follow text instructions to demonstrate solutions
- Generate videos that show logical problem-solving steps

## ✨ Key Features

- **🧩 Diverse Reasoning Tasks**: Evaluate models on maze solving, mental rotation, chess positions, and Raven's Progressive Matrices
- **🔌 Unified Interface**: Support for both open-source and closed-source video generation models
- **📊 Comprehensive Metrics**: Automatic evaluation of reasoning accuracy, solution correctness, and video quality
- **🚀 Easy Integration**: Simple API for adding new models and tasks
- **📈 Benchmarking Suite**: Compare performance across multiple models and tasks
- **🎨 Visualization Tools**: Generate detailed reports with video outputs and analysis

## 📋 Supported Tasks

### 1. **Maze Solving** 🌀
Tests spatial reasoning and pathfinding abilities. Models receive a maze image and must generate a video showing the solution path from start to finish.

### 2. **Mental Rotation** 🔄
Evaluates 3D spatial understanding. Models must generate videos showing objects rotating to match target orientations.

### 3. **Chess Puzzles** ♟️
Assesses strategic reasoning. Models generate videos demonstrating chess puzzle solutions with legal moves.

### 4. **Raven's Progressive Matrices** 🔲
Tests abstract pattern recognition. Models complete visual patterns by generating the missing sequence element.

## 🤖 Supported Models

VMEvalKit includes **4 fully implemented models** that support text+image→video generation for reasoning evaluation:

### ✅ Implemented Models

All models below accept BOTH an input image (the problem) AND a text prompt (instructions):

#### 1. **Luma Dream Machine**
- **Status**: ✅ Fully Implemented
- **API**: Requires `LUMA_API_KEY`
- **Notes**: Supports text prompts with image references for guided video generation

#### 2. **Google Veo v1 (veo-001)**
- **Status**: ✅ Fully Implemented  
- **API**: Requires Google Cloud credentials
- **Notes**: High-quality text+image→video generation via Vertex AI

#### 3. **Google Veo v2 (veo-002)**
- **Status**: ✅ Fully Implemented
- **API**: Requires Google Cloud credentials
- **Notes**: Latest version with improved quality and temporal consistency

#### 4. **Runway Gen4-Aleph**
- **Status**: ✅ Fully Implemented (with automatic workaround)
- **API**: Requires `RUNWAY_API_KEY`
- **Notes**: Uses video-to-video with text prompts. VMEvalKit automatically converts static images to video format for compatibility

### 🔍 Key Requirements

For VMEvalKit's reasoning tasks, all models MUST accept:
- 📸 **An input image** (e.g., maze, chess board, visual puzzle)
- 📝 **A text prompt** (e.g., "Solve this maze", "Show the next move")

All 4 implemented models meet these requirements and can evaluate reasoning capabilities through video generation.

## 🎯 Model Selection Guide

### Recommended Models for VMEvalKit Tasks

| **Use Case**                          | **Verified Models**                                                                              |
|---------------------------------------|--------------------------------------------------------------------------------------------------|
| **Production-ready APIs**             | Pika 2.0+, Luma Dream Machine, Genmo Mochi                                                    |
| **Enterprise solutions**              | Google Imagen Video, Stability AI Video                                                        |
| **Advanced capabilities**             | Genmo Mochi, Luma Dream Machine                                                                |
| **Best text+image support**           | Pika 2.0+, Luma Dream Machine, Genmo Mochi                                                    |

### Critical Capabilities for Reasoning Tasks

| **Required Feature**              | **Why It's Essential**                                                                |
|----------------------------------|----------------------------------------------------------------------------------------|
| Text prompt input                | Provides instructions like "solve the maze" or "show the next chess move"            |
| Image conditioning               | Preserves the problem visual (maze layout, chess position, etc.)                      |
| Temporal coherence               | Maintains consistency across frames to show logical progression                        |
| Sufficient video length          | Generates enough frames to demonstrate complete solutions                             |

### ⚠️ Model Verification Checklist

Before integrating a model, verify:
- [ ] Accepts both `image` and `text`/`prompt` parameters
- [ ] Documentation explicitly mentions "text-conditioned image-to-video"
- [ ] API/code examples show both inputs being used together
- [ ] Output videos preserve input image content while following text instructions
- [ ] Sufficient video duration for reasoning tasks (minimum 2-4 seconds)

## 🚀 Installation

### Prerequisites
```bash
# Python 3.8 or higher required
python --version

# CUDA 11.8+ for GPU support (recommended)
nvidia-smi
```

### Install from Source
```bash
# Clone the repository
git clone https://github.com/yourusername/VMEvalKit.git
cd VMEvalKit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install VMEvalKit
pip install -e .
```

### Install via pip (coming soon)
```bash
pip install vmevalkit
```

### 🔧 Configuration

#### S3 Configuration (for image hosting)
VMEvalKit uses S3 for temporary image hosting when working with cloud-based video generation APIs. The default S3 bucket `vmevalkit` is located in `us-east-2`.

**Important**: If you encounter S3 authentication errors, ensure your AWS configuration matches the bucket region:
```bash
# The S3 bucket is in us-east-2
export AWS_DEFAULT_REGION=us-east-2  # or configure in ~/.aws/config
```

If you use a different S3 bucket, update the configuration in your `.env` file:
```bash
S3_BUCKET=your-bucket-name
AWS_DEFAULT_REGION=your-bucket-region
```

## 📖 Quick Start

### Basic Usage

```python
from vmevalkit import InferenceRunner

# Initialize inference runner
runner = InferenceRunner(output_dir="./outputs")

# Run video generation with image + text
result = runner.run(
    model_name="luma-dream-machine",
    image_path="data/maze.png",
    text_prompt="Navigate through the maze from the green start to the red end point",
    duration=5.0,
    resolution=(512, 512)
)

print(f"Generated video: {result['video_path']}")
print(f"Generation time: {result['duration']:.1f}s")
```

### Command Line Interface

```bash
# Single inference
vmevalkit inference luma-dream-machine \
    --image data/maze.png \
    --prompt "Solve this maze step by step"

# Batch processing on dataset
vmevalkit batch luma-dream-machine \
    --dataset data/maze_tasks.json \
    --workers 4

# Compare multiple models
vmevalkit batch luma-dream-machine google-veo-001 \
    --dataset data/tasks.json \
    --max-tasks 10
```

See [USAGE.md](USAGE.md) for comprehensive documentation.

### Batch Evaluation

```python
import os
from vmevalkit import VMEvaluator

# Evaluate closed-source API models that support text+image→video
api_models = [
    "pika-2.2",          # Pika's text+image video generation
    "luma-dream-machine", # Luma's text+image video generation
    "genmo-mochi",       # Genmo's multimodal video API
    "google-imagen",     # Google's text+image cascade model
    "stability-ai-video" # Stability's text-conditioned animation
]

# Configure API keys
api_keys = {
    "pika-2.2": os.getenv("PIKA_API_KEY"),
    "luma-dream-machine": os.getenv("LUMA_API_KEY"),
    "genmo-mochi": os.getenv("GENMO_API_KEY"),
    "google-imagen": os.getenv("GOOGLE_API_KEY"),
    "stability-ai-video": os.getenv("STABILITY_API_KEY")
}

tasks = ["maze_solving", "mental_rotation", "chess", "ravens_matrices"]

# Run benchmark with API models
benchmark_results = evaluator.run_benchmark(
    models=api_models,
    api_keys=api_keys,
    tasks=tasks,
    output_dir="./results",
    generate_report=True,
    strict_mode=True,  # Fail if model doesn't support text+image
    test_inputs={
        "image": "test_image.png",
        "prompt": "Test prompt for verification"
    }
)
```

## 📊 Evaluation Metrics

### Reasoning Metrics
- **Solution Accuracy**: Correctness of the demonstrated solution
- **Step Validity**: Logical consistency of intermediate steps
- **Completion Rate**: Percentage of successfully completed tasks
- **Planning Efficiency**: Optimality of the solution path

### Video Quality Metrics
- **Temporal Consistency**: Frame-to-frame coherence
- **Visual Clarity**: Sharpness and detail preservation
- **Motion Smoothness**: Natural movement patterns
- **Prompt Adherence**: Alignment with input instructions

## 🏗️ Project Structure

```
VMEvalKit/
├── vmevalkit/
│   ├── api_clients/    # API client implementations
│   ├── core/           # Core evaluation framework
│   ├── inference/      # Inference module (no evaluation)
│   ├── models/         # Model interfaces and wrappers
│   ├── tasks/          # Task definitions and datasets
│   ├── metrics/        # Evaluation metrics
│   ├── prompts/        # Prompt templates
│   ├── utils/          # Utility functions
│   └── cli.py          # Command-line interface
├── data/               # Generated datasets and tasks
├── examples/           # Example scripts and notebooks
├── tests/              # Unit tests
├── docs/               # Documentation
└── USAGE.md            # Comprehensive usage guide
```

## 🔧 Configuration

Create a `config.yaml` file to customize evaluation settings:

```yaml
evaluation:
  batch_size: 4
  num_workers: 8
  seed: 42
  enable_api_batching: true
  rate_limiting:
    max_requests_per_minute: 60
    retry_on_failure: true

models:
  # Implemented Models (All support Text+Image→Video)
  luma-dream-machine:
    api_key: "${LUMA_API_KEY}"
    duration: 5
    resolution: [1024, 576]
    
  google-veo-002:
    project_id: "${GCP_PROJECT_ID}"
    location: "us-central1"
    quality: "1080p"
    duration: 8
    
  runway-gen4-aleph:
    api_key: "${RUNWAY_API_KEY}"
    duration: 5
    auto_convert: true  # Converts image to video automatically

tasks:
  maze_solving:
    difficulty_levels: ["easy", "medium", "hard"]
    time_limit: 60
    grid_size: [10, 10]
    require_video_solution: true
    
  mental_rotation:
    object_types: ["3d_shapes", "molecular_structures"]
    rotation_degrees: [90, 180, 270]
    
  chess:
    puzzle_sources: ["lichess", "chess.com"]
    elo_range: [1200, 2000]
    
  ravens_matrices:
    matrix_types: ["2x2", "3x3"]
    pattern_complexity: ["basic", "advanced"]
```

## 📈 Benchmarking

Run comprehensive benchmarks using the inference module:

```bash
# Process entire dataset with single model
vmevalkit batch luma-dream-machine --dataset data/maze_tasks.json

# Compare multiple models on same dataset
vmevalkit batch luma-dream-machine google-veo-001 runway-gen3 \
    --dataset data/benchmark_tasks.json \
    --workers 4

# Process specific tasks only
vmevalkit batch luma-dream-machine \
    --dataset data/tasks.json \
    --task-ids task_001 task_002 task_003
```

## 🧪 Adding Custom Models

```python
from vmevalkit.models import BaseVideoModel

class MyCustomModel(BaseVideoModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize your model
    
    def generate(self, prompt, image=None, **kwargs):
        # Implement video generation logic
        return generated_video
    
# Register the model
ModelRegistry.register("my_model", MyCustomModel)
```

## 🎯 Adding Custom Tasks

```python
from vmevalkit.tasks import BaseTask

class MyReasoningTask(BaseTask):
    def __init__(self):
        super().__init__(name="my_task")
    
    def generate_problem(self, difficulty):
        # Create problem instance
        return problem_data
    
    def evaluate_solution(self, video, ground_truth):
        # Evaluate the generated video
        return score

# Register the task
TaskLoader.register_task("my_task", MyReasoningTask)
```

## 📚 Documentation

For detailed documentation, visit our [docs](https://vmevalkit.readthedocs.io) or check the `docs/` directory.

- [API Reference](docs/api_reference.md)
- [Model Integration Guide](docs/model_integration.md)
- [Task Creation Guide](docs/task_creation.md)
- [Evaluation Metrics Details](docs/metrics.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔗 Submodules

VMEvalKit includes two important research repositories as git submodules:

1. **KnowWhat** (`submodules/KnowWhat/`) - Research on knowing-how vs knowing-that capabilities in models
   - Repository: [https://github.com/hokindeng/KnowWhat](https://github.com/hokindeng/KnowWhat)

2. **maze-dataset** (`submodules/maze-dataset/`) - Maze datasets for investigating out-of-distribution behavior of ML systems
   - Repository: [https://github.com/understanding-search/maze-dataset](https://github.com/understanding-search/maze-dataset)


To initialize these submodules after cloning:
```bash
git submodule update --init --recursive
```

## 📝 Citation

If you use VMEvalKit in your research, please cite our paper:

```bibtex
@article{vmevalkit2024,
  title={VMEvalKit: A Comprehensive Framework for Evaluating Reasoning in Video Generation Models},
  author={Your Name and Collaborators},
  journal={arXiv preprint arXiv:2024.xxxxx},
  year={2024}
}
```

## 🙏 Acknowledgments

We thank the authors of the video generation models and reasoning benchmarks that make this evaluation framework possible.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📮 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/VMEvalKit/issues)

---

<p align="center">
  Made with ❤️ by the VMEvalKit Team
</p>

## ☁️ S3 Data Versioning Sync

You can version and back up the `data/` folder to S3. Each sync creates a date-based path:

`s3://vmevalkit/<YYYYMMDD>/data`

Additionally, a `latest_data_path.txt` object is written at the bucket root with the most recent URI.

Setup:

```bash
cp env.template .env
# Fill in AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY or rely on instance role
# Optionally set S3_BUCKET (defaults to vmevalkit) and AWS_REGION
```

Usage:

```bash
# Activate venv
source venv/bin/activate

# Sync local data/ to S3 with today's date folder
python data/s3_sync.py

# Or specify custom directory/bucket/date
python data/s3_sync.py --data-dir ./data --bucket vmevalkit --date 20251008
```

This mechanism supports the project's goal of reproducible, date-versioned datasets for evaluating video reasoning models.
