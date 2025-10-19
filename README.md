# VMEvalKit 🎥🧠

Evaluate reasoning capabilities in video generation models through cognitive tasks.

## Overview

VMEvalKit tests whether video models can solve visual problems (mazes, chess, puzzles) by generating solution videos. 

**Key requirement**: Models must accept BOTH:
- 📸 An input image (the problem)
- 📝 A text prompt (instructions)

## Installation

```bash
git clone https://github.com/yourusername/VMEvalKit.git
cd VMEvalKit
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from vmevalkit.runner.inference import InferenceRunner

# Initialize runner with structured output
runner = InferenceRunner(output_dir="output")

# Generate video solution
result = runner.run(
    model_name="luma-ray-2",
    image_path="data/questions/maze_task/maze_0000/first_frame.png",
    text_prompt="Navigate the green dot through the maze corridors to reach the red flag"
)

print(f"Video saved to: {result['inference_dir']}")
# Each inference creates a self-contained folder with:
# - video/: Generated video file
# - question/: Input images and prompt  
# - metadata.json: Complete inference metadata
```

## Supported Models

VMEvalKit supports **40 models** across **11 families** using a clean modular architecture:

**Commercial APIs (29 models):**
- **Luma Dream Machine**: 2 models (`luma-ray-2`, `luma-ray-flash-2`)
- **Google Veo**: 3 models (`veo-2.0-generate`, `veo-3.0-generate`, etc.)
- **Google Veo 3.1**: 4 models (via WaveSpeed, with 720p/1080p variants)
- **WaveSpeed WAN**: 18 models (2.1 & 2.2 variants with LoRA/ultra-fast options)
- **Runway ML**: 3 models (Gen-3A Turbo, Gen-4 Turbo/Aleph)
- **OpenAI Sora**: 2 models (Sora-2, Sora-2-Pro)

**Open-Source Models (11 models):**
- **LTX-Video**: 3 models (13B distilled, 13B dev, 2B distilled)
- **HunyuanVideo**: 1 model (high-quality 720p)
- **VideoCrafter**: 1 model (text-guided generation)
- **DynamiCrafter**: 3 models (256p, 512p, 1024p)

All models support **image + text → video** for reasoning evaluation.

## Core Concepts

### Task Pair: The Fundamental Unit
Every VMEvalKit dataset consists of **Task Pairs** - the basic unit for video reasoning evaluation:

- 📸 **Initial state image** (`first_frame.png` - the reasoning problem)
- 🎯 **Final state image** (`final_frame.png` - the solution/goal state)  
- 📝 **Text prompt** (`prompt.txt` - instructions for video model)
- 📊 **Rich metadata** (`question_metadata.json` - difficulty, task-specific parameters, etc.)

Each task pair is organized in its own folder (`data/questions/{domain}_task/{question_id}/`) containing all four files. Models must generate videos showing the reasoning process from initial → final state.

## Tasks

- **Maze Solving**: Navigate from start to finish
- **Mental Rotation**: Rotate 3D objects to match targets
- **Chess Puzzles**: Demonstrate puzzle solutions
- **Raven's Matrices**: Complete visual patterns

## Configuration

Create `.env`:
```bash
LUMA_API_KEY=your_key_here
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=vmevalkit
AWS_DEFAULT_REGION=us-east-2
```

## Architecture

VMEvalKit uses a **clean modular architecture** with dynamic loading:

```
vmevalkit/
├── runner/
│   ├── MODEL_CATALOG.py    # 📋 Pure model registry (40 models, 11 families)
│   └── inference.py        # 🎭 Orchestration with dynamic loading
├── models/
│   ├── base.py            # 🔧 Abstract ModelWrapper interface
│   ├── luma_inference.py  # LumaInference + LumaWrapper
│   ├── veo_inference.py   # VeoService + VeoWrapper 
│   └── ...                # Each provider: Service + Wrapper
```

**Key Benefits:**
- **Dynamic Loading**: Models loaded on-demand from catalog
- **Family Organization**: Models grouped by provider families
- **Consistent Interface**: All wrappers inherit from `ModelWrapper`
- **Easy Extension**: Add models without touching core files

## Project Structure

```
VMEvalKit/
├── vmevalkit/
│   ├── runner/         # Inference runners + model catalog
│   ├── models/         # Model implementations (service + wrapper)
│   ├── core/           # Evaluation framework
│   ├── tasks/          # Task definitions
│   └── utils/          # Utilities
├── data/
│   └── questions/      # Dataset with per-question folders
│       ├── vmeval_dataset.json  # Master dataset manifest
│       ├── chess_task/          # Chess reasoning questions
│       │   └── chess_0000/      # Individual question folder
│       │       ├── first_frame.png
│       │       ├── final_frame.png
│       │       ├── prompt.txt
│       │       └── question_metadata.json
│       ├── maze_task/           # Maze navigation questions
│       ├── raven_task/          # Pattern completion questions
│       └── rotation_task/       # 3D rotation questions
├── output/             # Structured inference outputs
│   └── <inference_id>/ # Self-contained folders per inference
│       ├── video/      # Generated video file
│       ├── question/   # Input images and prompt
│       └── metadata.json # Complete inference metadata
├── examples/           # Example scripts
└── tests/              # Unit tests
```

## Structured Output System

Each inference creates a **self-contained folder** with all relevant data:

```
output/<model>_<question_id>_<timestamp>/
├── video/
│   └── generated_video.mp4    # Output video
├── question/
│   ├── first_frame.png        # Input image (sent to model)
│   ├── final_frame.png        # Reference image (not sent)
│   ├── prompt.txt             # Text prompt used
│   └── question_metadata.json # Full question data from dataset
└── metadata.json              # Complete inference metadata
```

This structure ensures reproducibility and makes batch analysis easy.

## Web Dashboard 🎨  

Visualize your results with the built-in web dashboard:

```bash
cd web
./start.sh
# Open http://localhost:5000
```

Features:
- 📊 Overview statistics and model performance
- 🎬 Video playback and comparison
- 🧠 Domain and task analysis
- ⚖️ Side-by-side model comparison

See [docs/WEB_DASHBOARD.md](docs/WEB_DASHBOARD.md) for details.

## Examples

See `examples/experiment_2025-10-14.py` for sequential inference across multiple models.

## Submodules

Initialize after cloning:
```bash
git submodule update --init --recursive
```

- **maze-dataset**: Maze datasets for ML evaluation
- **HunyuanVideo-I2V**: High-quality image-to-video generation (720p)
- **LTX-Video**: Real-time video generation models
- **VideoCrafter**: Text-guided video generation
- **DynamiCrafter**: Image animation with video diffusion

## Contributing

### Adding New Models

VMEvalKit supports 40 models across 11 families with a **modular architecture** designed for easy extension.

**Requirements:**
- Model must support **both image + text input** for reasoning evaluation
- Inherit from `ModelWrapper` base class for consistent interface

**Quick Steps:**
1. Create service + wrapper in `vmevalkit/models/{provider}_inference.py`
2. Register in `vmevalkit/runner/MODEL_CATALOG.py` (pure data)
3. Update imports in `vmevalkit/models/__init__.py`

**Key Features:**
- **Dynamic Loading**: No need to modify `inference.py`
- **Base Class**: Inherit from `ModelWrapper` for consistency
- **Family Organization**: Models grouped by provider families
- **String Module Paths**: Flexible loading without circular imports

**Documentation:**
- 📚 **Adding Models Guide**: [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) (includes architecture details)

Both API-based and open-source (submodule) integration patterns are supported.

## Running Experiments

### Quick Start

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

## License

Apache 2.0