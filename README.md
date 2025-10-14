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
    image_path="data/questions/maze_task/knowwhat_0000/first_frame.png",
    text_prompt="Move the blue star through white corridors to the red circle"
)

print(f"Video saved to: {result['inference_dir']}")
# Each inference creates a self-contained folder with:
# - video/: Generated video file
# - question/: Input images and prompt  
# - metadata.json: Complete inference metadata
```

## Supported Models

VMEvalKit supports **36+ models** across **9 providers**:

**Commercial APIs (28 models):**
- **Luma Dream Machine**: 2 models (`luma-ray-2`, `luma-ray-flash-2`)
- **Google Veo**: 3 models (`veo-2.0-generate`, `veo-3.0-generate`, etc.)
- **WaveSpeed WAN**: 18 models (2.1 & 2.2 variants)
- **Runway ML**: 3 models
- **OpenAI Sora**: 2 models

**Open-Source Models (8 models):**
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

## Project Structure

```
VMEvalKit/
├── vmevalkit/
│   ├── runner/         # Inference runners
│   ├── models/         # Model implementations
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

## Examples

See `examples/pilot_experiment.py` for parallel inference across multiple models.

## Submodules

Initialize after cloning:
```bash
git submodule update --init --recursive
```

- **KnowWhat**: Research on knowing-how vs knowing-that
- **maze-dataset**: Maze datasets for ML evaluation
- **HunyuanVideo-I2V**: High-quality image-to-video generation (720p)
- **LTX-Video**: Real-time video generation models
- **VideoCrafter**: Text-guided video generation
- **DynamiCrafter**: Image animation with video diffusion

## Contributing

### Adding New Models

VMEvalKit supports 36+ models across 9 providers and is designed to easily accommodate new models.

**Requirements:**
- Model must support **both image + text input** for reasoning evaluation
- Follow the unified inference interface

**Quick Steps:**
1. Create wrapper class in `vmevalkit/models/{provider}_inference.py`
2. Register in `vmevalkit/runner/inference.py` 
3. Update imports in `vmevalkit/models/__init__.py`

**Documentation:**
- 📚 **Complete Guide**: [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md)

Both API-based and open-source (submodule) integration patterns are supported.

## License

MIT