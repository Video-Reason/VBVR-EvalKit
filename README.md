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
from vmevalkit import run_inference

# Generate video solution
result = run_inference(
    model_name="luma-ray-2",
    image_path="data/questions/maze.png",
    text_prompt="Solve this maze from start to finish"
)

print(f"Video: {result['video_path']}")
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

- 📸 **Initial state image** (the reasoning problem)
- 🎯 **Final state image** (the solution/goal state)  
- 📝 **Text prompt** (instructions for video model)
- 📊 **Rich metadata** (difficulty, task-specific parameters, etc.)

Models must generate videos showing the reasoning process from initial → final state.

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
│   ├── questions/      # Dataset questions & images
│   └── outputs/        # Model-generated videos
├── examples/           # Example scripts
└── tests/              # Unit tests
```

## Examples

See `examples/simple_inference.py` for more usage patterns.

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