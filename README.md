# VMEvalKit 🎥🧠

A framework to evaluate reasoning capabilities in video generation models at scale, through cognitive tasks. We **make it very convenient** to [**add models**](docs/ADDING_MODELS.md), [**add tasks**](docs/ADDING_TASKS.md), [**run inferences**](docs/INFERENCE.md), [**run evaluations**](docs/EVALUATION.md), and [**display results**](https://grow-ai-like-a-child.com/video-reason/). It's **permissively open-source**, and we welcome everyone to [**join**](https://join.slack.com/t/growingailikeachild/shared_invite/zt-309yqd0sl-W8xzOkdBPha1Jh5rnee78A) us and **build in public together**! 🚀

![VMEvalKit Framework](paper/video-models-start-to-solve/assets/draft_1.jpg)

### Basic Idea

VMEvalKit aims to provide an infrastructure for reasoning research in video models at scale:

- **🎯 Task Creation at Scale**: Create question dataset of many different cognitive tasks programmatically at scale and our framework makes sure the dataset to be well-organized.
- **🚀 Model Inference at Scale**: Easy one-click inference of the entire question dataset across many video models (commercial APIs + open-source) with automatic resume, error handling, and structured output management, and automatically sync the inference results into the dataset. 
- **⚖️ Evaluation Pipeline**: Human evaluation via web interface and AI evaluation via automated MLLM scoring, also automatically sync the eval results into the dataset. 
- **☁️ Dataset Management**: Manage question datasets from task creation, inference results from video models, and evaluation results from humans or MLLM pipelines. Provide both AWS S3 or HuggingFace use case, with version tracking and built-in logging for reproducibility. 

We have completed running a question dataset of chess, maze, Sudoku, mental rotation, and Raven's Matrices on latest video models. Checkout our raw results ([**videos**](https://grow-ai-like-a-child.com/video-reason/)) on this [**website**](https://grow-ai-like-a-child.com/video-reason/). Here are a few examples.

Solving Chess

![Chess Example](paper/video-models-start-to-solve/assets/chess_example.jpg)

Solving Maze

![Maze Example](paper/video-models-start-to-solve/assets/maze_example.jpg)

Mental Rotation

![Rotation Example](paper/video-models-start-to-solve/assets/rotation_example.jpg)

Raven's Matrices

![Raven Example](paper/video-models-start-to-solve/assets/raven_example.jpg)

Sudoku Solving

![Sudoku Example](paper/video-models-start-to-solve/assets/sudoku_example.jpg)

## Installation

```bash
git clone https://github.com/hokindeng/VMEvalKit.git
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
```

**📊 See all experimental results and videos:** [**Interactive Results Page**](https://grow-ai-like-a-child.com/video-reason/)

## Tasks

VMEvalKit evaluates models across 5 cognitive reasoning domains:

![Task Pair Structure](paper/video-models-start-to-solve/assets/question_set.jpg)

### 🧩 Maze Solving
Navigate from start to finish through complex pathways.

![Maze Example](paper/video-models-start-to-solve/assets/maze_example.jpg)

### 🔄 Mental Rotation
Rotate 3D objects to match target orientations.

![Rotation Example](paper/video-models-start-to-solve/assets/rotation_example.jpg)

### ♟️ Chess Puzzles
Demonstrate chess puzzle solutions with strategic moves.

![Chess Example](paper/video-models-start-to-solve/assets/chess_example.jpg)

### 🎨 Raven's Matrices
Complete visual pattern reasoning tasks.

![Raven Example](paper/video-models-start-to-solve/assets/raven_example.jpg)

### 🔢 Sudoku Solving
Complete 3×3 grids using logical deduction.

![Sudoku Example](paper/video-models-start-to-solve/assets/sudoku_example.jpg)

**🔬 Research Findings**: Leading models achieve >60% success rates on reasoning tasks. See detailed performance analysis and example videos at the [**Results Page**](https://grow-ai-like-a-child.com/video-reason/).

## Configuration

Create `.env`:
```bash
LUMA_API_KEY=your_key_here
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=vmevalkit
AWS_DEFAULT_REGION=us-east-2
WAVESPEED_API_KEY=your_wavespeed_api_key
```

## Documentation

📚 **Core Documentation:**
- **[Inference Guide](docs/INFERENCE.md)** - Complete guide to running inference, supported models, and architecture
- **[Evaluation Guide](docs/EVALUATION.md)** - Human and automated evaluation methods
- **[Adding Models](docs/ADDING_MODELS.md)** - How to add new video generation models
- **[Adding Tasks](docs/ADDING_TASKS.md)** - How to create new reasoning tasks
- **[Web Dashboard](docs/WEB_DASHBOARD.md)** - Interactive results visualization

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

## Supported Models

VMEvalKit supports **40 models** across **11 families**:

**Commercial APIs (29 models):**
- Luma Dream Machine, Google Veo (2.0, 3.0, 3.1), WaveSpeed WAN, Runway ML, OpenAI Sora

**Open-Source Models (11 models):**
- LTX-Video, HunyuanVideo, VideoCrafter, DynamiCrafter

📚 **See [Inference Guide](docs/INFERENCE.md) for complete model list and usage.**

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

## Running Experiments

```bash
# Quick start - run 1 task per domain
python examples/experiment_2025-10-14.py

# Run all tasks with automatic resume
python examples/experiment_2025-10-14.py --all-tasks

# Run specific models
python examples/experiment_2025-10-14.py --all-tasks --only-model luma-ray-2 veo-3.0-generate
```

📚 **See [Inference Guide](docs/INFERENCE.md) for detailed documentation on running experiments.**

## Evaluation

```bash
# Human evaluation with web interface
python examples/run_evaluation.py human

# Automatic GPT-4O evaluation
export OPENAI_API_KEY=your_api_key
python examples/run_evaluation.py gpt4o
```

📚 **See [Evaluation Guide](docs/EVALUATION.md) for detailed documentation.**

## Contributing

We welcome contributions! Check out:
- 📚 **[Adding Models](docs/ADDING_MODELS.md)** - Add new video generation models
- 📚 **[Adding Tasks](docs/ADDING_TASKS.md)** - Create new reasoning tasks

## Paper & Research

**"Video Models Start to Solve Chess, Maze, Sudoku, Mental Rotation, and Raven's Matrices"**

This codebase implements the experimental framework from our research paper, which demonstrates that leading video generation models (Sora-2, Veo-3, etc.) can perform visual reasoning tasks with >60% success rates.

**Key Findings:**
- Sora-2 achieves 68% overall success rate (87% on maze navigation, 73% on chess)
- Strong correlation (r=0.949) between human and GPT-4o automated evaluation
- Sudoku is most tractable (57% average), mental rotation most challenging (11%)
- Clear performance hierarchy across 6 tested models

**Resources:**
- 📄 **Paper**: [paper/video-models-start-to-solve/Video_Model_Start_to_Solve.pdf](paper/video-models-start-to-solve/Video_Model_Start_to_Solve.pdf)
- 🌐 **Results Page**: [https://grow-ai-like-a-child.com/video-reason/](https://grow-ai-like-a-child.com/video-reason/) - Interactive visualization of all experimental results
- 📊 **Task Structure**: [paper/video-models-start-to-solve/assets/question_set.jpg](paper/video-models-start-to-solve/assets/question_set.jpg)

**Citation:**
```bibtex
@article{deng2025videoreasoning,
  title={Video Models Start to Solve Chess, Maze, Sudoku, Mental Rotation, and Raven's Matrices},
  author={Deng, Hokin},
  journal={arXiv preprint},
  year={2025},
  url={https://github.com/hokindeng/VMEvalKit}
}
```

## License

Apache 2.0