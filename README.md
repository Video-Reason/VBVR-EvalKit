# VMEvalKit 🎥🧠

A framework to evaluate reasoning capabilities in video generation models at scale, through cognitive tasks. We **make it very convenient** to [**add models**](docs/ADDING_MODELS.md), [**add tasks**](docs/ADDING_TASKS.md), [**run inferences**](docs/INFERENCE.md), [**run evaluations**](docs/EVALUATION.md), [**manage datasets**](docs/DATA_MANAGEMENT.md) and [**display results**](https://grow-ai-like-a-child.com/video-reason/). It's **permissively open-source**, and we welcome everyone to [**join**](https://join.slack.com/t/growingailikeachild/shared_invite/zt-309yqd0sl-W8xzOkdBPha1Jh5rnee78A) us and **build in public together**! 🚀 

👀 ✨ See preliminary [**results**](https://grow-ai-like-a-child.com/video-reason/) 🎬 🧠

![VMEvalKit Framework](paper/video-models-start-to-solve/assets/draft_1.jpg)

### Basic Idea

VMEvalKit aims to provide an infrastructure for reasoning research in video models at scale:

- 🎯  [**Task Creation at Scale**](docs/ADDING_TASKS.md): Create question dataset of many different cognitive tasks programmatically at scale and our framework makes sure the dataset to be well-organized.
- 🚀  [**Model Inference at Scale**](docs/INFERENCE.md): Easy one-click inference of the entire question dataset across many video models (commercial APIs + open-source) with automatic resume, error handling, and structured output management, and automatically sync the inference results into the dataset. 
- ⚖️  [**Evaluation Pipeline**](docs/EVALUATION.md): Human evaluation via web interface and AI evaluation via automated MLLM scoring, also automatically sync the eval results into the dataset. 
- ☁️  [**Dataset Management**](docs/DATA_MANAGEMENT.md): Manage question datasets from task creation, inference results from video models, and evaluation results from humans or MLLM pipelines. Provide both AWS S3 or HuggingFace use cases, with version tracking and built-in logging for reproducibility. 

We have completed running a question dataset of [**chess**](/vmevalkit/tasks/chess_task/CHESS.md), [**maze**](/vmevalkit/tasks/maze_task/MAZE.md), [**Sudoku**](/vmevalkit/tasks/sudoku_task/SUDOKU.md), [**mental rotation**](/vmevalkit/tasks/rotation_task/ROTATION.md), and [**Raven's Matrices**](/vmevalkit/tasks/raven_task/RAVEN.md) on [**latest video models**](https://grow-ai-like-a-child.com/video-reason/). Checkout our raw results videos on this [**website**](https://grow-ai-like-a-child.com/video-reason/). Here are a few examples.

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

## Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/hokindeng/VMEvalKit.git
cd VMEvalKit
```

2. **Initialize submodules** - good for optional open-source models and datasets
```bash
git submodule update --init --recursive
```

3. **Configure environment** - Copy the example environment file and add your API keys
```bash
cp env.template .env
```

4. **Set up Python environment** – Recommended: use a fresh virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

Alternatively, you can use other tools like [`uv`](https://github.com/astral-sh/uv) for faster install (`uv venv`), or [`conda`](https://docs.conda.io/) if your usecase has cross-language dependencies.

5. **Install dependencies:**

```bash
pip install -r requirements.txt
pip install -e .
```

## Tasks

Every VMEvalKit dataset consists of **Task Pairs** - the basic unit for video reasoning evaluation:

Each Task Pair consists of three core components:
- 📸 **Initial state image** (`first_frame.png`): shows the starting point or problem to be solved
- 🎯 **Final state image** (`final_frame.png`): illustrates the goal state or solution  
- 📝 **Text prompt** (`prompt.txt`): provides natural language instructions for the video model

There is also an accompanying `question_metadata.json` file with rich metadata. Each task pair is organized in its own folder (`data/questions/{domain}_task/{question_id}/`) containing all four files. 

![Task Pair Structure](paper/video-models-start-to-solve/assets/question_set.jpg)

## Documentation

📚 **Core Documentation:**
- **[Inference Guide](docs/INFERENCE.md)** - Complete guide to running inference, supported models, and architecture
- **[Evaluation Guide](docs/EVALUATION.md)** - Human and automated evaluation methods
- **[Data Management](docs/DATA_MANAGEMENT.md)** - Dataset organization, S3 sync, and version tracking
- **[Adding Models](docs/ADDING_MODELS.md)** - How to add new video generation models
- **[Adding Tasks](docs/ADDING_TASKS.md)** - How to create new reasoning tasks
- **[Web Dashboard](docs/WEB_DASHBOARD.md)** - Interactive results visualization

## Research

Here we keep track of papers spinned off from this code infrastructure and some works in progress.

- [**"Video Models Start to Solve Chess, Maze, Sudoku, Mental Rotation, and Raven's Matrices"**](paper/video-models-start-to-solve/Video_Model_Start_to_Solve.pdf)

This paper implements the experimental framework which demonstrates that leading video generation models (Sora-2 etc) can perform visual reasoning tasks with >60% success rates. See [**results**](https://grow-ai-like-a-child.com/video-reason/).

## License

Apache 2.0