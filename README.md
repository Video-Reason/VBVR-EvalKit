# VMEvalKit 🎥🧠


<div align="center">


[![results](https://img.shields.io/badge/Result-A42C2?style=for-the-badge&logo=googledisplayandvideo360&logoColor=white)](https://grow-ai-like-a-child.com/video-reason/)
[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](paper/video-models-start-to-solve/Video_Model_Start_to_Solve.pdf) 
[![Hugging Face](https://img.shields.io/badge/hf-fcd022?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/VideoReason)
[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white)](https://github.com/hokindeng/VMEvalKit/issues/132)


</div>


A framework to score reasoning capabilities in video generation models at scale, through cognitive tasks. We **make it very convenient** to [**add models**](docs/ADDING_MODELS.md), [**add tasks**](docs/ADDING_TASKS.md), [**run inferences**](docs/INFERENCE.md), [**run scoring**](docs/SCORING.md), [**manage datasets**](docs/DATA_MANAGEMENT.md) and [**display results**](https://grow-ai-like-a-child.com/video-reason/). It's **permissively open-source**, and we welcome everyone to [**join**](https://join.slack.com/t/growingailikeachild/shared_invite/zt-309yqd0sl-W8xzOkdBPha1Jh5rnee78A) us and **build in public together**! 🚀 


<p align="center">
    
</p>

![VMEvalKit Framework](paper/video-models-start-to-solve/assets/draft_1.jpg)


## 🎬 Supported Models

VMEvalKit provides unified access to **40 video generation models** across **11 provider families**:

For commercial APIs, we support Luma Dream Machine, Google Veo, Google Veo 3.1, WaveSpeed WAN 2.1, WaveSpeed WAN 2.2, Runway ML, OpenAI Sora. For open-source models, we support HunyuanVideo, VideoCrafter, DynamiCrafter, Stable Video Diffusion, Morphic, LTX-Video, and so on. See [here](docs/models/README.md) for details.


## 📊 Supported Datasets

VMEvalKit provides access to **9 local task generation engines(quickly increasing)** and other external benchmark datasets (HuggingFace) [here](docs/tasks/README.md).

### Local Task Generation Engines

Tasks supported by VMEvalKit:

Chess, Maze, Raven, Rotation, Sudoku, Object Subtraction, Clock, mirror clock. For more details, see [**Task Docs**](docs/tasks/README.md).

### Basic Idea

VMEvalKit aims to provide an infrastructure for reasoning research in video models at scale:

- 🎯  [**Task Creation at Scale**](docs/ADDING_TASKS.md): Create question dataset of many different cognitive tasks programmatically at scale and our framework makes sure the dataset to be well-organized.
- 🚀  [**Model Inference at Scale**](docs/INFERENCE.md): Easy one-click inference of the entire question dataset across many video models (commercial APIs + open-source) with automatic resume, error handling, and structured output management, and automatically sync the inference results into the dataset. 
- ⚖️  [**Scoring Pipeline**](docs/SCORING.md): Human scoring via web interface and AI scoring via automated MLLM scoring, also automatically sync the scoring results into the dataset. 
- ☁️  [**Dataset Management**](docs/DATA_MANAGEMENT.md): Manage question datasets from task creation, inference results from video models, and scoring results from humans or MLLM pipelines. Provides AWS S3 integration with version tracking and built-in logging for reproducibility. 

We have completed running a question dataset of [**chess**](/vmevalkit/tasks/chess_task/CHESS.md), [**maze**](/vmevalkit/tasks/maze_task/MAZE.md), [**Sudoku**](/vmevalkit/tasks/sudoku_task/SUDOKU.md), [**mental rotation**](/vmevalkit/tasks/rotation_task/ROTATION.md), and [**Raven's Matrices**](/vmevalkit/tasks/raven_task/RAVEN.md) on [**latest video models**](https://grow-ai-like-a-child.com/video-reason/). Checkout our raw results videos on this [**website**](https://grow-ai-like-a-child.com/video-reason/). Here are a few examples.

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

For open-source video generation and evaluator models, please refer to [**Open Source Models**](./examples/opensource/open_source.md) for detailed installation instructions.

## 🚀 Quick Start - End-to-End Example

Here's a complete workflow from creating questions to scoring results:

### 1️⃣ Create Questions
```bash
# Generate 5 chess and maze questions each
python examples/create_questions.py --task chess maze --pairs-per-domain 5

# Output: Creates data/questions/ with chess_task/ and maze_task/ folders
```

### 2️⃣ Generate Videos
```bash
# Run on specific model (e.g., stable video diffusion)
python examples/generate_videos.py --model svd --task chess maze

# Output: Creates data/outputs/pilot_experiment/ with generated videos
# for close source model, need to set key in .env file
```

### 3️⃣ Score Results
```bash
# open source VLM Automated scoring
bash script/lmdeploy_server.sh

# Human scoring via web interface
python examples/score_videos.py human

# Automated GPT-4O scoring
python examples/score_videos.py gpt4o
```

### 4️⃣ View Results
```bash
# Launch web dashboard to explore results
cd web && ./start.sh
# Open http://localhost:5000 in your browser
```

That's it! You now have:
- ✅ Custom reasoning questions in `data/questions/`  
- ✅ Generated videos in `data/outputs/`
- ✅ Scoring results in `data/scorings/`
- ✅ Interactive dashboard


## Tasks

Every VMEvalKit dataset consists of **Task Pairs** - the basic unit for video reasoning scoring:

We have two types of tasks:

### Final image

Each Task Pair consists of three core components:
- 📸 **Initial state image** (`first_frame.png`): shows the starting point or problem to be solved
- 🎯 **Final state image** (`final_frame.png`): illustrates the goal state or solution  
- 📝 **Text prompt** (`prompt.txt`): provides natural language instructions for the video model

There is also an accompanying `question_metadata.json` file with rich metadata. Each task pair is organized in its own folder (`data/questions/{domain}_task/{question_id}/`) containing all four files. 

![Task Pair Structure](paper/video-models-start-to-solve/assets/question_set.jpg)

### Final text answer

Each Task Pair consists of three core components:
- 📸 **Initial state image** (`first_frame.png`): shows the starting point or problem to be solved
- 📝 **Text answer** (`goal.txt`): provides the text answer to the question
- 📝 **Text prompt** (`prompt.txt`): provides natural language instructions for the video model

With our VMEvalKit, you can easily create tasks with final text answer by simply adding a `goal.txt` file to the task folder, so you could adapt your VQA datasets to video reasoning tasks.

For more details, see [**Task Docs**](docs/tasks/README.md).

## Inference Architecture

See **[Inference Guide](docs/INFERENCE.md)** for details. 

## Scoring Pipeline

See **[Scoring Guide](docs/SCORING.md)** for details.

## Dataset Management

See **[Data Management](docs/DATA_MANAGEMENT.md)** for details. 

## Display Results

See **[Web Dashboard](docs/WEB_DASHBOARD.md)** for details.

## Add Models or Tasks

You can add new video generation models and reasoning tasks with minimal effort:

**Adding New Models**

Add any video generation model (API-based or open-source) with just a few steps:

```python
# Example: Adding a new model wrapper
from vmevalkit.models.base import BaseVideoModel

class MyModelWrapper(BaseVideoModel):
    def generate_video(self, image_path, text_prompt, **kwargs):
        # Your model's video generation logic
        return video_path
```

Then register it in `MODEL_CATALOG.py`:
```python
"my-model": {
    "provider": "mycompany",
    "wrapper_path": "vmevalkit.models.my_model.MyModelWrapper",
    ...
}
```

See **[Adding Models Guide](docs/ADDING_MODELS.md)** for details.

**Adding New Tasks**

Create new reasoning tasks programmatically at scale:

```python
from vmevalkit.tasks.base_task import BaseTask

class MyTask(BaseTask):
    def generate_task_pair(self, ...):
        # Generate initial and final states
        initial_state = self.create_initial_state()
        final_state = self.create_final_state()
        prompt = self.create_prompt()
        
        return {
            "first_frame": initial_state,
            "final_frame": final_state, 
            "prompt": prompt,
            "metadata": {...}
        }
```

See **[Adding Tasks Guide](docs/ADDING_TASKS.md)** for details.

## Invitation to Collaborate 🤝

VMEvalKit is meant to be a permissively open-source **shared playground** for everyone. If you’re interested in machine cognition, video models, evaluation, or anything anything 🦄✨, we’d love to build with you:

* 🧪 Add new reasoning tasks (planning, causality, social, physical, etc.)
* 🎥 Plug in new video models (APIs or open-source)
* 📊 Experiment with better evaluation metrics and protocols
* 🧱 Improve infrastructure, logging, and the web dashboard
* 📚 Use VMEvalKit in your own research and share back configs/scripts
* 🌟🎉 Or Anything anything 🦄✨

💬 **Join us on Slack** to ask questions, propose ideas, or start a collab:
[Slack Invite](https://join.slack.com/t/growingailikeachild/shared_invite/zt-309yqd0sl-W8xzOkdBPha1Jh5rnee78A) 🚀

## Documentation

📚 **Core Documentation:**
- **[Inference Guide](docs/INFERENCE.md)** - Complete guide to running inference, supported models, and architecture
- **[Scoring Guide](docs/SCORING.md)** - Human and automated scoring methods
- **[Data Management](docs/DATA_MANAGEMENT.md)** - Dataset organization, S3 sync, and version tracking
- **[Adding Models](docs/ADDING_MODELS.md)** - How to add new video generation models
- **[Adding Tasks](docs/ADDING_TASKS.md)** - How to create new reasoning tasks
- **[Web Dashboard](docs/WEB_DASHBOARD.md)** - Interactive results visualization

## Research

Here we keep track of papers spinned off from this code infrastructure and some works in progress.

- [**"Video Models Start to Solve Chess, Maze, Sudoku, Mental Rotation, and Raven's Matrices"**](paper/video-models-start-to-solve/Video_Model_Start_to_Solve.pdf)

This paper implements our experimental framework and demonstrates that leading video generation models (Sora-2 etc) can perform visual reasoning tasks with >60% success rates. See [**results**](https://grow-ai-like-a-child.com/video-reason/).

## License

Apache 2.0


## Citation

If you find VMEvalKit useful in your research, please cite:

```bibtex
@misc{VMEvalKit,
  author       = {VMEvalKit Team},
  title        = {VMEvalKit: A framework for evaluating reasoning abilities in foundational video models},
  year         = {2025},
  howpublished = {\url{https://github.com/Video-Reason/VMEvalKit}}
}
```