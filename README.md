# VMEvalKit 🎥🧠


<div align="center">


[![results](https://img.shields.io/badge/Result-A42C2?style=for-the-badge&logo=googledisplayandvideo360&logoColor=white)](https://grow-ai-like-a-child.com/video-reason/)
[![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](paper/video-models-start-to-solve/Video_Model_Start_to_Solve.pdf) 
[![Hugging Face](https://img.shields.io/badge/hf-fcd022?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/VideoReason)
[![WeChat](https://img.shields.io/badge/WeChat-07C160?style=for-the-badge&logo=wechat&logoColor=white)](https://github.com/hokindeng/VMEvalKit/issues/132)
[![Homepage](https://img.shields.io/badge/Homepage-07D100?style=for-the-badge&logo=googlehome&logoColor=white)](https://video-reason.github.io/)


</div>


A framework for **inference and evaluation** of video generation models' reasoning abilities at scale. VMEvalKit provides a unified interface for **40+ video models** across **11 provider families** and comprehensive evaluation pipelines. We **make it very convenient** to [**add models**](docs/ADDING_MODELS.md), [**run inferences**](docs/INFERENCE.md), [**run scoring**](docs/SCORING.md), and [**display results**](https://grow-ai-like-a-child.com/video-reason/). It's **permissively open-source**, and we welcome everyone to [**join**](https://join.slack.com/t/growingailikeachild/shared_invite/zt-309yqd0sl-W8xzOkdBPha1Jh5rnee78A) us and **build in public together**! 🚀

**Note**: VMEvalKit now focuses specifically on **inference and evaluation**. Data generation is handled externally - simply prepare your reasoning task datasets in the [expected format](docs/DATA_MANAGEMENT.md) and VMEvalKit will handle the rest. 


<p align="center">
    
</p>

![VMEvalKit Framework](paper/video-models-start-to-solve/assets/draft_1.jpg)


## 🎬 Supported Models

VMEvalKit provides **unified inference interface** for **40+ video generation models** across **11 provider families**:

**Commercial APIs**: Luma Dream Machine, Google Veo (2.0, 3.0, 3.1), WaveSpeed WAN (2.1, 2.2), Runway ML (Gen-4), OpenAI Sora (2, 2-Pro)

**Open-Source Models**: HunyuanVideo, VideoCrafter, DynamiCrafter, Stable Video Diffusion, Morphic, LTX-Video, CogVideoX, SANA-Video, WAN (local)

See `vmevalkit/runner/MODEL_CATALOG.py` for complete list.

## 📊 Data Format  

VMEvalKit works with **simple reasoning task datasets** in this folder structure:

```
data/questions/{domain}_task/{task_id}/
├── first_frame.png          # Initial state image
├── final_frame.png          # Target state image  
├── prompt.txt              # Text instructions
└── ground_truth.mp4        # Optional ground truth video
```

**Supported Domains**: Any domain name works - just use `{domain}_task` folders. Examples: `chess_task`, `maze_task`, `sudoku_task`, etc.

### Basic Idea

VMEvalKit provides a focused infrastructure for **inference and evaluation** of video models' reasoning capabilities:

- 🚀  **Model Inference at Scale**: Unified interface for 40+ video models (commercial APIs + open-source) with automatic resume, error handling, and structured output management.
- ⚖️  **Evaluation Pipeline**: Multiple scoring methods including human evaluation (Gradio interface), GPT-4O automated scoring, and open-source vision model scoring.
- ☁️  **Cloud Integration**: AWS S3 and HuggingFace Hub integration for dataset and result management. 

We have completed running a question dataset of [**chess**](/vmevalkit/tasks/chess_task/CHESS.md), [**maze**](/vmevalkit/tasks/maze_task/MAZE.md), [**Sudoku**](/vmevalkit/tasks/sudoku_task/SUDOKU.md), [**mental rotation**](/vmevalkit/tasks/rotation_task/ROTATION.md), and [**Raven's Matrices**](/vmevalkit/tasks/raven_task/RAVEN.md) on [**latest video models**](https://grow-ai-like-a-child.com/video-reason/). Checkout our raw results videos on this [**website**](https://grow-ai-like-a-child.com/video-reason/). Here are a few examples.

## Installation & Setup

1. **Clone the repository**
```bash
git clone https://github.com/hokindeng/VMEvalKit.git
cd VMEvalKit
```

2. **Initialize submodules** - required for some open-source models
```bash
git submodule update --init --recursive
```

3. **Configure environment** - Copy the example environment file and add your API keys
```bash
cp env.template .env
# Edit .env with your API keys for commercial models
```

4. **Set up Python environment**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

5. **Install models** - Use the automated setup system
```bash
# Install specific models
bash setup/install_model.sh --model ltx-video --validate

# Install all open-source models
bash setup/install_model.sh --opensource --validate

# List available models
bash setup/install_model.sh --list
```

**Model Weights**: All model weights are stored in a centralized `weights/` directory managed by the setup scripts.

## 🚀 Quick Start

Ultra-simple workflow:

### 1️⃣ Setup
```bash
# Install VMEvalKit
pip install -r requirements.txt
pip install -e .

# Install a model (e.g., SVD)
bash setup/install_model.sh --model svd --validate

# Set API keys for commercial models (edit .env)
cp env.template .env
```

### 2️⃣ Prepare Data
Put your reasoning task data in this format:
```
data/questions/{domain}_task/{task_id}/
├── first_frame.png
├── final_frame.png  
├── prompt.txt
└── ground_truth.mp4 (optional)
```

### 3️⃣ Run Inference
```bash
python examples/generate_videos.py --model svd --task chess maze
```

### 4️⃣ Run Evaluation
```bash
# Human evaluation
python examples/score_videos.py human

# Automated evaluation  
python examples/score_videos.py gpt4o
```

**Done!** Results saved as JSON files in `data/outputs/` and `data/scorings/`.


## Data Format

VMEvalKit works with **Task Pairs** - the basic unit for video reasoning evaluation:

### Standard Task Format

Each Task Pair consists of four core components:
- 📸 **Initial state image** (`first_frame.png`): shows the starting point or problem to be solved
- 🎯 **Final state image** (`final_frame.png`): illustrates the goal state or solution  
- 📝 **Text prompt** (`prompt.txt`): provides natural language instructions for the video model
- 📊 **Metadata** (`question_metadata.json`): contains task parameters and ground truth

Each task pair is organized in its own folder (`data/questions/{domain}_task/{question_id}/`) containing all four files.

![Task Pair Structure](paper/video-models-start-to-solve/assets/question_set.jpg)

### Alternative: Text Answer Format

For text-based reasoning tasks:
- 📸 **Initial state image** (`first_frame.png`): shows the starting point or problem to be solved
- 📝 **Text answer** (`goal.txt`): provides the text answer to the question
- 📝 **Text prompt** (`prompt.txt`): provides natural language instructions for the video model
- 📊 **Metadata** (`question_metadata.json`): contains task parameters

You can easily adapt VQA datasets to video reasoning tasks by following this format structure.

**Note**: VMEvalKit focuses on inference and evaluation. Data generation should be done externally using the simple format shown above.

## Inference Architecture

See **[Inference Guide](docs/INFERENCE.md)** for details. 

## Scoring Pipeline

See **[Scoring Guide](docs/SCORING.md)** for details.

## Dataset Management

See **[Data Management](docs/DATA_MANAGEMENT.md)** for details. 

## Display Results

See **[Web Dashboard](docs/WEB_DASHBOARD.md)** for details.

## Add Models or Use External Data

**Adding New Models**

VMEvalKit supports easy addition of new video generation models (API-based or open-source):

```python
# Example: Adding a new model wrapper
from vmevalkit.models.base import ModelWrapper

class MyModelWrapper(ModelWrapper):
    def generate(self, image_path, text_prompt, **kwargs):
        # Your model's video generation logic
        return standardized_result_dict
```

Then register it in `MODEL_CATALOG.py`:
```python
"my-model": {
    "wrapper_module": "vmevalkit.models.my_model_inference",
    "wrapper_class": "MyModelWrapper",
    "family": "MyCompany",
    ...
}
```

See `vmevalkit/models/base.py` and existing model implementations for examples.

**Working with External Data**

VMEvalKit expects reasoning datasets in this simple format:

```
data/questions/{domain}_task/{task_id}/
├── first_frame.png          # Initial state image
├── final_frame.png          # Target state image  
├── prompt.txt              # Text instructions
└── ground_truth.mp4        # Optional ground truth video
```

That's it! If files are missing or malformed, VMEvalKit will report errors during execution.

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

## Usage

📚 **Core Components:**
- **Model Inference**: `examples/generate_videos.py` - Run 40+ models on your data
- **Evaluation Pipeline**: `examples/score_videos.py` - Human + automated scoring
- **Model Setup**: `setup/install_model.sh` - Install and test models

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
