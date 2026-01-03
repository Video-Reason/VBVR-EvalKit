# VMEvalKit 🎥🧠

**Unified inference and evaluation framework for 40+ video generation models.**

## Features

- **🚀 40+ Models**: Unified interface for commercial APIs (Luma, Veo, Sora, Runway) + open-source (LTX-Video, HunyuanVideo, DynamiCrafter, SVD, etc.)
- **⚖️ Evaluation Pipeline**: Human scoring (Gradio) + automated scoring (GPT-4O, InternVL)  
- **☁️ Cloud Integration**: S3 + HuggingFace Hub support

## Data Format

Questions should be organized in directories outside VMEvalKit:

```
questions/{domain}_task/{task_id}/
├── first_frame.png          # Initial state image (required)
├── prompt.txt              # Text instructions (required)
├── final_frame.png          # Target state image (optional)
└── ground_truth.mp4        # Optional ground truth video
```

## Quick Start

```bash
# 1. Install
git clone https://github.com/hokindeng/VMEvalKit.git
cd VMEvalKit
pip install -e .

# 2. Setup models
bash setup/install_model.sh --model svd --validate

# 3. Organize your questions data (see format above)
mkdir -p ~/my_research/questions

# 4. Run inference
python examples/generate_videos.py --questions-dir ~/my_research/questions --output-dir ~/my_research/outputs --model svd --task chess maze

# 5. Run evaluation  
python examples/score_videos.py human --inference-dir ~/my_research/outputs --eval-output-dir ~/my_research/evaluations
```

## API Keys

Set in `.env` file:
```bash
cp env.template .env
# Edit .env with your API keys:
# LUMA_API_KEY=...
# OPENAI_API_KEY=...  
# GEMINI_API_KEY=...
```

## Adding Models

```python
# Inherit from ModelWrapper
from vmevalkit.models.base import ModelWrapper

class MyModelWrapper(ModelWrapper):
    def generate(self, image_path, text_prompt, **kwargs):
        # Your inference logic
        return {"success": True, "video_path": "...", ...}
```

Register in `vmevalkit/runner/MODEL_CATALOG.py`:
```python
"my-model": {
    "wrapper_module": "vmevalkit.models.my_model_inference",
    "wrapper_class": "MyModelWrapper", 
    "family": "MyCompany"
}
```

## License

Apache 2.0