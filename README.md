# VMEvalKit 🎥🧠

**Unified inference and evaluation framework for 29+ video generation models.**

## Features

- **🚀 29+ Models**: Unified interface for commercial APIs (Luma, Veo, Sora, Runway) + open-source (LTX-Video, HunyuanVideo, DynamiCrafter, SVD, etc.)
- **⚖️ Evaluation Pipeline**: Human scoring (Gradio) + automated scoring (GPT-4O, InternVL)  
- **☁️ Cloud Integration**: S3 + HuggingFace Hub support

## Data Format

Organize your questions outside VMEvalKit in folders:
```
questions/{domain}_task/{task_id}/
├── first_frame.png   # required
├── prompt.txt        # required
├── final_frame.png   # optional
└── ground_truth.mp4  # optional
```

**Task IDs** use `{domain}_{i:04d}` (e.g., `chess_0064`, `chess_15000`). IDs are zero-padded for sorting and automatically expand beyond 4 digits when needed—no dataset size limit or code change required.

## Quick Start

```bash
# 1. Install
git clone https://github.com/Video-Reason/VMEvalKit.git
cd VMEvalKit

python -m venv venv
source venv/bin/activate

pip install -e .

# 2. Setup models
bash setup/install_model.sh --model svd --validate

# 3. Organize your questions data (see format above)
mkdir -p ~/my_research/questions

# 4. Run inference
python examples/generate_videos.py --questions-dir ~/my_research/questions --output-dir ~/my_research/outputs --model svd

# 5. Run evaluation  
# Create eval_config.json first:
echo '{"method": "human", "inference_dir": "~/my_research/outputs", "eval_output_dir": "~/my_research/evaluations"}' > eval_config.json
python examples/score_videos.py --eval-config eval_config.json
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