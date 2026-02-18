# VMEvalKit 🎥🧠

**Unified inference and evaluation framework for 37 video generation models.**

## Features

- **🚀 33 Models**: Unified interface for commercial APIs (Luma, Veo, Kling, Sora, Runway) + open-source (LTX-Video, LTX-2, HunyuanVideo, SVD, etc.)
- **⚖️ Evaluation Pipeline**: Human scoring (Gradio) + automated scoring (GPT-4O, InternVL, Qwen3-VL) + rule-based scoring (VBVR-Bench Rubrics)
- **☁️ Cloud Integration**: S3 + HuggingFace Hub support

## Data Format

Organize your questions outside VMEvalKit with the following structure:

```
questions/
└── {domain}_task/                    # task folder (e.g., chess_task, matching_object_task)
    ├── {domain}_0000/                # individual question folder
    │   ├── first_frame.png           # required: input image for video generation
    │   ├── prompt.txt                # required: text prompt describing the video
    │   ├── final_frame.png           # optional: expected final frame for evaluation
    │   └── ground_truth.mp4          # optional: reference video for evaluation
    ├── {domain}_0001/
    │   └── ...
    └── {domain}_0002/
        └── ...
```

**Example** with domain `chess`:
```
questions/
└── chess_task/
    ├── chess_0000/
    │   ├── first_frame.png
    │   ├── prompt.txt
    │   ├── final_frame.png
    │   └── ground_truth.mp4
    ├── chess_0001/
    │   └── ...
    └── chess_0002/
        └── ...
```

**Naming Convention:**
- **Task folder**: `{domain}_task` (e.g., `chess_task`, `matching_object_task`)
- **Question folders**: `{domain}_{i:04d}` where `i` is zero-padded (e.g., `chess_0000`, `chess_0064`). Padding automatically expands beyond 4 digits when needed—no dataset size limit.

## Quick Start

```bash
# 1. Install
git clone https://github.com/Video-Reason/VMEvalKit.git
cd VMEvalKit

python -m venv venv
source venv/bin/activate

pip install -e .

# 2. Setup models
# Option A: automated script (creates isolated venv per model)
bash setup/install_model.sh --model svd --validate
# Option B: manual install from requirements.txt
pip install -r setup/models/svd/requirements.txt
# NOTE: setup.sh pins specific torch/CUDA versions that may not match your
# environment. If installation fails, check setup/models/<model>/requirements.txt
# for the recommended dependency ranges and install manually.

# # 3. Organize your questions data (see format above)
# mkdir -p ~/my_research/questions

# 4. Run inference
python examples/generate_videos.py --questions-dir setup/test_assets/ --output-dir ./outputs --model svd
python examples/generate_videos.py --questions-dir setup/test_assets/ --output-dir ./outputs --model LTX-2
# 5. Run evaluation
# Create eval_config.json first:
echo '{"evaluator":"human","inference_dir":"~/my_research/outputs","eval_output_dir":"~/my_research/evaluations"}' > eval_config.json
python examples/score_videos.py --eval-config eval_config.json

# 6. Run VBVR-Bench rule-based (rubrics) evaluation (no API needed)
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs
```

## API Keys

Set in `.env` file:
```bash
cp env.template .env
# Edit .env with your API keys:
# LUMA_API_KEY=...
# OPENAI_API_KEY=...
# GEMINI_API_KEY=...
# KLING_API_KEY=...
# RUNWAYML_API_SECRET=...
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
