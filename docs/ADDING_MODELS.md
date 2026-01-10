# Adding Models to VMEvalKit

VMEvalKit supports two model types with different integration approaches:

## 📋 Quick Reference

### Commercial API Models
1. Create `{provider}_inference.py` with Service + Wrapper classes
2. Add entry to `MODEL_CATALOG.py`
3. Set API keys in `.env`
4. Test with `examples/generate_videos.py`

### Open-Source Models  
1. Create `{model}_inference.py` with Service + Wrapper classes
2. Create `setup/models/{model-name}/setup.sh` installation script
3. Register checkpoints in `setup/lib/share.sh`
4. Add entry to `MODEL_CATALOG.py`
5. Run `bash setup/install_model.sh --model {model-name}` to install

| Aspect | Commercial APIs | Open-Source |
|--------|----------------|-------------|
| **Setup** | API key only | Full installation (10-30min) |
| **Storage** | None | 5-25 GB per model |
| **GPU** | Not required | Required (8-24GB VRAM) |
| **Examples** | Luma, Veo, Sora | LTX-Video, SVD, HunyuanVideo |

## 🏗️ Architecture

VMEvalKit uses a **Service + Wrapper pattern**:
- **Service**: Handles API calls or model inference
- **Wrapper**: Inherits from `ModelWrapper`, provides unified interface
- **Registry**: `MODEL_CATALOG.py` lists all models with dynamic loading paths
- **Setup**: Open-source models need `setup/models/{name}/setup.sh` scripts

## 📝 Required Interface

All models must inherit from `ModelWrapper` and implement:

```python
class YourModelWrapper(ModelWrapper):
    def generate(self, image_path, text_prompt, **kwargs) -> Dict[str, Any]:
        # Must return exactly these 8 fields:
        return {
            "success": bool,
            "video_path": str | None, 
            "error": str | None,
            "duration_seconds": float,
            "generation_id": str,
            "model": str,
            "status": str,
            "metadata": Dict[str, Any]
        }
```

## 🚀 Installation

### Commercial APIs
```bash
# Add API key to .env
echo 'YOUR_PROVIDER_API_KEY=your_key' >> .env
# Ready to use immediately!
```

### Open-Source Models
```bash
# Install model and dependencies
bash setup/install_model.sh --model your-model-name

# Test installation
python examples/generate_videos.py --model your-model-name --task-id test_0001
```

## 📦 Open-Source Model Setup

### Setup Script Template
Create `setup/models/{model-name}/setup.sh`:

```bash
#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"

MODEL="your-model-name"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers==4.25.1 diffusers==0.31.0

deactivate

print_section "Checkpoints"
download_checkpoint_by_path "${MODEL_CHECKPOINT_PATHS[$MODEL]}"

print_success "${MODEL} setup complete"
```

### Register in `setup/lib/share.sh`
```bash
# Add to OPENSOURCE_MODELS array
OPENSOURCE_MODELS+=("your-model-name")

# Add checkpoint info
CHECKPOINTS+=("your-model/model.ckpt|https://huggingface.co/.../model.ckpt|5.2GB")
MODEL_CHECKPOINT_PATHS["your-model-name"]="your-model/model.ckpt"
```

## 🔌 Registration

### Add to MODEL_CATALOG.py

```python
# In vmevalkit/runner/MODEL_CATALOG.py
YOUR_MODELS = {
    "your-model-v1": {
        "wrapper_module": "vmevalkit.models.your_inference",
        "wrapper_class": "YourWrapper", 
        "model": "v1",
        "description": "Your model description",
        "family": "YourProvider"
    }
}

# Add to AVAILABLE_MODELS
AVAILABLE_MODELS = {**EXISTING_MODELS, **YOUR_MODELS}
```

## ✅ Testing

```bash
# Test installation
bash setup/install_model.sh --model your-model-name

# Test inference  
python examples/generate_videos.py --model your-model-name --task-id test_0001

# Verify all 8 required fields in return dict
```

## 🔧 Key Requirements

- **Inherit from ModelWrapper**: Use abstract base class
- **Return 8 required fields**: success, video_path, error, duration_seconds, generation_id, model, status, metadata
- **Handle errors gracefully**: Return error dict, don't raise exceptions
- **Use environment variables**: For API keys (never hardcode)
- **Exact package versions**: Use `package==X.Y.Z` in setup scripts  
- **Temperature = 0**: Keep results stable and reproducible

## 📚 Study Examples

- **Commercial API**: `vmevalkit/models/luma_inference.py`
- **Open-Source**: `vmevalkit/models/svd_inference.py`, `vmevalkit/models/ltx_inference.py`
- **Setup Scripts**: `setup/models/*/setup.sh`

Ready to add your model? Follow the patterns above and test thoroughly! 🚀
