# Adding Models to VMEvalKit

VMEvalKit uses a **clean modular architecture** with dynamic loading, designed for scalability and easy model integration. This guide provides comprehensive instructions for integrating new video generation models.

## 📋 Quick Reference

### Commercial API-Based Models
1. Add entry to MODEL_CATALOG.py
2. Create wrapper class inheriting from ModelWrapper
3. Implement generate() method with API integration
4. Set up API keys in .env
5. No installation script needed!

### Open-Source Models  
1. Add entry to MODEL_CATALOG.py
2. Create wrapper class inheriting from ModelWrapper
3. Implement generate() method (usually subprocess-based)
4. Create setup script at `setup/models/{model-name}/setup.sh`
5. Register checkpoints in `setup/lib/share.sh`
6. Initialize git submodule if needed

### Comparison Table

| Aspect | Commercial API Models | Open-Source Models |
|--------|----------------------|-------------------|
| **Setup Time** | < 1 minute (just API key) | 10-30 minutes (full installation) |
| **Storage** | None (cloud-hosted) | 5-25 GB per model |
| **Dependencies** | requests/httpx only | torch, transformers, model-specific packages |
| **Environment** | Shared venv OK | Isolated venv per model required |
| **Inference** | API calls (async/polling) | Local subprocess or direct Python |
| **Cost** | Pay per generation | Free (compute cost only) |
| **GPU** | Not required | Usually required (8-24 GB VRAM) |
| **Setup Script** | ❌ Not needed | ✅ Required |
| **Checkpoints** | ❌ Cloud-hosted | ✅ Local downloads required |
| **Submodule** | ❌ Not needed | ⚠️ Sometimes required |
| **Examples** | Luma, Veo, Runway, Sora | DynamiCrafter, LTX-Video, SVD, WAN |

## 🏗️ Architecture Overview

### Two Model Categories

VMEvalKit supports two distinct types of models:

#### 1. **Commercial API-Based Models** 🌐
- **Examples**: Luma Ray 2, Google Veo, Runway Gen-4, OpenAI Sora
- **Integration**: HTTP API calls with authentication
- **Setup**: Environment variables for API keys only
- **Deployment**: No local installation, instant availability
- **Pattern**: Service class handles API logic, Wrapper provides interface

#### 2. **Open-Source Models** 🔓
- **Examples**: DynamiCrafter, VideoCrafter, LTX-Video, HunyuanVideo, WAN
- **Integration**: Subprocess execution or direct Python imports
- **Setup**: Dedicated setup scripts with virtual environments
- **Deployment**: Local installation with model checkpoints
- **Pattern**: Service class manages inference, Wrapper provides interface
- **Additional Components**:
  - Setup script: `setup/models/{model-name}/setup.sh`
  - Virtual environment: `envs/{model-name}/`
  - Model weights: `weights/{model-family}/`
  - Git submodules: `submodules/{model-repo}/` (optional)

### Core System Design

```
vmevalkit/
├── runner/
│   ├── MODEL_CATALOG.py    # 📋 Pure model registry (40+ models, 11 families)
│   └── inference.py        # 🎭 Orchestration with dynamic loading  
├── models/
│   ├── base.py            # 🔧 Abstract ModelWrapper interface
│   ├── luma_inference.py  # Commercial: LumaInference + LumaWrapper
│   ├── veo_inference.py   # Commercial: VeoService + VeoWrapper
│   ├── dynamicrafter_inference.py  # Open-source: DynamiCrafterService + Wrapper
│   ├── ltx_inference.py   # Open-source: LTXVideoService + Wrapper
│   └── ...                # Each provider: Service + Wrapper pattern
setup/
├── models/
│   ├── dynamicrafter-512/
│   │   └── setup.sh       # Installation script for DynamiCrafter 512p
│   ├── ltx-video/
│   │   └── setup.sh       # Installation script for LTX-Video
│   └── ...
├── lib/
│   └── share.sh           # Shared utilities and model registry
└── install_model.sh       # Master installation orchestrator
```

### Key Architecture Principles

1. **Separation of Concerns**: Registry, orchestration, and implementations are completely separate
2. **Dynamic Loading**: Models loaded on-demand using string module paths to avoid circular imports
3. **Family Organization**: Models grouped by provider families for logical organization
4. **Consistent Interface**: All wrappers inherit from `ModelWrapper` abstract base class
5. **Service + Wrapper Pattern**: Service handles API/inference logic, Wrapper provides VMEvalKit interface

### Model Catalog System

`runner/MODEL_CATALOG.py` contains only model definitions - no imports or logic:

```python
LUMA_MODELS = {
    "luma-ray-2": {
        "wrapper_module": "vmevalkit.models.luma_inference",  # String path
        "wrapper_class": "LumaWrapper",                       # Class name  
        "service_class": "LumaInference",                     # Service class
        "model": "ray-2",                                     # Actual model ID
        "description": "Luma Ray 2 - Latest model",
        "family": "Luma Dream Machine"
    },
    "luma-ray-flash-2": {
        "wrapper_module": "vmevalkit.models.luma_inference",
        "wrapper_class": "LumaWrapper",
        "service_class": "LumaInference",
        "model": "ray-flash-2",                               # Different model ID
        "description": "Luma Ray Flash 2 - Faster generation",
        "family": "Luma Dream Machine"
    }
}

# Additional model-specific arguments can be included
WAVESPEED_MODELS = {
    "veo-3.1-720p": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "Veo31Wrapper",
        "service_class": "Veo31Service",
        "model": "veo-3.1-720p",
        "args": {"resolution": "720p"},    # Model-specific initialization args
        "description": "Google Veo 3.1 - 720p with audio generation",
        "family": "Google Veo 3.1"
    }
}
```

### Dynamic Loading System

Models are loaded on-demand without importing them upfront:

```python
def _load_model_wrapper(model_name: str) -> Type[ModelWrapper]:
    # 1. Look up model in catalog
    config = AVAILABLE_MODELS[model_name] 
    
    # 2. Dynamic import using string path
    module = importlib.import_module(config["wrapper_module"])
    wrapper_class = getattr(module, config["wrapper_class"])
    
    # 3. Return class (not instance)
    return wrapper_class
```

### Base Interface System

All wrappers **must** inherit from `ModelWrapper`:

```python
from abc import ABC, abstractmethod

class ModelWrapper(ABC):
    def __init__(self, model: str, output_dir: str = "./data/outputs", **kwargs):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs  # Store extra parameters
    
    @abstractmethod
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Standardized interface for all video generation models."""
        pass
```

**Required Return Format** (all models must return exactly this structure):

```python
{
    "success": bool,               # Whether generation succeeded
    "video_path": str | None,      # Path to generated video file  
    "error": str | None,           # Error message if failed
    "duration_seconds": float,     # Time taken for generation
    "generation_id": str,          # Unique identifier
    "model": str,                  # Model name used
    "status": str,                 # "success", "failed", etc.
    "metadata": Dict[str, Any]    # Additional metadata (prompt, image_path, etc.)
}
```

## ⚡ Requirements

### Functional Requirements
✅ **MUST support: Image + Text → Video** (essential for reasoning evaluation)  
✅ **Inherit from ModelWrapper**: Use abstract base class for consistency  
✅ **Unified interface**: `generate(image_path, text_prompt, duration, output_filename, **kwargs)`  
✅ **Parameter separation**: Constructor for config, generate() for runtime inputs
✅ **Return all required fields**: Every field in the return format is mandatory

### Technical Requirements
✅ **Handle authentication properly**: Use environment variables for API keys  
✅ **Support async operations**: Use asyncio.run() for async services  
✅ **Consistency**: Temperature always be set to zero to keep the results stable

### Setup Requirements (Open-Source Models Only)
✅ **Setup script**: Create `setup/models/{model-name}/setup.sh`  
✅ **Exact versions**: Always use `package==X.Y.Z` format in pip install  
✅ **Virtual environment**: One isolated venv per model  
✅ **Checkpoint registration**: Add to `setup/lib/share.sh`  
✅ **Submodule management**: Pin to specific commit, not floating HEAD

## 🚀 Model Installation & Deployment

### Commercial API Models - Quick Start

```bash
# 1. Set up API key
echo 'NEWPROVIDER_API_KEY=your_api_key_here' >> .env

# 2. Model is ready to use immediately!
python examples/generate_videos.py --model newprovider-v1 --task-id tests_0001
```

**No installation needed!** Commercial models are instantly available once API keys are configured.

### Open-Source Models - Installation Workflow

```bash
# 1. Initialize submodule (if model uses one)
git submodule update --init --recursive

# 2. Run installation script
bash setup/install_model.sh opensourcemodel-512

# Behind the scenes, this:
# - Creates virtual environment at envs/opensourcemodel-512/
# - Installs exact package versions
# - Downloads model checkpoints to weights/
# - Verifies installation

# 3. Model is ready to use
python examples/generate_videos.py --model opensourcemodel-512 --task-id tests_0001
```

### Installation Script Master Orchestrator

The `setup/install_model.sh` script handles all model installations:

```bash
# Install single model
bash setup/install_model.sh dynamicrafter-512

# Install multiple models
bash setup/install_model.sh ltx-video svd videocrafter2-512

# Install all open-source models
bash setup/install_model.sh --all-opensource

# Validate after installation
bash setup/install_model.sh dynamicrafter-512 --validate
```

### Directory Structure After Installation

```
VMEvalKit/
├── envs/
│   ├── opensourcemodel-512/      # Virtual environment
│   │   ├── bin/python             # Model-specific Python
│   │   └── lib/python3.10/        # Isolated packages
│   ├── dynamicrafter-512/
│   └── ltx-video/
├── weights/
│   ├── opensourcemodel/
│   │   └── model.safetensors      # 5.2GB checkpoint
│   ├── dynamicrafter/
│   │   ├── dynamicrafter_256_v1/
│   │   ├── dynamicrafter_512_v1/
│   │   └── dynamicrafter_1024_v1/
│   └── ltx-video/
├── submodules/
│   ├── OpenSourceModel/           # Git submodule
│   ├── DynamiCrafter/
│   └── LTX-Video/
└── .env                           # API keys for commercial models
```

## 📦 Open-Source Model Integration

### Overview

Open-source models require additional setup beyond the wrapper class:
1. **Installation Script**: Automates environment and dependency setup
2. **Virtual Environment**: Isolated Python environment per model
3. **Model Checkpoints**: Downloaded weights and configuration files
4. **Submodule Integration**: Optional git submodule for model code

### Directory Structure for Open-Source Models

```
your-model-name/
├── Setup Components:
│   ├── setup/models/your-model-name/
│   │   ├── setup.sh              # Installation script
│   │   └── test.sh               # Validation script (optional)
│   └── setup/lib/share.sh        # Register checkpoints here
├── Runtime Components:
│   ├── vmevalkit/models/
│   │   └── yourmodel_inference.py  # Service + Wrapper classes
│   ├── envs/your-model-name/     # Virtual environment (created by setup)
│   ├── weights/your-model/       # Model checkpoints (downloaded by setup)
│   └── submodules/YourModel/     # Git submodule (if needed)
```

### Creating a Setup Script

Setup scripts follow a standardized pattern using shared utilities from `setup/lib/share.sh`:

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="your-model-name"

print_section "System Dependencies"
ensure_ffmpeg_dependencies

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers==4.25.1 diffusers==0.31.0 accelerate==1.2.1
pip install -q Pillow==9.5.0 numpy==1.24.2 opencv-python==4.8.1.78 pydantic==2.12.5 pydantic-settings==2.12.0 python-dotenv==1.2.1

deactivate

print_section "Checkpoints"
download_checkpoint_by_path "${MODEL_CHECKPOINT_PATHS[$MODEL]}"

print_success "${MODEL} setup complete"
```

**Style Guidelines:**
- Use compact style: group related packages on same line
- No inline comments in the Dependencies section
- Keep it consistent with other models (see `dynamicrafter-256`, `svd`, `videocrafter2-512`)
- Always use exact versions: `package==X.Y.Z`

**Key Functions from `share.sh`:**
- `create_model_venv "$MODEL"`: Creates fresh virtual environment
- `activate_model_venv "$MODEL"`: Activates virtual environment
- `download_checkpoint_by_path "$PATH"`: Downloads registered checkpoint
- `ensure_ffmpeg_dependencies`: Installs FFmpeg system libraries
- `print_section`, `print_success`, etc.: Formatted output

### Registering Model Checkpoints

Add your model's checkpoints to `setup/lib/share.sh`:

```bash
# In setup/lib/share.sh

# 1. Add to CHECKPOINTS array
declare -a CHECKPOINTS=(
    # Existing checkpoints...
    "your-model/model.ckpt|https://huggingface.co/org/repo/resolve/main/model.ckpt|5.2GB"
    "your-model/config.yaml|https://huggingface.co/org/repo/resolve/main/config.yaml|4KB"
)

# 2. Add to MODEL_CHECKPOINT_PATHS mapping
declare -A MODEL_CHECKPOINT_PATHS=(
    # Existing mappings...
    ["your-model-name"]="your-model/model.ckpt"
)

# 3. Add to OPENSOURCE_MODELS list
declare -a OPENSOURCE_MODELS=(
    # Existing models...
    "your-model-name"
)
```

### Git Submodules for Open-Source Models

If your model requires a specific codebase (e.g., from GitHub):

```bash
# Add submodule to your project
cd /path/to/VMEvalKit
git submodule add https://github.com/org/YourModel.git submodules/YourModel
git submodule update --init --recursive

# In your inference file, add the submodule to Python path:
import sys
from pathlib import Path

YOURMODEL_PATH = Path(__file__).parent.parent.parent / "submodules" / "YourModel"
sys.path.insert(0, str(YOURMODEL_PATH))

# Now you can import from the submodule
from yourmodel import YourModelPipeline
```

**Submodule Management:**
- Submodules are tracked separately from main repo
- Users must run `git submodule update --init --recursive`
- Keep submodule commits stable (pin to specific versions)
- Document any required patches or modifications

### Subprocess Pattern for Open-Source Models

Many open-source models use subprocess execution for isolation:

```python
class YourModelService:
    def generate_video(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        **kwargs
    ) -> bool:
        """Generate video using subprocess for memory isolation."""
        
        # 1. Create inference script
        script_path = self._create_inference_script(
            image_path=image_path,
            prompt=prompt,
            output_path=output_path,
            **kwargs
        )
        
        # 2. Get virtual environment Python
        venv_python = self._get_venv_python()
        
        # 3. Run in subprocess with timeout
        try:
            result = subprocess.run(
                [venv_python, str(script_path)],
                cwd=str(YOURMODEL_PATH),
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes
                check=True
            )
            return True
        except subprocess.TimeoutExpired:
            raise TimeoutError("Generation timed out")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Generation failed: {e.stderr}")
        finally:
            # Clean up temp script
            if script_path.exists():
                script_path.unlink()
    
    def _get_venv_python(self) -> str:
        """Get path to model's virtual environment Python."""
        venv_path = Path(__file__).parent.parent.parent / "envs" / self.model_id
        python_path = venv_path / "bin" / "python"
        
        if not python_path.exists():
            raise FileNotFoundError(
                f"Virtual environment not found: {venv_path}\n"
                f"Run: bash setup/install_model.sh {self.model_id}"
            )
        
        return str(python_path)
```

**Why Subprocess Pattern?**
- **Memory Isolation**: Each inference runs in separate process
- **Environment Isolation**: Uses model-specific virtual environment
- **Dependency Isolation**: Avoids conflicts between model dependencies
- **Stability**: Process crash doesn't affect main program

### Testing Your Setup Script

```bash
# Test installation
bash setup/models/your-model-name/setup.sh

# Verify virtual environment
ls envs/your-model-name/bin/python

# Verify checkpoints
ls weights/your-model/

# Run validation
python examples/generate_videos.py --model your-model-name --task-id tests_0001
```

## 🔌 Registration Process

### Step 1: Add to MODEL_CATALOG.py

#### For Commercial API-Based Models

```python
# In vmevalkit/runner/MODEL_CATALOG.py

# Define your model family
{PROVIDER}_MODELS = {
    "{provider}-model-v1": {
        "wrapper_module": "vmevalkit.models.{provider}_inference",
        "wrapper_class": "{Provider}Wrapper",
        "service_class": "{Provider}Service",
        "model": "model-v1",  # Actual model ID used by API
        "description": "Brief model description",
        "family": "{Provider}"
    },
    "{provider}-model-v2": {
        "wrapper_module": "vmevalkit.models.{provider}_inference",
        "wrapper_class": "{Provider}Wrapper",
        "service_class": "{Provider}Service", 
        "model": "model-v2",
        "args": {"special_param": "value"},  # Optional model-specific args
        "description": "V2 with improvements",
        "family": "{Provider}"
    }
}
```

#### For Open-Source Models

```python
# In vmevalkit/runner/MODEL_CATALOG.py

# Under "OPEN-SOURCE MODELS (SUBMODULES)" section
YOURMODEL_MODELS = {
    "your-model-512": {
        "wrapper_module": "vmevalkit.models.yourmodel_inference",
        "wrapper_class": "YourModelWrapper",
        "service_class": "YourModelService",
        "model": "your-model-512",  # Model variant identifier
        "description": "Your Model 512p - Image to video generation",
        "family": "YourModel"
    },
    "your-model-1024": {
        "wrapper_module": "vmevalkit.models.yourmodel_inference",
        "wrapper_class": "YourModelWrapper",
        "service_class": "YourModelService",
        "model": "your-model-1024",
        "args": {"resolution": "1024"},  # Variant-specific config
        "description": "Your Model 1024p - Higher resolution",
        "family": "YourModel"
    }
}
```

#### Add to Combined Registries

```python
# At the bottom of MODEL_CATALOG.py

# Add to combined registries
AVAILABLE_MODELS = {
    # Commercial models
    **LUMA_MODELS,
    **VEO_MODELS,
    **{PROVIDER}_MODELS,  # Your commercial models
    
    # Open-source models
    **LTX_VIDEO_MODELS,
    **DYNAMICRAFTER_MODELS,
    **YOURMODEL_MODELS,  # Your open-source models
    # ... other models
}

MODEL_FAMILIES = {
    # Commercial
    "Luma Dream Machine": LUMA_MODELS,
    "Google Veo": VEO_MODELS,
    "{Provider}": {PROVIDER}_MODELS,
    
    # Open-source
    "LTX-Video": LTX_VIDEO_MODELS,
    "DynamiCrafter": DYNAMICRAFTER_MODELS,
    "YourModel": YOURMODEL_MODELS,
    # ... other families
}
```

### Step 2: Update models/__init__.py

```python
# In vmevalkit/models/__init__.py

# Import your new model classes
from .{provider}_inference import {Provider}Service, {Provider}Wrapper

__all__ = [
    # Existing exports
    "LumaInference", "LumaWrapper",
    "VeoService", "VeoWrapper",
    # Add your exports
    "{Provider}Service", "{Provider}Wrapper",
]
```

### Step 3: Set Environment Variables

Create or update `.env` file in project root:

```bash
# API Keys for various providers
LUMA_API_KEY=your_luma_key
WAVESPEED_API_KEY=your_wavespeed_key
{PROVIDER}_API_KEY=your_{provider}_key

# Google Cloud for Veo (if using)
GCP_PROJECT_ID=your-project-id
GCP_LOCATION=us-central1
```

## ✅ Testing Your Integration

### Quick Testing (Recommended)

Install and validate in one command:

```bash
# Install and test any model
bash setup/install_model.sh --model your-model-name --validate

# Test multiple models
bash setup/install_model.sh --opensource --validate
bash setup/install_model.sh --all --validate
```

This runs the full validation:
1. Installs the model (creates venv, downloads weights)
2. Generates 2 test videos using `examples/generate_videos.py`
3. Verifies output quality
4. Reports pass/fail

### Manual Testing Steps

For detailed verification without using `--validate`:

```bash
# 1. Install the model
bash setup/install_model.sh --model your-model-name

# 2. Verify virtual environment was created
ls -la envs/your-model-name/bin/python
source envs/your-model-name/bin/activate
python -c "import torch; print(torch.__version__)"
deactivate

# 3. Verify checkpoints were downloaded (open-source only)
ls -lh weights/your-model/

# 4. Verify submodule (if applicable)
ls -la submodules/YourModel/

# 5. Run manual validation
python examples/generate_videos.py \
    --model your-model-name \
    --task-id tests_0001 tests_0002

# 6. Check validation output
ls -la data/outputs/pilot_experiment/your-model-name/tests_task/
```

### Testing Commercial API Models

For commercial models, verify API key first:

```bash
# 1. Verify API key is set
echo $YOUR_PROVIDER_API_KEY

# 2. Install and test
bash setup/install_model.sh --model your-provider-model --validate

# Or test manually
python examples/generate_videos.py \
    --model your-provider-model \
    --task-id tests_0001
```

### Basic Functionality Test (Both Types)

```python
# test_model.py
from vmevalkit.runner.inference import run_inference, InferenceRunner
from vmevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS

# 1. Verify model is registered
print(f"Model registered: {'your-model-name' in AVAILABLE_MODELS}")
print(f"Config: {AVAILABLE_MODELS.get('your-model-name')}")

# 2. Test dynamic loading
from vmevalkit.runner.inference import _load_model_wrapper
wrapper_class = _load_model_wrapper("your-model-name")
print(f"Loaded class: {wrapper_class.__name__}")

# 3. Test direct inference
result = run_inference(
    model_name="your-model-name", 
    image_path="test_images/chess_0000_first.png",
    text_prompt="Move the white queen to checkmate the black king",
    output_dir="./test_outputs"
)

# 4. Verify result format
required_fields = [
    "success", "video_path", "error", "duration_seconds",
    "generation_id", "model", "status", "metadata"
]
missing = [f for f in required_fields if f not in result]
assert not missing, f"Missing fields: {missing}"

# 5. Test via runner (production usage)
runner = InferenceRunner(output_dir="./test_outputs")
result = runner.run(
    model_name="your-model-name",
    image_path="test_images/maze_0000_first.png",
    text_prompt="Navigate through the maze to reach the goal"
)
print(f"Inference directory: {result['inference_dir']}")
```

### Integration Test Script

```python
# integration_test.py
import json
from pathlib import Path
from vmevalkit.runner.inference import InferenceRunner

def test_model_integration(model_name: str):
    """Comprehensive integration test."""
    
    runner = InferenceRunner()
    
    # Test different task types
    test_cases = [
        {
            "image": "data/questions/chess_task/chess_0000/first_frame.png",
            "prompt": "Find the checkmate move"
        },
        {
            "image": "data/questions/maze_task/maze_0000/first_frame.png", 
            "prompt": "Navigate to the red flag"
        }
    ]
    
    results = []
    for i, test in enumerate(test_cases):
        print(f"\nTest {i+1}: {test['prompt'][:30]}...")
        
        try:
            result = runner.run(
                model_name=model_name,
                image_path=test["image"],
                text_prompt=test["prompt"]
            )
            
            # Validate required fields
            required_fields = [
                "success", "video_path", "error", "duration_seconds",
                "generation_id", "model", "status", "metadata"
            ]
            
            missing_fields = [f for f in required_fields if f not in result]
            if missing_fields:
                print(f"  ❌ Missing fields: {missing_fields}")
            else:
                print(f"  ✅ All required fields present")
            
            if result["success"]:
                print(f"  ✅ Video generated: {result['video_path']}")
            else:
                print(f"  ❌ Failed: {result['error']}")
            
            results.append(result)
            
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            results.append({"error": str(e)})
    
    # Save test results
    output_file = Path(f"test_results_{model_name}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Summary
    successful = sum(1 for r in results if r.get("success"))
    print(f"\nSummary: {successful}/{len(test_cases)} tests passed")

# Run test
if __name__ == "__main__":
    test_model_integration("your-model-name")
```

## 🔧 Common Issues and Solutions

### General Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `Model not found` | Not registered in catalog | Add to MODEL_CATALOG.py and AVAILABLE_MODELS |
| `ImportError: No module named` | Wrong module path | Check `wrapper_module` string in catalog |
| `AttributeError: module has no attribute` | Wrong class name | Verify `wrapper_class` matches actual class |
| `generate() missing required field` | Incomplete return dictionary | Ensure all 8 required fields are returned |
| `Image not found` | Invalid image path | Use absolute paths or verify relative path |
| `Timeout during generation` | Long processing time | Increase timeout in subprocess/httpx calls |

### Commercial API-Based Models

| Issue | Cause | Solution |
|-------|-------|----------|
| `API key not found` | Missing environment variable | Set `{PROVIDER}_API_KEY` in .env file |
| `Authentication failed` | Invalid or expired API key | Verify API key is correct and active |
| `Rate limit exceeded` | Too many requests | Implement exponential backoff, reduce concurrency |
| `Invalid model ID` | Wrong model identifier | Check actual model ID from provider docs |
| `Request timeout` | Slow API response | Increase httpx timeout value |
| `403 Forbidden` | Insufficient permissions | Check API key has required permissions |
| `422 Unprocessable Entity` | Invalid request parameters | Validate image format, prompt length, etc. |

### Open-Source Models

| Issue | Cause | Solution |
|-------|-------|----------|
| `Virtual environment not found` | Setup script not run | Run `bash setup/install_model.sh {model-name}` |
| `Checkpoint not found` | Download failed or wrong path | Check `weights/` directory, re-run setup |
| `Submodule not found` | Git submodule not initialized | Run `git submodule update --init --recursive` |
| `CUDA out of memory` | Model too large for GPU | Reduce batch size, use gradient checkpointing |
| `ModuleNotFoundError` in subprocess | Wrong virtual environment | Verify `_get_venv_python()` points to correct venv |
| `ImportError: libav` | Missing FFmpeg libraries | Run `ensure_ffmpeg_dependencies` in setup script |
| `Dependency conflict` | Package version mismatch | Use exact versions (==X.Y.Z) in setup script |
| `Model weights not loading` | Incompatible checkpoint version | Verify checkpoint matches model version |
| `Subprocess crash` | Code error in inference script | Check subprocess stderr output for traceback |
| `Permission denied: /weights/` | Insufficient permissions | Check directory permissions: `chmod -R u+w weights/` |

## 🎯 Best Practices

### General (Both Model Types) ✅

- **Inherit from ModelWrapper**: Always use the base class for consistency
- **Return all required fields**: Every field in the standard format is mandatory
- **Handle errors gracefully**: Return error dictionary with success=False, don't raise
- **Validate inputs early**: Check image exists before processing
- **Use logging**: Import and use logger for debugging information
- **Clean up resources**: Close clients, delete temp files, free GPU memory
- **Document model variants**: Clear descriptions in MODEL_CATALOG.py
- **Test thoroughly**: Use the integration test scripts before committing
- **Set temperature to 0**: Keep results stable and reproducible

### Commercial API Models ✅

- **Use environment variables**: For API keys and secrets (NEVER hardcode)
- **Support async properly**: Use `asyncio.run()` for async services
- **Implement retry logic**: Handle transient API failures with exponential backoff
- **Set reasonable timeouts**: Don't wait forever for API responses
- **Upload images securely**: Use temporary signed URLs when possible
- **Handle rate limits**: Implement proper backoff strategies
- **Validate API responses**: Check for errors before processing
- **Log API interactions**: Track request IDs for debugging

### Open-Source Models ✅

- **Use exact package versions**: Always specify `package==X.Y.Z` in setup scripts
- **Isolate environments**: One virtual environment per model
- **Use subprocess pattern**: Isolate model inference for memory safety
- **Verify checkpoints**: Check file integrity after download
- **Document GPU requirements**: Specify minimum VRAM needed
- **Handle CUDA errors**: Catch and report GPU-related failures
- **Clean up after inference**: Delete temporary scripts and intermediate files
- **Pin submodule commits**: Don't use floating HEAD for submodules
- **Test installation script**: Verify setup.sh works on clean system

### DON'T ❌

#### General
- **Skip the base class**: All wrappers must inherit from ModelWrapper
- **Import models in catalog**: Use string paths only for dynamic loading
- **Mix concerns**: Keep Service (logic) and Wrapper (interface) separate
- **Raise exceptions in generate()**: Return error dictionary instead
- **Forget to save videos**: Always write output to self.output_dir
- **Skip input validation**: Always check image path exists
- **Use relative imports**: Use absolute imports for clarity

#### Commercial API Models
- **Hardcode credentials**: Always use environment variables
- **Ignore API errors**: Handle all possible API failure modes
- **Skip timeout handling**: APIs can hang indefinitely
- **Forget metadata**: Include helpful debugging info in responses

#### Open-Source Models
- **Use floating package versions**: `pip install package` without version is unstable
- **Mix dependencies**: Don't install multiple models in same venv
- **Skip FFmpeg check**: Many models need video codec libraries
- **Ignore GPU memory**: Check and report OOM errors properly
- **Forget submodule init**: Document submodule initialization steps
- **Hardcode paths**: Use Path(__file__) for relative path construction
- **Run without venv**: Always use model-specific virtual environment

## 📚 Real Examples to Study

Study these actual implementations for patterns:

### Commercial API-Based Models

1. **API with Polling**: `vmevalkit/models/luma_inference.py`
   - S3 image upload for URL generation
   - Async polling for completion
   - Retry logic with tenacity
   - Error handling for API failures

2. **Google Cloud Auth**: `vmevalkit/models/veo_inference.py`
   - Application Default Credentials
   - Fallback to gcloud CLI
   - Project ID handling
   - Gemini API integration

3. **REST API Pattern**: `vmevalkit/models/runway_inference.py`
   - Simple HTTP POST/GET workflow
   - Task polling with status checks
   - Download and save video files

4. **Multiple Model Variants**: `vmevalkit/models/wavespeed_inference.py`
   - Enum for model selection
   - Different wrapper classes per variant
   - Shared service class for common logic

### Open-Source Models

1. **Subprocess Pattern**: `vmevalkit/models/ltx_inference.py`
   - Clean subprocess execution
   - Virtual environment integration
   - Config file generation
   - Timeout handling

2. **Direct Python Import**: `vmevalkit/models/svd_inference.py`
   - Direct diffusers pipeline usage
   - GPU memory management
   - Torch optimizations
   - Model loading from HuggingFace

3. **Submodule Integration**: `vmevalkit/models/dynamicrafter_inference.py`
   - Git submodule code import
   - Temporary script generation
   - Config file mapping per variant
   - Checkpoint path handling

4. **Complex Setup Script**: `setup/models/hunyuan-video-i2v/setup.sh`
   - Multi-stage installation
   - Dependency conflict resolution
   - Large checkpoint download
   - System library requirements

5. **Distributed Inference**: `vmevalkit/models/morphic_inference.py`
   - Multi-GPU support
   - Torchrun subprocess pattern
   - Complex checkpoint dependencies
   - LoRA weight handling

## 🎯 Complete Integration Examples

### Example 1: Commercial API Model (REST API Pattern)

**File: `vmevalkit/models/newprovider_inference.py`**

```python
"""NewProvider API Integration for VMEvalKit"""

import os
import httpx
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .base import ModelWrapper

class NewProviderService:
    """Service class for NewProvider API."""
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key or os.getenv("NEWPROVIDER_API_KEY")
        if not self.api_key:
            raise ValueError("NEWPROVIDER_API_KEY not found")
        
        self.base_url = "https://api.newprovider.com/v1"
        self.client = httpx.Client(timeout=300.0)
    
    def generate_video(
        self,
        image_url: str,
        prompt: str,
        duration: float = 5.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Submit video generation request."""
        
        response = self.client.post(
            f"{self.base_url}/generate",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "image_url": image_url,
                "prompt": prompt,
                "duration": duration,
                "temperature": 0  # Keep results stable
            }
        )
        response.raise_for_status()
        return response.json()
    
    def poll_status(self, generation_id: str, timeout: int = 600) -> Dict[str, Any]:
        """Poll for generation completion."""
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            response = self.client.get(
                f"{self.base_url}/status/{generation_id}",
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            response.raise_for_status()
            data = response.json()
            
            if data["status"] == "completed":
                return data
            elif data["status"] == "failed":
                raise RuntimeError(f"Generation failed: {data.get('error')}")
            
            time.sleep(5)
        
        raise TimeoutError(f"Generation timed out after {timeout}s")

class NewProviderWrapper(ModelWrapper):
    """Wrapper for NewProvider API."""
    
    def __init__(self, model: str, output_dir: str = "./data/outputs", **kwargs):
        super().__init__(model, output_dir, **kwargs)
        self.service = NewProviderService(**kwargs)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video from image and text prompt."""
        
        start_time = time.time()
        
        try:
            # 1. Upload image to get URL
            image_url = self._upload_image(image_path)
            
            # 2. Submit generation request
            result = self.service.generate_video(
                image_url=image_url,
                prompt=text_prompt,
                duration=duration,
                **kwargs
            )
            generation_id = result["id"]
            
            # 3. Poll for completion
            completed = self.service.poll_status(generation_id)
            video_url = completed["video_url"]
            
            # 4. Download video
            output_path = self._download_video(video_url, output_filename)
            
            return {
                "success": True,
                "video_path": str(output_path),
                "error": None,
                "duration_seconds": time.time() - start_time,
                "generation_id": generation_id,
                "model": self.model,
                "status": "success",
                "metadata": {
                    "prompt": text_prompt,
                    "image_path": str(image_path),
                    "duration": duration
                }
            }
        
        except Exception as e:
            return {
                "success": False,
                "video_path": None,
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "generation_id": "",
                "model": self.model,
                "status": "failed",
                "metadata": {"prompt": text_prompt, "image_path": str(image_path)}
            }
```

**Registration in `MODEL_CATALOG.py`:**

```python
NEWPROVIDER_MODELS = {
    "newprovider-v1": {
        "wrapper_module": "vmevalkit.models.newprovider_inference",
        "wrapper_class": "NewProviderWrapper",
        "service_class": "NewProviderService",
        "model": "v1",
        "description": "NewProvider V1 - High-quality video generation",
        "family": "NewProvider"
    }
}

# Add to AVAILABLE_MODELS and MODEL_FAMILIES...
```

### Example 2: Open-Source Model (Subprocess Pattern)

**File: `vmevalkit/models/opensourcemodel_inference.py`**

```python
"""OpenSourceModel Integration for VMEvalKit"""

import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .base import ModelWrapper
import time

OPENSOURCEMODEL_PATH = Path(__file__).parent.parent.parent / "submodules" / "OpenSourceModel"
VMEVAL_ROOT = Path(__file__).parent.parent.parent

class OpenSourceModelService:
    """Service for open-source model inference."""
    
    def __init__(self, model_id: str = "opensourcemodel-512", **kwargs):
        self.model_id = model_id
        self.kwargs = kwargs
        
        # Verify submodule exists
        if not OPENSOURCEMODEL_PATH.exists():
            raise FileNotFoundError(
                f"Submodule not found: {OPENSOURCEMODEL_PATH}\n"
                f"Run: git submodule update --init --recursive"
            )
    
    def generate_video(
        self,
        image_path: str,
        prompt: str,
        output_path: str,
        **kwargs
    ) -> bool:
        """Generate video using subprocess."""
        
        # 1. Create inference script
        script_path = self._create_inference_script(
            image_path, prompt, output_path, **kwargs
        )
        
        # 2. Get virtual environment Python
        venv_python = self._get_venv_python()
        
        # 3. Run in subprocess
        try:
            result = subprocess.run(
                [venv_python, str(script_path)],
                cwd=str(OPENSOURCEMODEL_PATH),
                capture_output=True,
                text=True,
                timeout=600,
                check=True
            )
            return True
        finally:
            script_path.unlink()
    
    def _create_inference_script(self, image_path: str, prompt: str, 
                                 output_path: str, **kwargs) -> Path:
        """Create temporary inference script."""
        
        script_content = f'''
import sys
sys.path.insert(0, "{OPENSOURCEMODEL_PATH}")

from opensourcemodel import Pipeline
import torch

# Load model
pipeline = Pipeline.from_pretrained(
    "{VMEVAL_ROOT}/weights/opensourcemodel/",
    torch_dtype=torch.float16
).to("cuda")

# Generate
output = pipeline(
    image="{image_path}",
    prompt="{prompt}",
    num_frames=16,
    guidance_scale=7.5
)

# Save
output.save("{output_path}")
'''
        
        script_path = Path(tempfile.mktemp(suffix=".py"))
        script_path.write_text(script_content)
        return script_path
    
    def _get_venv_python(self) -> str:
        """Get virtual environment Python path."""
        venv_path = VMEVAL_ROOT / "envs" / self.model_id
        python_path = venv_path / "bin" / "python"
        
        if not python_path.exists():
            raise FileNotFoundError(
                f"Virtual environment not found: {venv_path}\n"
                f"Run: bash setup/install_model.sh {self.model_id}"
            )
        
        return str(python_path)

class OpenSourceModelWrapper(ModelWrapper):
    """Wrapper for open-source model."""
    
    def __init__(self, model: str, output_dir: str = "./data/outputs", **kwargs):
        super().__init__(model, output_dir, **kwargs)
        self.service = OpenSourceModelService(model_id=model, **kwargs)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video from image and text prompt."""
        
        start_time = time.time()
        output_path = self.output_dir / (output_filename or f"{self.model}_{int(time.time())}.mp4")
        
        try:
            success = self.service.generate_video(
                image_path=str(image_path),
                prompt=text_prompt,
                output_path=str(output_path),
                **kwargs
            )
            
            return {
                "success": success,
                "video_path": str(output_path) if success else None,
                "error": None,
                "duration_seconds": time.time() - start_time,
                "generation_id": output_path.stem,
                "model": self.model,
                "status": "success" if success else "failed",
                "metadata": {"prompt": text_prompt, "image_path": str(image_path)}
            }
        
        except Exception as e:
            return {
                "success": False,
                "video_path": None,
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "generation_id": "",
                "model": self.model,
                "status": "failed",
                "metadata": {"prompt": text_prompt, "image_path": str(image_path)}
            }
```

**Setup Script: `setup/models/opensourcemodel-512/setup.sh`**

```bash
#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../lib/share.sh"

MODEL="opensourcemodel-512"

print_section "System Dependencies"
ensure_ffmpeg_dependencies

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers==4.25.1 diffusers==0.31.0 accelerate==1.2.1
pip install -q Pillow==9.5.0 numpy==1.24.2 opencv-python==4.8.1.78 pydantic==2.12.5 python-dotenv==1.2.1

deactivate

print_section "Checkpoints"
download_checkpoint_by_path "${MODEL_CHECKPOINT_PATHS[$MODEL]}"

print_success "${MODEL} setup complete"
```

**Register in `setup/lib/share.sh`:**

```bash
# Add to OPENSOURCE_MODELS
declare -a OPENSOURCE_MODELS=(
    # ... existing models
    "opensourcemodel-512"
)

# Add checkpoint
declare -a CHECKPOINTS=(
    # ... existing checkpoints
    "opensourcemodel/model.safetensors|https://huggingface.co/org/repo/resolve/main/model.safetensors|5.2GB"
)

# Add mapping
declare -A MODEL_CHECKPOINT_PATHS=(
    # ... existing mappings
    ["opensourcemodel-512"]="opensourcemodel/model.safetensors"
)
```

## 🚀 Advanced Features

### Adding Model Constraints

```python
def _get_model_constraints(self, model: str) -> Dict[str, Any]:
    """Define model-specific constraints."""
    
    constraints = {
        "model-v1": {
            "max_duration": 5.0,
            "resolutions": ["480p", "720p"],
            "aspect_ratios": ["16:9", "1:1"],
            "fps": 24
        },
        "model-v2": {
            "max_duration": 10.0,
            "resolutions": ["720p", "1080p"],
            "aspect_ratios": ["16:9", "9:16", "1:1"],
            "fps": 30
        }
    }
    
    return constraints.get(model, constraints["model-v1"])
```

---

## 📝 Quick Checklist

### For Commercial API Models ✅

- [ ] API key obtained and tested
- [ ] Added model to `MODEL_CATALOG.py` with correct module/class names
- [ ] Created `{provider}_inference.py` with Service and Wrapper classes
- [ ] Service class handles API authentication and requests
- [ ] Wrapper inherits from `ModelWrapper` and implements `generate()`
- [ ] All 8 required fields returned from `generate()`
- [ ] Error handling returns error dict (not raises exception)
- [ ] Added exports to `vmevalkit/models/__init__.py`
- [ ] Created `.env` entry for API key
- [ ] Tested with `examples/generate_videos.py`
- [ ] Validated output format and video quality

### For Open-Source Models ✅

- [ ] Created `setup/models/{model-name}/setup.sh` script
- [ ] Script uses functions from `setup/lib/share.sh`
- [ ] All pip packages use exact versions (==X.Y.Z)
- [ ] Added checkpoint entries to `setup/lib/share.sh` CHECKPOINTS array
- [ ] Added model to OPENSOURCE_MODELS list
- [ ] Added checkpoint mapping to MODEL_CHECKPOINT_PATHS
- [ ] Created inference file `{model}_inference.py` with Service and Wrapper
- [ ] Service class implements subprocess or direct inference
- [ ] Wrapper inherits from `ModelWrapper` and implements `generate()`
- [ ] All 8 required fields returned from `generate()`
- [ ] Virtual environment path resolution implemented
- [ ] Submodule added (if needed) and documented
- [ ] Added model to `MODEL_CATALOG.py`
- [ ] Added exports to `vmevalkit/models/__init__.py`
- [ ] Ran setup script successfully
- [ ] Tested inference with `examples/generate_videos.py`
- [ ] Validated output format and video quality

---

## 🎓 Learning Path

**New to VMEvalKit?** Follow this progression:

1. **Start Simple**: Study a commercial API model (e.g., `luma_inference.py`)
2. **Understand Catalog**: Read `MODEL_CATALOG.py` to see all models
3. **Read Base Class**: Study `ModelWrapper` in `models/base.py`
4. **Study Setup**: Look at `setup/lib/share.sh` for open-source utilities
5. **Try Integration**: Add a simple commercial model first
6. **Advanced**: Tackle open-source model with full setup script

**Need Help?**
- Check existing implementations for similar patterns
- Review error messages carefully (they include hints)
- Test incrementally (catalog → wrapper → setup → inference)
- Use validation scripts to verify each step

---

Ready to add your model? Follow this guide and your model will be seamlessly integrated into VMEvalKit's architecture! 🎯

Whether you're integrating a cutting-edge commercial API or deploying a powerful open-source model, VMEvalKit's modular architecture makes it straightforward. Happy coding! 🚀