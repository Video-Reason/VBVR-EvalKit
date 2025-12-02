# Adding Models to VMEvalKit

VMEvalKit uses a **clean modular architecture** with dynamic loading, designed for scalability and easy model integration. This guide provides comprehensive instructions for integrating new video generation models.


### Adding Custom Models

1. Add entry to MODEL_CATALOG.py
2. Create wrapper class inheriting from ModelWrapper
3. Implement generate() method
4. No changes needed in core files!


## 🏗️ Architecture Overview

### Core System Design

```
vmevalkit/
├── runner/
│   ├── MODEL_CATALOG.py    # 📋 Pure model registry (40+ models, 11 families)
│   └── inference.py        # 🎭 Orchestration with dynamic loading  
├── models/
│   ├── base.py            # 🔧 Abstract ModelWrapper interface
│   ├── luma_inference.py  # LumaInference + LumaWrapper
│   ├── veo_inference.py   # VeoService + VeoWrapper
│   └── ...                # Each provider: Service + Wrapper pattern
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
✅ **Consistency**: Tempature always be set to zero to keep the results stable.

## 🔌 Registration Process

### Step 1: Add to MODEL_CATALOG.py

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

# Add to combined registries (at the bottom of file)
AVAILABLE_MODELS = {
    **LUMA_MODELS,
    **VEO_MODELS,
    **{PROVIDER}_MODELS,  # Add your models here
    # ... other models
}

MODEL_FAMILIES = {
    "Luma Dream Machine": LUMA_MODELS,
    "Google Veo": VEO_MODELS,
    "{Provider}": {PROVIDER}_MODELS,  # Add your family here
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

### Basic Functionality Test

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

# 4. Test via runner (production usage)
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

| Issue | Cause | Solution |
|-------|-------|----------|
| `Model not found` | Not registered in catalog | Add to MODEL_CATALOG.py |
| `ImportError: No module named` | Wrong module path | Check `wrapper_module` string in catalog |
| `AttributeError: module has no attribute` | Wrong class name | Verify `wrapper_class` matches actual class |
| `API key not found` | Missing environment variable | Set `{PROVIDER}_API_KEY` in .env or environment |
| `Submodule not found` | Git submodule not initialized | Run `git submodule update --init --recursive` |
| `generate() missing required field` | Incomplete return dictionary | Ensure all 8 required fields are returned |
| `Image not found` | Invalid image path | Use absolute paths or verify relative path |
| `Timeout during generation` | Long processing time | Increase timeout in subprocess/httpx calls |
| `Invalid model ID` | Wrong model identifier | Check actual model ID used by API |
| `Memory error` | Large model loading | Use subprocess pattern for isolation |

## 🎯 Best Practices

### DO ✅
- **Inherit from ModelWrapper**: Always use the base class
- **Return all required fields**: Every field in the standard format is mandatory
- **Use environment variables**: For API keys and secrets
- **Handle errors gracefully**: Return error dictionary, don't raise
- **Validate inputs early**: Check image exists before API calls
- **Use logging**: Import and use logger for debugging
- **Support async properly**: Use `asyncio.run()` for async services
- **Clean up resources**: Close clients, delete temp files
- **Document model variants**: Clear descriptions in catalog
- **Test thoroughly**: Use the integration test script

### DON'T ❌
- **Skip the base class**: All wrappers must inherit from ModelWrapper
- **Import models in catalog**: Use string paths only
- **Mix concerns**: Keep Service and Wrapper separate
- **Hardcode credentials**: Always use environment variables
- **Ignore timeouts**: Set reasonable timeouts for all operations
- **Forget metadata**: Include helpful debugging info
- **Raise exceptions in generate()**: Return error dictionary instead
- **Use relative imports**: Use absolute imports for clarity
- **Forget to save videos**: Always write to self.output_dir
- **Skip input validation**: Always check image exists

## 📚 Real Examples to Study

Study these actual implementations for patterns:

1. **API with Polling**: `vmevalkit/models/luma_inference.py`
   - S3 image upload for URL generation
   - Async polling for completion
   - Retry logic with tenacity

2. **Google Cloud Auth**: `vmevalkit/models/veo_inference.py`
   - Application Default Credentials
   - Fallback to gcloud CLI
   - Project ID handling

3. **Subprocess Pattern**: `vmevalkit/models/ltx_inference.py`
   - Clean subprocess execution
   - Config file mapping
   - Timeout handling

4. **Multiple Model Variants**: `vmevalkit/models/wavespeed_inference.py`
   - Enum for model selection
   - Different wrapper classes per variant
   - Shared service class

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

Ready to add your model? Follow this guide and your model will be seamlessly integrated into VMEvalKit's architecture! 🎯