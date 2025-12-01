# Adding Models to VMEvalKit

VMEvalKit uses a **clean modular architecture** with dynamic loading, designed for scalability and easy model integration. This guide provides comprehensive instructions for integrating new video generation models.


### Adding Custom Models

See [ADDING_MODELS.md](ADDING_MODELS.md) for the complete guide. Quick overview:

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
6. **No Circular Imports**: String paths in catalog eliminate import dependencies

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

## 🎯 Implementation Guide

### API Models - Complete Template

**File**: `vmevalkit/models/{provider}_inference.py`

```python
"""
{Provider} Video Generation Service for VMEvalKit.

Supports text + image → video generation using {Provider}'s API.
"""

import os
import time
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from .base import ModelWrapper

logger = logging.getLogger(__name__)


class {Provider}Service:
    """Service class for {Provider} API interactions."""
    
    def __init__(
        self,
        model_id: str,
        api_key: Optional[str] = None,
        base_url: str = "https://api.{provider}.com/v1",
        **kwargs
    ):
        """
        Initialize {Provider} service.
        
        Args:
            model_id: Model identifier
            api_key: API key (falls back to environment variable)
            base_url: API endpoint base URL
            **kwargs: Additional parameters
        """
        # Handle API key with clear error message
        self.api_key = api_key or os.getenv("{PROVIDER}_API_KEY")
        if not self.api_key:
            raise ValueError(
                "{Provider} API not configured: {PROVIDER}_API_KEY environment variable required.\n"
                "Set {PROVIDER}_API_KEY in your environment or .env file."
            )
        
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        
        # Setup headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize any required clients (e.g., httpx)
        import httpx
        self.client = httpx.AsyncClient(
            headers=self.headers,
            timeout=httpx.Timeout(300.0)  # 5 minute timeout
        )
    
    async def generate_video(
        self,
        prompt: str,
        image_path: Union[str, Path],
        duration: float = 8.0,
        **kwargs
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """
        Generate video using {Provider} API.
        
        Args:
            prompt: Text instructions
            image_path: Path to input image
            duration: Video duration in seconds
            **kwargs: Additional API parameters
            
        Returns:
            Tuple of (video_bytes, metadata)
        """
        try:
            # 1. Prepare image (may need to upload or encode)
            image_url = await self._prepare_image(image_path)
            
            # 2. Create generation request
            request_data = {
                "model": self.model_id,
                "prompt": prompt,
                "image": image_url,
                "duration": duration,
                **kwargs  # Pass through additional parameters
            }
            
            # 3. Make API call
            response = await self.client.post(
                f"{self.base_url}/generations",
                json=request_data
            )
            response.raise_for_status()
            
            # 4. Handle response (may need polling)
            result = response.json()
            generation_id = result.get("id")
            
            # 5. Poll for completion if needed
            video_url = await self._poll_for_completion(generation_id)
            
            # 6. Download video
            video_response = await self.client.get(video_url)
            video_bytes = video_response.content
            
            metadata = {
                "generation_id": generation_id,
                "model": self.model_id,
                "video_url": video_url
            }
            
        return video_bytes, metadata
            
        except Exception as e:
            logger.error(f"{Provider} generation failed: {e}")
            return None, {"error": str(e)}
        
        finally:
            # Clean up if needed
            pass
    
    async def _prepare_image(self, image_path: Union[str, Path]) -> str:
        """
        Prepare image for API (upload or encode as needed).
        
        Returns:
            Image URL or base64 string
        """
        # Implementation depends on API requirements
        # Option 1: Upload to provider's storage
        # Option 2: Upload to S3 (see luma_inference.py for S3ImageUploader)
        # Option 3: Encode as base64
        
        import base64
        from PIL import Image
        import io
        
        # Example: base64 encoding
        img = Image.open(image_path)
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"
    
    async def _poll_for_completion(
        self,
        generation_id: str,
        max_wait: int = 300
    ) -> Optional[str]:
        """
        Poll API for generation completion.
        
        Returns:
            Video URL when ready
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            response = await self.client.get(
                f"{self.base_url}/generations/{generation_id}"
            )
            
            if response.status_code == 200:
                data = response.json()
                status = data.get("status")
                
                if status == "completed":
                    return data.get("video_url")
                elif status == "failed":
                    raise Exception(f"Generation failed: {data.get('error')}")
            
            await asyncio.sleep(5)  # Check every 5 seconds
        
        raise TimeoutError(f"Generation timed out after {max_wait} seconds")
    
    async def __aenter__(self):
        """Async context manager support."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async client."""
        await self.client.aclose()


class {Provider}Wrapper(ModelWrapper):
    """VMEvalKit wrapper for {Provider} models."""
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize wrapper.
        
        Args:
            model: Model identifier (from catalog)
            output_dir: Directory to save videos
            api_key: Optional API key (falls back to env var)
            **kwargs: Additional parameters
        """
        super().__init__(model, output_dir, **kwargs)
        
        # Initialize service
        self.service = {Provider}Service(
            model_id=model,
            api_key=api_key,
            **kwargs
        )
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video from image and text prompt.
        
        Implements ModelWrapper.generate() interface.
        """
        start_time = time.time()
        
        # Validate inputs
        image_path = Path(image_path)
        if not image_path.exists():
            return {
                "success": False, 
                "video_path": None,
                "error": f"Image not found: {image_path}",
                "duration_seconds": 0,
                "generation_id": "error",
                "model": self.model,
                "status": "failed",
                "metadata": {
                    "prompt": text_prompt,
                    "image_path": str(image_path)
                }
            }
        
        try:
            # Run async service method
        video_bytes, metadata = asyncio.run(
                self.service.generate_video(
                    prompt=text_prompt,
                    image_path=str(image_path),
                    duration=duration,
                    **kwargs
                )
        )
        
        if video_bytes:
                # Save video to file
            if not output_filename:
                    timestamp = int(time.time())
                    output_filename = f"{self.model.replace('/', '_')}_{timestamp}.mp4"
                
            video_path = self.output_dir / output_filename
            video_path.write_bytes(video_bytes)
            
            return {
                "success": True, 
                "video_path": str(video_path),
                "error": None,
                "duration_seconds": time.time() - start_time,
                    "generation_id": metadata.get("generation_id", "unknown"),
                "model": self.model,
                "status": "success",
                    "metadata": {
                        "prompt": text_prompt,
                        "image_path": str(image_path),
                        **metadata
                    }
                }
            else:
                error_msg = metadata.get("error", "No video generated")
        return {
            "success": False, 
            "video_path": None,
                    "error": error_msg,
            "duration_seconds": time.time() - start_time,
            "generation_id": "failed",
            "model": self.model,
            "status": "failed",
                    "metadata": {
                        "prompt": text_prompt,
                        "image_path": str(image_path),
                        **metadata
                    }
                }
                
        except Exception as e:
            return {
                "success": False,
                "video_path": None,
                "error": str(e),
                "duration_seconds": time.time() - start_time,
                "generation_id": "error",
                "model": self.model,
                "status": "failed",
                "metadata": {
                    "prompt": text_prompt,
                    "image_path": str(image_path)
                }
            }
```

### Open-Source Models - Subprocess Template

**File**: `vmevalkit/models/{model}_inference.py`

```python
"""
{ModelName} Inference Service for VMEvalKit.

Wrapper for the {ModelName} model (submodules/{ModelName}) to integrate with
VMEvalKit's unified inference interface.
"""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .base import ModelWrapper
import json
import logging

logger = logging.getLogger(__name__)

# Add submodule to path
{MODEL}_PATH = Path(__file__).parent.parent.parent / "submodules" / "{ModelName}"
sys.path.insert(0, str({MODEL}_PATH))


class {Model}Service:
    """Service class for {ModelName} inference integration."""
    
    def __init__(
        self,
        model_id: str = "default",
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """
        Initialize {ModelName} service.
        
        Args:
            model_id: Model variant to use
            output_dir: Directory to save generated videos
            **kwargs: Additional parameters
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Check if submodule exists
        if not {MODEL}_PATH.exists():
            raise FileNotFoundError(
                f"{ModelName} not available. Please initialize submodule:\n"
                f"git submodule update --init --recursive submodules/{ModelName}"
            )
        
        # Map model IDs to config files if needed
        self.config_mapping = {
            "default": "configs/default.yaml",
            # Add more model variants
        }
        
        self.config_path = {MODEL}_PATH / self.config_mapping.get(
            model_id, "configs/default.yaml"
        )
    
    def _run_subprocess_inference(
        self,
        image_path: str,
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference using subprocess to avoid dependency conflicts.
        
        Returns:
            Dictionary with inference results and metadata
        """
        start_time = time.time()
        
        # Generate output filename
        if not output_filename:
            timestamp = int(time.time())
            output_filename = f"{self.model_id}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Prepare command
        cmd = [
            sys.executable,
            str({MODEL}_PATH / "inference.py"),
            "--prompt", text_prompt,
            "--image", str(image_path),
            "--output", str(output_path),
            "--duration", str(duration)
        ]
        
        # Add any model-specific parameters
        if self.config_path.exists():
            cmd.extend(["--config", str(self.config_path)])
        
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        try:
            # Run inference
            result = subprocess.run(
                cmd,
                cwd=str({MODEL}_PATH),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            success = result.returncode == 0 and output_path.exists()
            error_msg = result.stderr if not success else None
            
            # Parse any JSON output if available
            try:
                if result.stdout and result.stdout.strip().startswith('{'):
                    output_data = json.loads(result.stdout)
                else:
                    output_data = {}
            except:
                output_data = {}
            
            return {
                "success": success, 
                "video_path": str(output_path) if success else None,
                "error": error_msg,
                "duration_seconds": time.time() - start_time,
                "generation_id": f"{self.model_id}_{int(time.time())}",
                "model": self.model_id,
                "status": "success" if success else "failed",
                "metadata": {
                    "text_prompt": text_prompt,
                    "image_path": image_path,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    **output_data
                }
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False, 
                "video_path": None,
                "error": "Inference timed out after 5 minutes",
                "duration_seconds": time.time() - start_time,
                "generation_id": "timeout",
                "model": self.model_id,
                "status": "failed",
                "metadata": {
                    "text_prompt": text_prompt,
                    "image_path": image_path
                }
            }
        except Exception as e:
            return {
                "success": False,
                "video_path": None,
                "error": f"Subprocess failed: {str(e)}",
                "duration_seconds": time.time() - start_time,
                "generation_id": "error",
                "model": self.model_id,
                "status": "failed",
                "metadata": {
                    "text_prompt": text_prompt,
                    "image_path": image_path
                }
            }
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video from image and text prompt."""
        
        # Validate input
        if not Path(image_path).exists():
            return {
                "success": False, 
                "video_path": None,
                "error": f"Image not found: {image_path}",
                "duration_seconds": 0,
                "generation_id": "error",
                "model": self.model_id,
                "status": "failed",
                "metadata": {
                    "text_prompt": text_prompt,
                    "image_path": str(image_path)
                }
            }
        
        return self._run_subprocess_inference(
            str(image_path),
            text_prompt,
            duration,
            output_filename,
            **kwargs
        )


class {Model}Wrapper(ModelWrapper):
    """VMEvalKit wrapper for {ModelName}."""
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """Initialize wrapper."""
        super().__init__(model, output_dir, **kwargs)
        
        self.service = {Model}Service(
            model_id=model,
            output_dir=output_dir,
            **kwargs
        )
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video from image and text prompt."""
        return self.service.generate(
            image_path,
            text_prompt,
            duration,
            output_filename,
            **kwargs
        )
```

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

print(f"Success: {result['success']}")
print(f"Video: {result.get('video_path')}")
print(f"Error: {result.get('error')}")
print(f"Duration: {result.get('duration_seconds'):.2f}s")

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

### Supporting Multiple Authentication Methods

```python
def get_api_key(provider: str) -> str:
    """Get API key with multiple fallbacks."""
    
    # 1. Direct environment variable
    key = os.getenv(f"{provider.upper()}_API_KEY")
    if key:
        return key
    
    # 2. Check .env file
    from dotenv import load_dotenv
    load_dotenv()
    key = os.getenv(f"{provider.upper()}_API_KEY")
    if key:
        return key
    
    # 3. Check config file
    config_path = Path.home() / f".{provider}" / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
            key = config.get("api_key")
            if key:
                return key
    
    raise ValueError(f"No API key found for {provider}")
```

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

### Image Preprocessing

```python
def preprocess_image(
    self,
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (512, 512)
) -> str:
    """Preprocess image for model requirements."""
    
    from PIL import Image
    import tempfile
    
    img = Image.open(image_path)
    
    # Resize if needed
    if img.size != target_size:
        img = img.resize(target_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        img.save(tmp.name, 'PNG')
        return tmp.name
```

## 🎓 Architecture Benefits Summary

| **Aspect** | **Benefit** |
|------------|-------------|
| **Dynamic Loading** | Models loaded only when needed, reducing memory and startup time |
| **String-based Registry** | No circular imports, clean dependency graph |
| **Service + Wrapper Pattern** | Clear separation of API logic and VMEvalKit interface |
| **Abstract Base Class** | Enforces consistent interface across all models |
| **Family Organization** | Logical grouping for bulk operations and discovery |
| **Standardized Returns** | Predictable data structure for all models |
| **Error Handling** | Graceful failures with informative messages |
| **Async Support** | Efficient handling of long-running operations |

---

Ready to add your model? Follow this guide and your model will be seamlessly integrated into VMEvalKit's architecture! 🎯