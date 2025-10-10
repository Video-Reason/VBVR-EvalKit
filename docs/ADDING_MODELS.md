# Adding New Models to VMEvalKit

**One comprehensive guide** for integrating new video generation models into VMEvalKit's unified inference system.

## 🚀 Quick Start

### Requirements
- **MUST support Image + Text → Video** (critical for reasoning evaluation)
- Follow unified interface: `generate(image_path, text_prompt, duration=..., output_filename=None, **kwargs)`

### Parameter Separation (MANDATORY)
- Put long-lived configuration in `__init__` (e.g., `model`, `api_key`, `output_dir`, defaults).
- Put per-run inputs in `generate(...)` (e.g., `image_path`, `text_prompt`, `duration`, `output_filename`, seeds).
- Do NOT pass per-run args to constructors.

### For API Models (3 steps):
1. **Create `vmevalkit/models/{provider}_inference.py`** with Service + Wrapper classes
2. **Register in `vmevalkit/runner/inference.py`**: Add to `AVAILABLE_MODELS` and `MODEL_FAMILIES` 
3. **Update `vmevalkit/models/__init__.py`** imports

### For Open-Source Models (4 steps):
1. **Add submodule**: `git submodule add https://github.com/owner/repo.git submodules/ModelName`
2. **Create wrapper** (same as API but with subprocess/direct calls)
3. **Register** (same as API)
4. **Update imports** (same as API)

---

## 📋 Detailed Guide

### 🎯 Architecture Overview

VMEvalKit uses a **unified model registry** supporting:

- **API Models**: Commercial services (Luma, Veo, etc.) via remote APIs
- **Open-Source Models**: Local inference via git submodules

Both use identical interface:
```python
def generate(self, image_path, text_prompt, duration=8.0, output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]
```

### 🌐 API-Based Models

Create `vmevalkit/models/{provider}_inference.py`:

```python
"""
{Provider} Inference Service for VMEvalKit
"""
import asyncio
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
import time

class {Provider}Service:
    def __init__(self, model_id: str, api_key: Optional[str] = None, **kwargs):
        self.model_id = model_id
        self.api_key = api_key or os.getenv("{PROVIDER}_API_KEY")
        
    async def generate_video(self, prompt: str, image_path: str, **kwargs):
        """
        YOUR API CALL HERE:
        1. Prepare request with image + text
        2. Make API call  
        3. Handle response
        4. Return (video_bytes, metadata)
        """
        # TODO: Implement API call
        video_bytes = b""  # Your API response
        metadata = {"request_id": "id", "model": self.model_id}
        return video_bytes, metadata

class {Provider}Wrapper:
    def __init__(self, model: str, output_dir: str = "./data/outputs", api_key: Optional[str] = None, **kwargs):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.service = {Provider}Service(model_id=model, api_key=api_key, **kwargs)
    
    def generate(self, image_path: Union[str, Path], text_prompt: str, duration: float = 8.0, output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """VMEvalKit unified interface - EXACT signature required"""
        start_time = time.time()
        
        # Validate input
        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}", "video_path": None}
        
        # Run API call
        video_bytes, metadata = asyncio.run(
            self.service.generate_video(prompt=text_prompt, image_path=str(image_path), **kwargs)
        )
        
        # Save video
        if video_bytes:
            if not output_filename:
                output_filename = f"{self.model.replace('/', '_')}_{int(time.time())}.mp4"
            video_path = self.output_dir / output_filename
            with open(video_path, 'wb') as f:
                f.write(video_bytes)
            
            return {"success": True, "video_path": str(video_path), "duration": time.time() - start_time, **metadata}
        
        return {"success": False, "error": "No video generated", "video_path": None}
```

### 🔓 Open-Source Models (Submodules)

**Step 1: Add submodule**
```bash
git submodule add https://github.com/owner/repo.git submodules/ModelName
```

**Step 2: Create wrapper** `vmevalkit/models/{model}_inference.py`:

```python
"""
{Model} Inference Service for VMEvalKit  
"""
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union
import time

# Add submodule to path
{MODEL}_PATH = Path(__file__).parent.parent.parent / "submodules" / "{ModelName}"
sys.path.insert(0, str({MODEL}_PATH))

class {Model}Service:
    def __init__(self, model_id: str = "default", output_dir: str = "./data/outputs", **kwargs):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Verify submodule exists
        if not {MODEL}_PATH.exists():
            raise FileNotFoundError(f"{Model} submodule not found. Run: git submodule update --init {ModelName}")

    def _run_subprocess_inference(self, image_path: str, text_prompt: str, output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Use this for models that need subprocess execution"""
        start_time = time.time()
        # Use provided filename or generate one
        if output_filename:
            output_path = self.output_dir / output_filename
        else:
            output_path = self.output_dir / f"{self.model_id}_{int(time.time())}.mp4"
        
        cmd = [
            sys.executable, str({MODEL}_PATH / "inference.py"),  # Adjust to actual script
            "--prompt", text_prompt, "--image", str(image_path), "--output", str(output_path)
        ]
        
        try:
            result = subprocess.run(cmd, cwd=str({MODEL}_PATH), capture_output=True, text=True, timeout=300)
            success = result.returncode == 0 and output_path.exists()
            return {
                "success": success, 
                "video_path": str(output_path) if success else None,
                "error": result.stderr if not success else None,
                "duration": time.time() - start_time
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "error": "Timeout", "video_path": None}
    
    def _run_direct_inference(self, image_path: str, text_prompt: str, output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Use this for models you can import directly"""
        try:
            # TODO: Import and run model directly
            # from {model_module} import {ModelClass}
            # model = {ModelClass}()
            # result = model.generate(image_path, text_prompt, output_filename=output_filename)
            return {"success": False, "error": "Direct inference not implemented", "video_path": None}
        except Exception as e:
            return {"success": False, "error": str(e), "video_path": None}
    
    def generate(self, image_path: Union[str, Path], text_prompt: str, duration: float = 8.0, output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        if not Path(image_path).exists():
            return {"success": False, "error": f"Image not found: {image_path}", "video_path": None}
        
        # Choose your approach:
        return self._run_subprocess_inference(str(image_path), text_prompt, output_filename=output_filename, **kwargs)
        # OR: return self._run_direct_inference(str(image_path), text_prompt, output_filename=output_filename, **kwargs)

class {Model}Wrapper:
    def __init__(self, model: str, output_dir: str = "./data/outputs", **kwargs):
        self.model = model
        self.service = {Model}Service(model_id=model, output_dir=output_dir, **kwargs)
    
    def generate(self, image_path: Union[str, Path], text_prompt: str, duration: float = 8.0, output_filename: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """VMEvalKit unified interface - EXACT signature required"""
        return self.service.generate(image_path=image_path, text_prompt=text_prompt, duration=duration, output_filename=output_filename, **kwargs)
```

### 📝 Registration (Same for Both Types)

**Step 1: Import** in `vmevalkit/runner/inference.py`:
```python
from ..models import {Provider}Wrapper  # Add this import
```

**Step 2: Define models**:
```python
{PROVIDER}_MODELS = {
    "{provider}-model-name": {
        "class": {Provider}Wrapper,
        "model": "actual-model-id-from-provider",
        "description": "Brief model description", 
        "family": "{Provider}"
    },
    # Add more variants...
}
```

**Step 3: Register in dictionaries**:
```python
AVAILABLE_MODELS = {
    **EXISTING_MODELS,
    **{PROVIDER}_MODELS  # Add this line
}

MODEL_FAMILIES = {
    **EXISTING_FAMILIES,
    "{Provider}": {PROVIDER}_MODELS  # Add this line
}
```

**Step 4: Update imports** in `vmevalkit/models/__init__.py`:
```python
from .{provider}_inference import {Provider}Service, {Provider}Wrapper

__all__ = [
    # ... existing imports
    "{Provider}Service", "{Provider}Wrapper"
]
```

## 🧪 Testing

```python
# Test registration
from vmevalkit.runner.inference import AVAILABLE_MODELS, list_all_families
print("Model registered:", "your-model-name" in AVAILABLE_MODELS)
print("All families:", list_all_families())

# Test generation
from vmevalkit.runner.inference import run_inference
result = run_inference(
    model_name="your-model-name",
    image_path="test.png",
    text_prompt="test prompt",
    output_dir="./data/outputs/test",
    # Per-run options go here:
    output_filename="example.mp4",
    duration=8
)
print("Success:", result.get("success"))
print("Video:", result.get("video_path"))
```

## ✅ Best Practices

- **Error Handling**: Validate inputs, handle timeouts, return consistent format
- **File Management**: Use `Path` objects, create dirs automatically, unique filenames
- **Documentation**: Clear docstrings, document limitations, provide examples
- **Naming**: `{Provider}Service`/`{Provider}Wrapper`, `{provider}_inference.py`

### Do / Don't
- Do: put `model`, `api_key`, `output_dir` in `__init__`; keep it stable across runs.
- Do: pass `output_filename`, `duration`, `seed`, etc. to `generate(...)` per run.
- Don't: pass per-run args (e.g., `output_filename`) to `__init__`.

## 🔧 Troubleshooting

**Model not in registry**: Check both `AVAILABLE_MODELS` and `MODEL_FAMILIES` updated
**Import errors**: Verify `__init__.py` imports added
**Submodule issues**: Run `git submodule update --init --recursive`  
**API auth**: Use environment variables, provide clear error messages

## 📊 Current VMEvalKit Status

**36 models across 9 providers:**
- Commercial APIs: Luma (2), Veo (3), WaveSpeed (18), Runway (3), Sora (2)
- Open-Source: LTX-Video (3), HunyuanVideo (1), VideoCrafter (1), DynamiCrafter (3)

Following this guide ensures seamless integration with VMEvalKit's unified inference system! 🚀