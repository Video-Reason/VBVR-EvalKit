"""VMEvalKit Inference Runner - Multi-Provider Video Generation"""

import importlib
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

from .MODEL_CATALOG import AVAILABLE_MODELS, MODEL_FAMILIES
from ..models.base import ModelWrapper

# Path to the subprocess worker script
_WORKER_SCRIPT = Path(__file__).parent.parent / "models" / "_subprocess_worker.py"

# VMEvalKit root directory
_VMEVAL_ROOT = Path(__file__).parent.parent.parent
_UNKNOWN_DOMAIN = "unknown_task"
_UNKNOWN_TASK_ID = "unknown"


def _json_safe_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return kwargs converted to JSON-safe values."""
    serializable_kwargs: Dict[str, Any] = {}
    for key, value in kwargs.items():
        try:
            json.dumps(value, default=str)
            serializable_kwargs[key] = value
        except (TypeError, ValueError):
            serializable_kwargs[key] = str(value)
    return serializable_kwargs


def _build_failed_result(
    model_name: str,
    start_time: float,
    error: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Standardize failed subprocess result payloads."""
    return {
        "success": False,
        "video_path": None,
        "error": error,
        "duration_seconds": time.time() - start_time,
        "generation_id": None,
        "model": model_name,
        "status": "failed",
        "metadata": metadata or {},
    }


def _extract_domain_and_task(question_data: Optional[Dict[str, Any]]) -> tuple[str, str]:
    """Extract output domain folder and task id from optional question metadata."""
    if not question_data:
        return _UNKNOWN_DOMAIN, _UNKNOWN_TASK_ID

    domain = question_data.get("domain_dir") or question_data.get("domain") or _UNKNOWN_DOMAIN
    task_id = question_data.get("id", _UNKNOWN_TASK_ID)
    return domain, task_id


def _build_wrapper_init_kwargs(model_name: str, output_dir: Path) -> Dict[str, Any]:
    """Build constructor kwargs for a model wrapper from catalog config."""
    model_config = AVAILABLE_MODELS[model_name]
    init_kwargs: Dict[str, Any] = {
        "model": model_config["model"],
        "output_dir": str(output_dir),
    }
    if "args" in model_config:
        init_kwargs.update(model_config["args"])
    return init_kwargs


def _get_model_venv_python(model_name: str) -> Optional[str]:
    """Get the venv Python path for a model, or None if no venv exists."""
    # Check catalog for venv_id override, otherwise use model_name
    config = AVAILABLE_MODELS.get(model_name, {})
    venv_id = config.get("venv_id", model_name)
    venv_python = _VMEVAL_ROOT / "envs" / venv_id / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return None


def _run_via_subprocess(
    model_name: str,
    venv_python: str,
    image_path: Union[str, Path],
    text_prompt: str,
    output_dir: str,
    **kwargs
) -> Dict[str, Any]:
    """Run model inference in a subprocess using the model's venv Python."""
    start_time = time.time()

    serializable_kwargs = _json_safe_kwargs(kwargs)

    cmd = [
        venv_python,
        str(_WORKER_SCRIPT),
        "--model-name", model_name,
        "--image-path", str(image_path),
        "--prompt", text_prompt,
        "--output-dir", output_dir,
        "--kwargs-json", json.dumps(serializable_kwargs, default=str),
    ]

    print(f"[subprocess] Running {model_name} with venv: {venv_python}")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(_VMEVAL_ROOT),
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour timeout
        )

        # Parse result from stdout (look for the marker line)
        stdout = result.stdout
        stderr = result.stderr

        if stderr:
            # Print stderr for debugging (model loading progress, warnings, etc.)
            for line in stderr.strip().split("\n")[-20:]:
                print(f"  [stderr] {line}")

        # Find the result JSON in stdout
        parsed_result = None
        for line in stdout.split("\n"):
            if line.startswith("__VMEVAL_RESULT__"):
                json_str = line[len("__VMEVAL_RESULT__"):]
                parsed_result = json.loads(json_str)
                break

        if parsed_result is not None:
            return parsed_result

        # No result marker found - check return code
        if result.returncode != 0:
            return _build_failed_result(
                model_name=model_name,
                start_time=start_time,
                error=f"Subprocess failed (exit {result.returncode}): {stderr[-500:] if stderr else 'no stderr'}",
                metadata={"stdout": stdout[-500:] if stdout else ""},
            )

        # Return code 0 but no result marker - try to find video
        return _build_failed_result(
            model_name=model_name,
            start_time=start_time,
            error=f"No result marker in output. stdout: {stdout[-500:]}",
        )

    except subprocess.TimeoutExpired:
        return _build_failed_result(
            model_name=model_name,
            start_time=start_time,
            error="Subprocess timed out (2 hours)",
        )
    except Exception as e:
        return _build_failed_result(
            model_name=model_name,
            start_time=start_time,
            error=f"Subprocess error: {e}",
        )


def _load_model_wrapper(model_name: str) -> Type[ModelWrapper]:
    """Load wrapper class dynamically from catalog."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(AVAILABLE_MODELS.keys())}"
        )

    config = AVAILABLE_MODELS[model_name]
    module = importlib.import_module(config["wrapper_module"])
    wrapper_class = getattr(module, config["wrapper_class"])

    return wrapper_class


def run_inference(
    model_name: str,
    image_path: Union[str, Path],
    text_prompt: str,
    output_dir: str = "./outputs",
    question_data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Run inference with specified model using dynamic loading."""
    generation_kwargs = dict(kwargs)
    if question_data:
        generation_kwargs["question_data"] = question_data

    # Check for venv - use subprocess if available
    venv_python = _get_model_venv_python(model_name)
    if venv_python:
        result = _run_via_subprocess(
            model_name, venv_python, image_path, text_prompt, output_dir, **generation_kwargs
        )
        result["question_data"] = question_data
        return result

    # Fallback: direct import (for commercial API models without venvs)
    wrapper_class = _load_model_wrapper(model_name)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_id = generation_kwargs.pop('inference_id', f"{model_name}_{timestamp}")
    inference_dir = Path(output_dir) / inference_id
    video_dir = inference_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)

    init_kwargs = _build_wrapper_init_kwargs(model_name, video_dir)
    wrapper = wrapper_class(**init_kwargs)

    result = wrapper.generate(image_path, text_prompt, **generation_kwargs)

    result["inference_dir"] = str(inference_dir)

    result["question_data"] = question_data
    return result


class InferenceRunner:
    """Enhanced inference runner with dynamic model loading."""

    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self._wrapper_cache = {}

    def _get_or_create_wrapper(self, model_name: str) -> ModelWrapper:
        """Lazily initialize and cache wrappers for non-venv models."""
        if model_name not in self._wrapper_cache:
            wrapper_class = _load_model_wrapper(model_name)
            init_kwargs = _build_wrapper_init_kwargs(model_name, self.output_dir)
            self._wrapper_cache[model_name] = wrapper_class(**init_kwargs)
            print(f"Loaded model: {model_name}")
        return self._wrapper_cache[model_name]

    def run(
        self,
        model_name: str,
        image_path: Union[str, Path],
        text_prompt: str,
        question_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run inference and save video as {task_id}.mp4 under domain folder."""
        domain_dir_name, task_id = _extract_domain_and_task(question_data)

        domain_dir = self.output_dir / domain_dir_name
        domain_dir.mkdir(parents=True, exist_ok=True)
        generation_kwargs = dict(kwargs)
        if question_data:
            generation_kwargs['question_data'] = question_data

        # Use subprocess for venv models, direct import for API models
        venv_python = _get_model_venv_python(model_name)
        if venv_python:
            result = _run_via_subprocess(
                model_name, venv_python, image_path, text_prompt,
                str(domain_dir), **generation_kwargs
            )
        else:
            wrapper = self._get_or_create_wrapper(model_name)
            wrapper.output_dir = domain_dir
            result = wrapper.generate(image_path, text_prompt, **generation_kwargs)

        self._rename_video_to_task_id(domain_dir, task_id, result)
        print(f"\nInference complete: {domain_dir / f'{task_id}.mp4'}")
        return result

    def _rename_video_to_task_id(self, domain_dir: Path, task_id: str, result: Dict[str, Any]):
        """Rename generated video to {task_id}.mp4."""
        video_path = result.get("video_path")
        if not video_path:
            return

        video_path = Path(video_path)
        if not video_path.exists():
            return

        target_path = domain_dir / f"{task_id}.mp4"
        if video_path != target_path:
            video_path.rename(target_path)
            result["video_path"] = str(target_path)

    def list_models(self) -> Dict[str, str]:
        """List available models and their descriptions."""
        return {
            name: config["description"]
            for name, config in AVAILABLE_MODELS.items()
        }

    def list_models_by_family(self) -> Dict[str, Dict[str, str]]:
        """List models organized by family."""
        return {
            family_name: {
                name: config["description"]
                for name, config in family_models.items()
            }
            for family_name, family_models in MODEL_FAMILIES.items()
        }

    def get_model_families(self) -> Dict[str, int]:
        """Get model family statistics."""
        return {
            family_name: len(family_models)
            for family_name, family_models in MODEL_FAMILIES.items()
        }


# Re-export catalog utility functions for backward compatibility
from .MODEL_CATALOG import get_models_by_family, get_model_family, list_all_families, add_model_family
