"""VMEvalKit Inference Runner - Multi-Provider Video Generation"""

import shutil
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type
from datetime import datetime

from .MODEL_CATALOG import AVAILABLE_MODELS, MODEL_FAMILIES
from ..models.base import ModelWrapper


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
    wrapper_class = _load_model_wrapper(model_name)
    model_config = AVAILABLE_MODELS[model_name]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_id = kwargs.pop('inference_id', f"{model_name}_{timestamp}")
    inference_dir = Path(output_dir) / inference_id
    video_dir = inference_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    init_kwargs = {
        "model": model_config["model"],
        "output_dir": str(video_dir),
    }
    
    if "args" in model_config:
        init_kwargs.update(model_config["args"])
    
    wrapper = wrapper_class(**init_kwargs)
    
    if question_data:
        kwargs['question_data'] = question_data
    
    result = wrapper.generate(image_path, text_prompt, **kwargs)
    
    result["inference_dir"] = str(inference_dir)
    result["question_data"] = question_data
    
    return result


class InferenceRunner:
    """Enhanced inference runner with dynamic model loading."""
    
    def __init__(self, output_dir: str = "./outputs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.runs = []
        self._wrapper_cache = {}
    
    def run(
        self,
        model_name: str,
        image_path: Union[str, Path],
        text_prompt: str,
        question_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Run inference and save video as {task_id}.mp4 under domain folder."""
        start_time = datetime.now()
        
        domain_dir_name = "unknown_task"
        task_id = "unknown"
        if question_data:
            domain_dir_name = question_data.get("domain_dir") or question_data.get("domain") or "unknown_task"
            task_id = question_data.get("id", task_id)

        # Save video directly under domain folder as {task_id}.mp4
        domain_dir = self.output_dir / domain_dir_name
        domain_dir.mkdir(parents=True, exist_ok=True)
        
        if model_name not in self._wrapper_cache:
            wrapper_class = _load_model_wrapper(model_name)
            model_config = AVAILABLE_MODELS[model_name]
            
            init_kwargs = {
                "model": model_config["model"],
                "output_dir": str(self.output_dir),
            }
            
            if "args" in model_config:
                init_kwargs.update(model_config["args"])
            
            self._wrapper_cache[model_name] = wrapper_class(**init_kwargs)
            print(f"Loaded model: {model_name}")
        
        wrapper = self._wrapper_cache[model_name]
        
        # Set output dir to domain folder
        wrapper.output_dir = domain_dir
        
        if question_data:
            kwargs['question_data'] = question_data
        
        result = wrapper.generate(image_path, text_prompt, **kwargs)
        
        # Rename video to {task_id}.mp4
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
    
    def _cleanup_failed_folder(self, task_dir: Path):
        """Clean up folder if video generation failed."""
        if task_dir.exists():
            video_files = list(task_dir.glob("*.mp4")) + list(task_dir.glob("*.webm"))
            if video_files:
                return
            shutil.rmtree(task_dir)
            print(f"Cleaned up empty folder: {task_dir.name}")
    
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


def get_models_by_family(family_name: str) -> Dict[str, Dict[str, Any]]:
    """Get all models from a specific family."""
    if family_name not in MODEL_FAMILIES:
        raise ValueError(f"Unknown family: {family_name}. Available: {list(MODEL_FAMILIES.keys())}")
    return MODEL_FAMILIES[family_name]


def get_model_family(model_name: str) -> str:
    """Get the family name for a specific model."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return AVAILABLE_MODELS[model_name]["family"]


def list_all_families() -> Dict[str, int]:
    """List all model families and their counts."""
    return {
        family_name: len(family_models)
        for family_name, family_models in MODEL_FAMILIES.items()
    }


def add_model_family(family_name: str, models: Dict[str, Dict[str, Any]]) -> None:
    """Add a new model family to the registry."""
    for model_config in models.values():
        model_config["family"] = family_name
    
    MODEL_FAMILIES[family_name] = models
    AVAILABLE_MODELS.update(models)
