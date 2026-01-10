"""
VMEvalKit Inference Runner - Multi-Provider Video Generation

Unified interface for 37+ text+image→video models across 9 major providers.
Uses dynamic loading from MODEL_CATALOG for clean separation of concerns.
"""

import shutil
import importlib
from pathlib import Path
from typing import Dict, Any, Optional, Union, Type
from datetime import datetime
import json

from .MODEL_CATALOG import AVAILABLE_MODELS, MODEL_FAMILIES
from ..models.base import ModelWrapper


def _load_model_wrapper(model_name: str) -> Type[ModelWrapper]:
    """
    Load wrapper class dynamically from catalog.
    
    Args:
        model_name: Name of model to load
        
    Returns:
        ModelWrapper class
        
    Raises:
        ValueError: If model not found
        ImportError: If module/class cannot be imported
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(AVAILABLE_MODELS.keys())}"
        )
    
    config = AVAILABLE_MODELS[model_name]
    
    # Dynamic import
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
    """
    Run inference with specified model using dynamic loading.
    
    Args:
        model_name: Name of model to use (from MODEL_CATALOG)
        image_path: Path to input image
        text_prompt: Text instructions for video generation
        output_dir: Directory to save outputs
        question_data: Optional question metadata including final_image_path
        **kwargs: Additional model-specific parameters
        
    Returns:
        Dictionary with inference results
    """
    # Load wrapper dynamically
    wrapper_class = _load_model_wrapper(model_name)
    model_config = AVAILABLE_MODELS[model_name]
    
    # Create structured output directory for this inference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_id = kwargs.pop('inference_id', f"{model_name}_{timestamp}")
    inference_dir = Path(output_dir) / inference_id
    video_dir = inference_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model instance with clean initialization
    init_kwargs = {
        "model": model_config["model"],
        "output_dir": str(video_dir),
    }
    
    # Add any model-specific args from catalog
    if "args" in model_config:
        init_kwargs.update(model_config["args"])
    
    wrapper = wrapper_class(**init_kwargs)
    
    # Add question_data to kwargs so models can access it (e.g., for final_image_path)
    if question_data:
        kwargs['question_data'] = question_data
    
    # Run inference
    result = wrapper.generate(image_path, text_prompt, **kwargs)
    
    # Add structured output directory to result
    result["inference_dir"] = str(inference_dir)
    result["question_data"] = question_data
    
    return result


class InferenceRunner:
    """
    Enhanced inference runner with dynamic model loading.
    
    Each inference creates a self-contained folder with:
    - video/: Generated video as video.mp4
    - question/: Input images, prompt, and metadata
    """
    
    def __init__(self, output_dir: str = "./outputs"):
        """
        Initialize runner.
        
        Args:
            output_dir: Directory to save generated videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Keep in-memory run log
        self.runs = []
        
        # Cache wrapper instances to avoid reloading models
        self._wrapper_cache = {}
    
    def run(
        self,
        model_name: str,
        image_path: Union[str, Path],
        text_prompt: str,
        question_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference and create structured output folder.
        
        Args:
            model_name: Model to use (from MODEL_CATALOG)
            image_path: Input image (first frame)
            text_prompt: Text instructions
            question_data: Optional question metadata
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results
        """
        start_time = datetime.now()
        
        # Create structured output directory mirroring questions structure
        domain = None
        task_id = "unknown"
        if question_data:
            domain = question_data.get("domain") or question_data.get("task_category")
            task_id = question_data.get("id", task_id)
        domain_dir_name = f"{domain}_task" if domain else "unknown_task"

        # Use flat structure: no run timestamp folder
        inference_dir = self.output_dir / domain_dir_name / task_id
        inference_dir.mkdir(parents=True, exist_ok=True)
        
        # Get or create cached wrapper for this model
        if model_name not in self._wrapper_cache:
            wrapper_class = _load_model_wrapper(model_name)
            model_config = AVAILABLE_MODELS[model_name]
            
            init_kwargs = {
                "model": model_config["model"],
                "output_dir": str(self.output_dir),  # Will be updated to specific inference_dir/video later
            }
            
            if "args" in model_config:
                init_kwargs.update(model_config["args"])
            
            self._wrapper_cache[model_name] = wrapper_class(**init_kwargs)
            print(f"📦 Loaded model: {model_name} (will be reused for all tasks)")
        
        wrapper = self._wrapper_cache[model_name]
        
        # Update output dir for this specific task
        video_dir = inference_dir / "video"
        video_dir.mkdir(parents=True, exist_ok=True)
        wrapper.output_dir = video_dir
        
        # Run inference using cached wrapper
        if question_data:
            kwargs['question_data'] = question_data
        
        result = wrapper.generate(image_path, text_prompt, **kwargs)
        
        # Create question folder and copy images
        self._setup_question_folder(inference_dir, image_path, text_prompt, question_data)
        
        # Rename video to video.mp4
        self._rename_video_to_standard(video_dir, result)
        
        print(f"\n✅ Inference complete! Output saved to: {inference_dir}")
        print(f"   - Generated: {inference_dir}/video/video.mp4")
        print(f"   - Question: {inference_dir}/question/")
        
        return result
    
    def _setup_question_folder(self, inference_dir: Path, first_image: Union[str, Path], 
                               prompt: str, question_data: Optional[Dict[str, Any]]):
        """Create question folder with input images, prompt, and ground truth video."""
        question_dir = inference_dir / "question"
        question_dir.mkdir(exist_ok=True)
        
        # Copy first image
        first_image_path = Path(first_image)
        if first_image_path.exists():
            dest_first = question_dir / f"first_frame{first_image_path.suffix}"
            shutil.copy2(first_image_path, dest_first)
        
        # Copy final image if available
        if question_data and 'final_image_path' in question_data and question_data['final_image_path'] is not None:
            final_image_path = Path(question_data['final_image_path'])
            if final_image_path.exists():
                dest_final = question_dir / f"final_frame{final_image_path.suffix}"
                shutil.copy2(final_image_path, dest_final)
        
        # Copy ground truth video if available
        if question_data and 'ground_truth_video' in question_data and question_data['ground_truth_video'] is not None:
            gt_video_path = Path(question_data['ground_truth_video'])
            if gt_video_path.exists():
                dest_gt = question_dir / "ground_truth.mp4"
                shutil.copy2(gt_video_path, dest_gt)
        
        # Save prompt
        prompt_file = question_dir / "prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(prompt)
    
    def _rename_video_to_standard(self, video_dir: Path, result: Dict[str, Any]):
        """Rename generated video to video.mp4."""
        video_path = result.get("video_path")
        if not video_path:
            return
        
        video_path = Path(video_path)
        if not video_path.exists():
            return
        
        # Rename to video.mp4
        target_path = video_dir / "video.mp4"
        if video_path != target_path:
            video_path.rename(target_path)
            result["video_path"] = str(target_path)
    
    def _cleanup_failed_folder(self, inference_dir: Path):
        """Clean up folder if video generation failed."""
        video_dir = inference_dir / "video"
        
        # Check if video directory exists and has content
        if video_dir.exists():
            video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm"))
            if video_files:
                return  # Keep folder if it has video files
        
        # Remove the entire inference directory if no videos were generated
        if inference_dir.exists():
            shutil.rmtree(inference_dir)
            print(f"   Cleaned up empty folder: {inference_dir.name}")
    
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


# ========================================
# CATALOG UTILITY FUNCTIONS
# ========================================

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
    """
    Add a new model family to the registry.
    
    Args:
        family_name: Name of the model family
        models: Dictionary of model configurations
    """
    # Add family info to each model
    for model_config in models.values():
        model_config["family"] = family_name
    
    # Add to registries
    MODEL_FAMILIES[family_name] = models
    AVAILABLE_MODELS.update(models)
