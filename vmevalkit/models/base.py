"""
Base classes for VMEvalKit model wrappers.

Provides abstract interfaces to ensure consistency across all video generation models.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Union, Optional
from pathlib import Path
import sys


class ModelWrapper(ABC):
    """
    Abstract base class for all model wrappers in VMEvalKit.
    
    Ensures consistent interface across all video generation models while allowing
    each implementation to handle their specific service logic.
    """
    
    def __init__(self, model: str, output_dir: str = "./outputs", **kwargs):
        """
        Initialize model wrapper.
        
        Args:
            model: Model identifier/name
            output_dir: Directory to save generated videos
            **kwargs: Additional model-specific parameters
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Get VMEvalKit root directory
        self.vmeval_root = Path(__file__).parent.parent.parent
    
    def get_model_python_interpreter(self, model_id: Optional[str] = None) -> str:
        """
        Get the Python interpreter path for the model-specific virtual environment.
        
        Args:
            model_id: Model identifier (defaults to self.model)
            
        Returns:
            Path to model-specific Python interpreter, falls back to system Python if not found
        """
        if model_id is None:
            model_id = self.model
            
        model_venv_python = self.vmeval_root / "envs" / model_id / "bin" / "python"
        
        if model_venv_python.exists():
            print(f"Using model-specific Python: {model_venv_python}")
            return str(model_venv_python)
        else:
            print(f"Warning: Model-specific venv not found at {model_venv_python}, using system Python")
            return sys.executable
    
    @abstractmethod
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
        
        Args:
            image_path: Path to input image (first frame)
            text_prompt: Text instructions for video generation
            duration: Video duration in seconds
            output_filename: Optional output filename (auto-generated if None)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Dictionary with standardized keys:
            - success: bool - Whether generation succeeded
            - video_path: str | None - Path to generated video file
            - error: str | None - Error message if failed
            - duration_seconds: float - Time taken for generation
            - generation_id: str - Unique identifier for this generation
            - model: str - Model name/identifier used
            - status: str - Generation status ("success", "failed", etc.)
            - metadata: Dict[str, Any] - Additional metadata
        """
        pass


class ModelService(ABC):
    """
    Abstract base class for model services.
    
    Optional base class for the underlying service implementations.
    Services handle the actual API calls/inference logic.
    """
    
    @abstractmethod
    async def generate_video(
        self,
        prompt: str,
        image_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using the model service.
        
        Args:
            prompt: Text prompt for generation
            image_path: Path to input image
            **kwargs: Service-specific parameters
            
        Returns:
            Service-specific result dictionary
        """
        pass
