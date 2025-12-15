"""
CogVideoX Image-to-Video Generation Integration for VMEvalKit.

Implements CogVideoX-5B-I2V and CogVideoX1.5-5B-I2V models from Zhipu AI/THUDM
using Hugging Face Diffusers pipelines.

Models:
- CogVideoX-5B-I2V: 6s videos (49 frames @ 8fps) at 720x480
- CogVideoX1.5-5B-I2V: 10s videos (81 frames @ 16fps) at 1360x768
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from pydantic import BaseModel, Field, field_validator

from .base import ModelWrapper

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models for Configuration
# ============================================================================

class CogVideoXConfig(BaseModel):
    """Configuration for CogVideoX generation parameters."""
    
    model_id: str
    resolution: tuple[int, int] = Field(default=(720, 480))
    num_frames: int = Field(default=49, ge=1, le=81)
    fps: int = Field(default=8, ge=1, le=30)
    guidance_scale: float = Field(default=6.0)
    num_inference_steps: int = Field(default=50, ge=1, le=100)
    
    @field_validator('resolution')
    @classmethod
    def validate_resolution(cls, v):
        """Ensure resolution meets CogVideoX requirements."""
        width, height = v
        min_dim = min(width, height)
        max_dim = max(width, height)
        
        if min_dim < 480:
            raise ValueError(f"Minimum dimension must be >= 480, got {min_dim}")
        if max_dim > 1360:
            raise ValueError(f"Maximum dimension must be <= 1360, got {max_dim}")
        if max_dim % 16 != 0:
            raise ValueError(f"Maximum dimension must be divisible by 16, got {max_dim}")
        
        return v


class GenerationResult(BaseModel):
    """Pydantic model for generation results."""
    
    success: bool
    video_path: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float
    generation_id: str
    model: str
    status: str
    metadata: Dict[str, Any]


# ============================================================================
# Service Class (Direct Diffusers Pipeline)
# ============================================================================

class CogVideoXService:
    """
    Service for CogVideoX I2V inference using Diffusers pipeline.
    
    Follows the SVD pattern for direct pipeline usage with memory optimizations.
    """
    
    def __init__(self, config: CogVideoXConfig):
        """
        Initialize CogVideoX service.
        
        Args:
            config: Pydantic configuration model
        """
        self.config = config
        self.pipe = None
        self.device = None
    
    def _load_model(self):
        """Lazy load the CogVideoX pipeline with memory optimizations."""
        if self.pipe is not None:
            return
        
        logger.info(f"Loading CogVideoX model: {self.config.model_id}")
        
        from diffusers import CogVideoXImageToVideoPipeline
        
        # Determine device and dtype
        if torch.cuda.is_available():
            self.device = "cuda"
            torch_dtype = torch.bfloat16  # CogVideoX uses BF16
        else:
            self.device = "cpu"
            torch_dtype = torch.float32
        
        # Load pipeline from HuggingFace
        self.pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype
        )
        
        # CRITICAL: Memory optimizations to reduce VRAM from ~26GB to ~5-10GB
        self.pipe.enable_sequential_cpu_offload()
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()
        
        logger.info(
            f"CogVideoX model loaded on {self.device} with memory optimizations "
            f"(sequential offload + VAE tiling/slicing)"
        )
    
    def _prepare_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load and prepare image for CogVideoX.
        
        Args:
            image_path: Path to input image
            
        Returns:
            Prepared PIL Image resized to target resolution
        """
        from diffusers.utils import load_image
        
        image = load_image(str(image_path))
        
        # Ensure RGB mode
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize to target resolution
        target_width, target_height = self.config.resolution
        image = image.resize(
            (target_width, target_height),
            Image.Resampling.LANCZOS
        )
        
        logger.info(f"Prepared image: {image.size} (mode={image.mode})")
        return image
    
    def generate_video(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        output_path: Path,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video from image and text prompt.
        
        NOTE: No try-except here per user requirement "Never use Try - Catch".
        Exceptions will propagate to caller (wrapper layer).
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            output_path: Path where video should be saved
            seed: Random seed for reproducibility (default: 42)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results and metadata
        """
        start_time = time.time()
        
        # Load model if not already loaded
        self._load_model()
        
        # Prepare input image
        image = self._prepare_image(image_path)
        
        # Create generator for reproducibility (fixed seed = temperature 0)
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        logger.info(
            f"Generating {self.config.num_frames} frames at {self.config.fps}fps "
            f"with guidance_scale={self.config.guidance_scale} (seed={seed})"
        )
        
        # Generate video frames
        result = self.pipe(
            prompt=text_prompt,
            image=image,
            num_frames=self.config.num_frames,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,  # Fixed for reproducibility
            generator=generator,
            **kwargs
        )
        
        frames = result.frames[0]
        
        # Export to video file
        from diffusers.utils import export_to_video
        output_path.parent.mkdir(parents=True, exist_ok=True)
        export_to_video(frames, str(output_path), fps=self.config.fps)
        
        duration_taken = time.time() - start_time
        logger.info(f"Video saved to: {output_path} (took {duration_taken:.2f}s)")
        
        return {
            "video_path": str(output_path),
            "frames": frames,
            "num_frames": self.config.num_frames,
            "fps": self.config.fps,
            "duration_seconds": duration_taken,
            "model": self.config.model_id,
            "status": "success",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "resolution": self.config.resolution,
                "guidance_scale": self.config.guidance_scale,
                "num_inference_steps": self.config.num_inference_steps,
                "seed": seed
            }
        }


# ============================================================================
# Wrapper Class (VMEvalKit Interface)
# ============================================================================

class CogVideoXWrapper(ModelWrapper):
    """
    Wrapper for CogVideoX models conforming to VMEvalKit interface.
    
    Supports:
    - CogVideoX-5B-I2V: 6s videos at 720x480 (49 frames @ 8fps)
    - CogVideoX1.5-5B-I2V: 10s videos at 1360x768 (81 frames @ 16fps)
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """
        Initialize CogVideoX wrapper.
        
        Args:
            model: Model identifier (HuggingFace repo path)
            output_dir: Directory to save generated videos
            **kwargs: Additional parameters (including 'args' from catalog)
        """
        super().__init__(model, output_dir, **kwargs)
        
        # Create Pydantic config from catalog args
        config_dict = {
            "model_id": model,
            **kwargs.get("args", {})  # Get args from MODEL_CATALOG
        }
        self.config = CogVideoXConfig(**config_dict)
        
        # Initialize service
        self.service = CogVideoXService(config=self.config)
        
        logger.info(
            f"Initialized CogVideoX: {model} "
            f"({self.config.num_frames} frames @ {self.config.fps}fps, "
            f"resolution={self.config.resolution})"
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
        
        Returns standardized VMEvalKit result dictionary with all 8 required fields.
        
        Args:
            image_path: Path to input image (first frame)
            text_prompt: Text instructions for video generation
            duration: Video duration in seconds (ignored, uses model config)
            output_filename: Optional output filename (auto-generated if None)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with standardized VMEvalKit keys:
            - success: bool
            - video_path: str | None
            - error: str | None
            - duration_seconds: float
            - generation_id: str
            - model: str
            - status: str
            - metadata: Dict[str, Any]
        """
        start_time = time.time()
        
        # Validate input image exists
        image_path = Path(image_path)
        if not image_path.exists():
            return GenerationResult(
                success=False,
                video_path=None,
                error=f"Input image not found: {image_path}",
                duration_seconds=0,
                generation_id=f"cogvideox_error_{int(time.time())}",
                model=self.model,
                status="failed",
                metadata={
                    "text_prompt": text_prompt,
                    "image_path": str(image_path)
                }
            ).model_dump()
        
        # Generate output filename if not provided
        if not output_filename:
            timestamp = int(time.time())
            safe_model = self.model.replace("/", "-").replace("_", "-")
            output_filename = f"cogvideox_{safe_model}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Set seed for reproducibility (equivalent to temperature=0)
        seed = kwargs.get("seed", 42)  # Default deterministic seed
        
        # Generate video (exceptions will propagate as per user requirement)
        result = self.service.generate_video(
            image_path=image_path,
            text_prompt=text_prompt,
            output_path=output_path,
            seed=seed,
            **kwargs
        )
        
        # Convert to VMEvalKit standardized format
        total_duration = time.time() - start_time
        
        return GenerationResult(
            success=True,
            video_path=result["video_path"],
            error=None,
            duration_seconds=total_duration,
            generation_id=f"cogvideox_{int(time.time())}",
            model=self.model,
            status="success",
            metadata={
                **result["metadata"],
                "duration_requested": duration,
                "output_filename": output_filename
            }
        ).model_dump()

