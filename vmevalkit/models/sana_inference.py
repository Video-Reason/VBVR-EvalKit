"""SANA-Video Integration for VMEvalKit

Uses SanaImageToVideoPipeline from diffusers for image → video generation.

Supported model:
- sana-video-2b-480p: Short-video model (~5 seconds, 81 frames, 480x832)

Performance: ~22GB VRAM, ~4 minutes on RTX A6000 (50 steps)

Requirements:
- diffusers>=0.36.0

References:
- HuggingFace: https://huggingface.co/Efficient-Large-Model/SANA-Video_2B_480p_diffusers
- Diffusers: https://huggingface.co/docs/diffusers/main/en/api/pipelines/sana_video
"""

import time
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging

import torch
from PIL import Image
from diffusers import SanaImageToVideoPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video, load_image

from .base import ModelWrapper

logger = logging.getLogger(__name__)


class SanaVideoService:
    """Service for SANA-Video inference using diffusers pipeline.
    
    Uses SanaVideoPipeline for text+image to video generation.
    The 2B backbone supports text-only, image-only, and text+image modes.
    
    Features:
    - Motion score control for video dynamics
    - Negative prompt support
    - Seed-based reproducibility
    - Automatic image resizing to model constraints
    - Memory-optimized VAE (float32) for stability
    """
    
    def __init__(self, model: str = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers"):
        """Initialize SANA-Video service.
        
        Args:
            model: HuggingFace model ID for SANA-Video
        """
        self.model_id = model
        self.pipe = None
        self.device = None
        
        # SANA-Video 2B 480p default constraints
        self.model_constraints = {
            "height": 480,
            "width": 832,
            "num_frames": 81,
            "fps": 16,
            "guidance_scale": 6.0,
            "num_inference_steps": 50
        }
    
    def _load_model(self):
        """Lazy load the SANA-Video pipeline with optimized dtypes.
        
        Uses bfloat16 for transformer and text encoder, float32 for VAE
        to balance memory usage and numerical stability.
        """
        if self.pipe is not None:
            return
        
        logger.info(f"Loading SANA-Video model: {self.model_id}")
        
        if torch.cuda.is_available():
            self.device = "cuda"
            transformer_dtype = torch.bfloat16
            encoder_dtype = torch.bfloat16
            vae_dtype = torch.float32  # Keep VAE at float32 for stability
        else:
            self.device = "cpu"
            transformer_dtype = torch.float32
            encoder_dtype = torch.float32
            vae_dtype = torch.float32
        
        logger.info("Using pipeline: SanaImageToVideoPipeline")
        self.pipe = SanaImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=transformer_dtype
        )
        self.pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config, flow_shift=8.0
        )

        self.pipe.vae.to(vae_dtype)
        self.pipe.text_encoder.to(encoder_dtype)
        self.pipe.to(self.device)
        
        logger.info(f"SANA-Video model loaded on {self.device}")
        logger.info(f"Dtypes - Transformer: {transformer_dtype}, Encoder: {encoder_dtype}, VAE: {vae_dtype}")

    def _prepare_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load and prepare image for inference.
        
        Args:
            image_path: Path to input image
            
        Returns:
            PIL Image in RGB mode
        """
        image = load_image(str(image_path))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        logger.info(f"Prepared image for SANA: {image.size}")
        return image

    def generate_video(
        self,
        image_path: Union[str, Path],
        text_prompt: str = "",
        negative_prompt: str = "",
        motion_score: int = 30,
        num_frames: int = 81,
        guidance_scale: float = 6.0,
        num_inference_steps: int = 50,
        height: int = 480,
        width: int = 832,
        seed: int = 42,
        fps: int = 16,
        output_path: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video from image and text prompt.
        
        Args:
            image_path: Path to input image
            text_prompt: Text description for video generation
            negative_prompt: Negative prompt for generation control
            motion_score: Motion intensity score (0-100, appended to prompt)
            num_frames: Number of frames to generate (default: 81)
            guidance_scale: Classifier-free guidance scale (default: 6.0)
            num_inference_steps: Denoising steps (default: 50)
            height: Output video height (default: 480)
            width: Output video width (default: 832)
            seed: Random seed for reproducibility
            fps: Output video FPS (default: 16)
            output_path: Path to save output video
            **kwargs: Additional pipeline arguments
            
        Returns:
            Dictionary with video_path, frames, and metadata
        """
        start_time = time.time()

        self._load_model()
        image = self._prepare_image(image_path)

        # Compose prompt with motion score
        motion_prompt = f" motion score: {motion_score}."
        composed_prompt = text_prompt + motion_prompt
        
        logger.info(f"Generating video with prompt: {composed_prompt[:80]}...")
        logger.info(f"Dimensions: {width}x{height}, frames={num_frames}, steps={num_inference_steps}")

        generator = torch.Generator(device=self.device).manual_seed(seed)
        logger.info(f"Using seed: {seed}")

        # Generate using SanaVideoPipeline or LongSanaVideoPipeline
        # Note: diffusers uses 'frames' parameter, not 'num_frames'
        pipeline_kwargs = {
            "image": image,
            "prompt": composed_prompt,
            "height": height,
            "width": width,
            "frames": num_frames,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
        }
        
        if negative_prompt:
            pipeline_kwargs["negative_prompt"] = negative_prompt
            logger.info(f"Using negative prompt: {negative_prompt[:50]}...")

        output = self.pipe(**pipeline_kwargs)
        frames_output = output.frames[0]

        video_path = None
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames_output, str(output_path), fps=fps)
            video_path = str(output_path)
            logger.info(f"SANA video saved to: {video_path}")

        duration_taken = time.time() - start_time

        return {
            "video_path": video_path,
            "frames": frames_output,
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": duration_taken,
            "model": self.model_id,
            "status": "success" if video_path else "completed",
            "metadata": {
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "image_size": image.size,
                "motion_score": motion_score,
                "seed": seed,
                "negative_prompt": negative_prompt
            }
        }


class SanaVideoWrapper(ModelWrapper):
    """Wrapper for SANA-Video models conforming to VMEvalKit interface.
    
    Supports both base 480p model and LongLive extended variant.
    Provides advanced features:
    - Motion score control for video dynamics
    - Negative prompt support
    - Reproducible generation via seed
    - Automatic parameter optimization
    """
    
    def __init__(
        self,
        model: str = "Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        output_dir: str = "./outputs",
        **kwargs
    ):
        """Initialize SANA-Video wrapper.
        
        Args:
            model: HuggingFace model ID
            output_dir: Directory for output videos
            **kwargs: Additional configuration parameters
        """
        super().__init__(model=model, output_dir=output_dir, **kwargs)
        self.sana_service = SanaVideoService(model=model)

    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str = "",
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video from image and text prompt.
        
        Args:
            image_path: Path to input image
            text_prompt: Text description for video generation
            duration: Desired video duration in seconds (used to calculate frames)
            output_filename: Custom output filename (auto-generated if not provided)
            **kwargs: Additional parameters:
                - num_frames: Override frame count
                - fps: Frames per second (default: 16)
                - height: Video height (default: 480)
                - width: Video width (default: 832)
                - num_inference_steps: Denoising steps (default: 50)
                - guidance_scale: CFG scale (default: 6.0)
                - negative_prompt: Negative prompt for control
                - motion_score: Motion intensity (0-100)
                - seed: Random seed for reproducibility
            
        Returns:
            Dictionary with success, video_path, error, duration_seconds,
            generation_id, model, status, and metadata fields
        """
        start_time = time.time()

        fps = kwargs.get("fps", 16)
        if "num_frames" not in kwargs:
            kwargs["num_frames"] = max(1, int(duration * fps))

        kwargs.setdefault("height", 480)
        kwargs.setdefault("width", 832)
        kwargs.setdefault("guidance_scale", 6.0)
        kwargs.setdefault("num_inference_steps", 50)
        kwargs.setdefault("motion_score", 30)
        kwargs.setdefault("seed", 42)
        kwargs.setdefault("fps", fps)

        if not output_filename:
            output_filename = "video.mp4"
        
        output_path = self.output_dir / output_filename

        result = self.sana_service.generate_video(
            image_path=str(image_path),
            text_prompt=text_prompt,
            output_path=output_path,
            **kwargs,
        )

        duration_taken = time.time() - start_time

        return {
            "success": bool(result.get("video_path")),
            "video_path": result.get("video_path"),
            "error": None,
            "duration_seconds": duration_taken,
            "generation_id": f"sana_{int(time.time())}",
            "model": self.model,
            "status": "success" if result.get("video_path") else "failed",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "num_frames": result.get("num_frames"),
                "fps": result.get("fps"),
                "sana_result": result
            }
        }
