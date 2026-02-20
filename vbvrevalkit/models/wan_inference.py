import time
import numpy as np
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from PIL import Image
from .base import ModelWrapper
from ..utils.image import load_image_rgb

logger = logging.getLogger(__name__)


class WanService:
    
    def __init__(self, model: str = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers"):
        self.model_id = model
        self.pipe = None
        self.image_encoder = None
        self.vae = None
        self.device = None
        
        self.model_constraints = {
            "max_area": 720 * 1280,
            "fps": 16,
            "guidance_scale": 5.5
        }
    
    def _load_model(self):
        if self.pipe is not None:
            return
        
        logger.info(f"Loading WAN model: {self.model_id}")
        import torch
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        from transformers import CLIPVisionModel
        
        if torch.cuda.is_available():
            self.device = "cuda"
            model_dtype = torch.bfloat16
        else:
            self.device = "cpu"
            model_dtype = torch.float32
        
        load_kwargs = {
            "low_cpu_mem_usage": True,
            "torch_dtype": model_dtype
        }
        
        # Load pipeline directly - it will load all components automatically
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            self.model_id,
            **load_kwargs
        )
        self.pipe.to(self.device)
        logger.info(f"WAN model loaded on {self.device}")
    
    def _aspect_ratio_resize(self, image: Image.Image, max_area: Optional[int] = None) -> tuple:
        if max_area is None:
            max_area = self.model_constraints["max_area"]
        
        aspect_ratio = image.height / image.width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        
        resized_image = image.resize((width, height), Image.Resampling.LANCZOS)
        
        logger.info(f"Aspect ratio resize: {image.size} -> {resized_image.size}")
        return resized_image, height, width

    def _prepare_image(self, image_path: Union[str, Path]) -> tuple:
        """Load, convert to RGB, and resize image using aspect ratio constraints."""
        image = load_image_rgb(image_path)
        image, height, width = self._aspect_ratio_resize(image)
        logger.info(f"Prepared image: {image.size}")
        return image, height, width

    def generate_video(
        self,
        image_path: Union[str, Path],
        text_prompt: str = "",
        guidance_scale: Optional[float] = None,
        fps: Optional[int] = None,
        output_path: Optional[Path] = None,
        num_frames: Optional[int] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        start_time = time.time()
        self._load_model()

        if height is not None and width is not None:
            image = load_image_rgb(image_path)
            mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
            height = round(height / mod_value) * mod_value
            width = round(width / mod_value) * mod_value
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image to aligned dimensions: {width}x{height}")
        else:
            image, height, width = self._prepare_image(image_path)
        
        guidance_scale = guidance_scale or self.model_constraints["guidance_scale"]
        fps = fps or self.model_constraints["fps"]
        
        logger.info(f"Generating video with prompt: {text_prompt[:80]}...")
        logger.info(f"Dimensions: {width}x{height}, guidance_scale={guidance_scale}, fps={fps}, num_frames={num_frames}")
        
        # Build pipeline kwargs
        pipe_kwargs = {
            "image": image,
            "prompt": text_prompt,
            "height": height,
            "width": width,
            "guidance_scale": guidance_scale
        }
        if num_frames is not None:
            pipe_kwargs["num_frames"] = num_frames
        
        output = self.pipe(**pipe_kwargs)
        frames = output.frames[0]
        
        video_path = None
        if output_path:
            from diffusers.utils import export_to_video
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(output_path), fps=fps)
            video_path = str(output_path)
            logger.info(f"Video saved to: {video_path}")
        
        duration_taken = time.time() - start_time
        
        return {
            "video_path": video_path,
            "frames": frames,
            "fps": fps,
            "duration_seconds": duration_taken,
            "model": self.model_id,
            "status": "success" if video_path else "completed",
            "metadata": {
                "guidance_scale": guidance_scale,
                "height": height,
                "width": width,
                "image_size": image.size
            }
        }


class WanWrapper(ModelWrapper):
    
    def __init__(
        self,
        model: str = "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
        output_dir: str = "./outputs",
        **kwargs
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        self.wan_service = WanService(model=model)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str = "",
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        guidance_scale = kwargs.pop("guidance_scale", None)
        fps = kwargs.pop("fps", None)
        num_frames = kwargs.pop("num_frames", None)
        height = kwargs.pop("height", None)
        width = kwargs.pop("width", None)
        kwargs.pop("question_data", None)

        output_path = self.output_dir / (output_filename or "video.mp4")

        result = self.wan_service.generate_video(
            image_path=str(image_path),
            text_prompt=text_prompt,
            guidance_scale=guidance_scale,
            fps=fps,
            output_path=output_path,
            num_frames=num_frames,
            height=height,
            width=width,
            **kwargs
        )

        has_video = bool(result.get("video_path"))
        return {
            "success": has_video,
            "video_path": result.get("video_path"),
            "error": None,
            "duration_seconds": result.get("duration_seconds", 0),
            "generation_id": f"wan_{int(time.time())}",
            "model": self.model,
            "status": "success" if has_video else "failed",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "fps": result.get("fps"),
                "wan_result": result
            }
        }
