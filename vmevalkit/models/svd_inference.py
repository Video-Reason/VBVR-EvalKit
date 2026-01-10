import os
import time
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from PIL import Image
from .base import ModelWrapper

logger = logging.getLogger(__name__)


class SVDService:
    
    def __init__(self, model: str = "stabilityai/stable-video-diffusion-img2vid-xt"):
        self.model_id = model
        self.pipe = None
        self.device = None
        
        self.model_constraints = {
            "recommended_size": (1024, 576),
            "fps": 7,
            "num_frames": 25,
            "num_inference_steps": 25,
            "motion_bucket_id": 127,
            "decode_chunk_size": 8
        }
    
    def _load_model(self):
        if self.pipe is not None:
            return
        
        logger.info(f"Loading SVD model: {self.model_id}")
        import torch
        from diffusers import StableVideoDiffusionPipeline
        
        if torch.cuda.is_available():
            self.device = "cuda"
            torch_dtype = torch.float16
            variant = "fp16"
        else:
            self.device = "cpu"
            torch_dtype = torch.float32
            variant = None
        
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            variant=variant
        )
        self.pipe.to(self.device)
        logger.info(f"SVD model loaded on {self.device}")
    
    def _prepare_image(self, image_path: Union[str, Path]) -> Image.Image:
        from diffusers.utils import load_image
        
        image = load_image(str(image_path))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        target_size = self.model_constraints["recommended_size"]
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        
        logger.info(f"Prepared image: {image.size}")
        return image
    
    def generate_video(
        self,
        image_path: Union[str, Path],
        text_prompt: str = "",
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        motion_bucket_id: Optional[int] = None,
        decode_chunk_size: Optional[int] = None,
        fps: Optional[int] = None,
        output_path: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        self._load_model()
        
        image = self._prepare_image(image_path)
        
        num_frames = num_frames or self.model_constraints["num_frames"]
        num_inference_steps = num_inference_steps or self.model_constraints["num_inference_steps"]
        motion_bucket_id = motion_bucket_id or self.model_constraints["motion_bucket_id"]
        decode_chunk_size = decode_chunk_size or self.model_constraints["decode_chunk_size"]
        fps = fps or self.model_constraints["fps"]
        
        logger.info(f"Generating video with {num_frames} frames, {num_inference_steps} steps")
        
        frames = self.pipe(
            image,
            num_frames=num_frames,
            decode_chunk_size=decode_chunk_size,
            num_inference_steps=num_inference_steps,
            motion_bucket_id=motion_bucket_id,
        ).frames[0]
        
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
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": duration_taken,
            "model": self.model_id,
            "status": "success" if video_path else "completed",
            "metadata": {
                "num_inference_steps": num_inference_steps,
                "motion_bucket_id": motion_bucket_id,
                "decode_chunk_size": decode_chunk_size,
                "image_size": image.size
            }
        }


class SVDWrapper(ModelWrapper):
    
    def __init__(
        self,
        model: str = "stabilityai/stable-video-diffusion-img2vid-xt",
        output_dir: str = "./outputs",
        **kwargs
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        self.svd_service = SVDService(model=model)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str = "",
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        fps = kwargs.get("fps", self.svd_service.model_constraints["fps"])
        if "num_frames" not in kwargs:
            kwargs["num_frames"] = int(duration * fps)
        
        if not output_filename:
            output_filename = "video.mp4"
        
        output_path = self.output_dir / output_filename
        
        result = self.svd_service.generate_video(
            image_path=str(image_path),
            text_prompt=text_prompt,
            output_path=output_path,
            **kwargs
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "success": bool(result.get("video_path")),
            "video_path": result.get("video_path"),
            "error": None,
            "duration_seconds": duration_taken,
            "generation_id": f"svd_{int(time.time())}",
            "model": self.model,
            "status": "success" if result.get("video_path") else "failed",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "num_frames": result.get("num_frames"),
                "fps": result.get("fps"),
                "svd_result": result
            }
        }
