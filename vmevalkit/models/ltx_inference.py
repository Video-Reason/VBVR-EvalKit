import time
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from PIL import Image
from .base import ModelWrapper

logger = logging.getLogger(__name__)


class LTXVideoService:
    
    def __init__(self, model: str = "Lightricks/LTX-Video"):
        self.model_id = model
        self.pipe = None
        self.device = None
        
        self.model_constraints = {
            "fps": 24,
            "num_frames": 161,
            "num_inference_steps": 50,
            "width": 704,
            "height": 480
        }
    
    def _load_model(self):
        if self.pipe is not None:
            return
        
        logger.info(f"Loading LTX model: {self.model_id}")
        import torch
        from diffusers import LTXImageToVideoPipeline
        
        if torch.cuda.is_available():
            self.device = "cuda"
            torch_dtype = torch.bfloat16
        else:
            self.device = "cpu"
            torch_dtype = torch.float32
        
        self.pipe = LTXImageToVideoPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype
        )
        self.pipe.to(self.device)
        logger.info(f"LTX model loaded on {self.device}")
    
    def _prepare_image(self, image_path: Union[str, Path]) -> Image.Image:
        from diffusers.utils import load_image
        
        image = load_image(str(image_path))
        
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        logger.info(f"Prepared image: {image.size}")
        return image
    
    def generate_video(
        self,
        image_path: Union[str, Path],
        text_prompt: str = "",
        negative_prompt: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        fps: Optional[int] = None,
        output_path: Optional[Path] = None,
        **kwargs
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        self._load_model()
        
        image = self._prepare_image(image_path)
        
        width = width or self.model_constraints["width"]
        height = height or self.model_constraints["height"]
        num_frames = num_frames or self.model_constraints["num_frames"]
        num_inference_steps = num_inference_steps or self.model_constraints["num_inference_steps"]
        fps = fps or self.model_constraints["fps"]
        
        if negative_prompt is None:
            negative_prompt = "worst quality, inconsistent motion, blurry, jittery, distorted"
        
        logger.info(f"Generating video with prompt: {text_prompt[:80]}...")
        logger.info(f"Dimensions: {width}x{height}, num_frames={num_frames}, steps={num_inference_steps}")
        
        output = self.pipe(
            image=image,
            prompt=text_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
        )
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
            "num_frames": num_frames,
            "fps": fps,
            "duration_seconds": duration_taken,
            "model": self.model_id,
            "status": "success" if video_path else "completed",
            "metadata": {
                "num_inference_steps": num_inference_steps,
                "width": width,
                "height": height,
                "image_size": image.size
            }
        }


class LTXVideoWrapper(ModelWrapper):
    
    def __init__(
        self,
        model: str = "Lightricks/LTX-Video",
        output_dir: str = "./outputs",
        **kwargs
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        self.ltx_service = LTXVideoService(model=model)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str = "",
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        start_time = time.time()
        
        # Filter out parameters not used by LTXVideoService
        kwargs.pop("question_data", None)
        
        negative_prompt = kwargs.pop("negative_prompt", None)
        width = kwargs.pop("width", None)
        height = kwargs.pop("height", None)
        num_frames = kwargs.pop("num_frames", None)
        num_inference_steps = kwargs.pop("num_inference_steps", None)
        fps = kwargs.pop("fps", None)
        
        if num_frames is None:
            fps = fps or self.ltx_service.model_constraints["fps"]
            num_frames = int(duration * fps)
        
        if not output_filename:
            output_filename = "video.mp4"
        
        output_path = self.output_dir / output_filename
        
        result = self.ltx_service.generate_video(
            image_path=str(image_path),
            text_prompt=text_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            fps=fps,
            output_path=output_path,
            **kwargs
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "success": bool(result.get("video_path")),
            "video_path": result.get("video_path"),
            "error": None,
            "duration_seconds": duration_taken,
            "generation_id": f"ltx_{int(time.time())}",
            "model": self.model,
            "status": "success" if result.get("video_path") else "failed",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "num_frames": result.get("num_frames"),
                "fps": result.get("fps"),
                "ltx_result": result
            }
        }
