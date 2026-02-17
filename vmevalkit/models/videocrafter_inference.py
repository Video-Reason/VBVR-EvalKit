"""
VideoCrafter Inference Service for VMEvalKit

Wrapper for the VideoCrafter model (submodules/VideoCrafter) to integrate with VMEvalKit's
unified inference interface. Supports text-guided image-to-video generation.

This implementation loads the model once and performs real diffusion-based video generation.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .base import ModelWrapper
import time

# Add VideoCrafter submodule to path
VIDEOCRAFTER_PATH = Path(__file__).parent.parent.parent / "submodules" / "VideoCrafter"
VMEVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(VIDEOCRAFTER_PATH))
sys.path.insert(0, str(VIDEOCRAFTER_PATH / "scripts" / "evaluation"))

# Import torch and checkpoint module BEFORE importing VideoCrafter modules
# VideoCrafter's lvdm.common module expects torch.utils.checkpoint to be available
import torch
import torch.utils.checkpoint

# VideoCrafter optionally uses xFormers "memory_efficient_attention" for *spatial* attention
# when xformers is importable. On some GPU/driver/CUDA combos this can crash at runtime with:
#   RuntimeError: cutlassF: no kernel found to launch!
# To keep VMEvalKit inference robust across machines, we force VideoCrafter to use the
# standard PyTorch attention implementation (no xformers fast-path) by flipping the
# module-level availability flag *before* the model is instantiated.
import importlib

_vc_attention = importlib.import_module("lvdm.modules.attention")
_vc_attention.XFORMERS_IS_AVAILBLE = False
import torchvision
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

# Import VideoCrafter modules
from funcs import load_model_checkpoint, load_image_batch, batch_ddim_sampling
from utils.utils import instantiate_from_config


class VideoCrafterService:
    """
    Service class for VideoCrafter inference integration.
    Loads model once and performs real diffusion-based video generation.
    """
    
    def __init__(
        self,
        model_id: str = "videocrafter2-512",
        output_dir: str = "./outputs",
        **kwargs
    ):
        """
        Initialize VideoCrafter service and load model.
        
        Args:
            model_id: VideoCrafter model variant (e.g., "videocrafter2-512")
            output_dir: Directory to save generated videos
            **kwargs: Additional parameters (device, ddim_steps, cfg_scale, etc.)
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Parse model variant
        self.resolution = 512  # Default
        if "512" in model_id:
            self.resolution = 512
        elif "1024" in model_id:
            self.resolution = 1024
        
        # Model paths
        # Note: VideoCrafter has separate models for t2v and i2v
        # i2v model is from VideoCrafter/Image2Video-512-v1.0 (has IP-Adapter + image encoder)
        # t2v model is from VideoCrafter/VideoCrafter2 (text-only, no image conditioning)
        self.config_path = VIDEOCRAFTER_PATH / "configs" / "inference_i2v_512_v1.0.yaml"
        self.ckpt_path = VMEVAL_ROOT / "weights" / "videocrafter" / "i2v_512_v1" / "model.ckpt"
        
        # Check if files exist
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"VideoCrafter config not found at {self.config_path}.\n"
                f"Please ensure VideoCrafter submodule is initialized."
            )
        
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                f"VideoCrafter checkpoint not found at {self.ckpt_path}.\n"
                f"Please download the model checkpoint. See setup script for details."
            )
        
        # Load model once
        self._load_model()

    def _load_model(self):
        """Load VideoCrafter model once at initialization."""
        print(f"Loading VideoCrafter model from {self.ckpt_path}...")
        
        # Load config
        config = OmegaConf.load(self.config_path)
        model_config = config.pop("model", OmegaConf.create())
        
        # Disable gradient checkpointing for inference (faster)
        if 'unet_config' in model_config.get('params', {}):
            model_config['params']['unet_config']['params']['use_checkpoint'] = False
        
        # Instantiate model
        self.model = instantiate_from_config(model_config)
        
        # Load checkpoint
        self.model = load_model_checkpoint(self.model, str(self.ckpt_path))
        
        # Move to GPU and set to eval mode
        self.model = self.model.cuda()
        self.model.eval()
        
        # Store model properties
        self.channels = self.model.channels
        self.temporal_length = self.model.temporal_length
        
        print(f"✓ VideoCrafter model loaded successfully")
        print(f"  - Temporal length: {self.temporal_length} frames")
        print(f"  - Channels: {self.channels}")

    def _run_videocrafter_inference(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        height: int = 320,
        width: int = 512,
        num_frames: Optional[int] = None,
        fps: int = 8,
        seed: Optional[int] = None,
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        cfg_scale: float = 12.0,
        save_fps: int = 8,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run real VideoCrafter diffusion-based inference.
        
        This performs actual video generation using the loaded model,
        not a placeholder or subprocess hack.
        
        Returns:
            Dictionary with inference results and metadata
        """
        start_time = time.time()
        timestamp = int(start_time)
        
        # Set random seed if provided
        if seed is not None:
            seed_everything(seed)
        
        # Use model's default temporal length if not specified
        if num_frames is None:
            num_frames = self.temporal_length
        
        # Generate output filename
        output_filename = "video.mp4"
        output_path = self.output_dir / output_filename
        
        # Ensure output directory exists
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        error_msg = None
        
        # Perform inference
        with torch.no_grad():
            # Set up batch parameters
            batch_size = 1
            h, w = height // 8, width // 8  # Latent space dimensions
            noise_shape = [batch_size, self.channels, num_frames, h, w]
            
            # Load and preprocess image
            cond_images = load_image_batch([str(image_path)], (height, width))
            cond_images = cond_images.to(self.model.device)
            
            # Get text embeddings
            text_emb = self.model.get_learned_conditioning([text_prompt])
            
            # Get image embeddings
            img_emb = self.model.get_image_embeds(cond_images)
            
            # Concatenate text and image embeddings (i2v conditioning)
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
            
            # Prepare conditioning with fps
            fps_tensor = torch.tensor([fps] * batch_size).to(self.model.device).long()
            cond = {"c_crossattn": [imtext_cond], "fps": fps_tensor}
            
            # Run DDIM sampling (actual diffusion inference!)
            batch_samples = batch_ddim_sampling(
                self.model,
                cond,
                noise_shape,
                n_samples=1,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
                cfg_scale=cfg_scale,
                **kwargs
            )
            
            # Process and save video
            # batch_samples shape: [batch, samples, c, t, h, w]
            vid_tensor = batch_samples[0, 0]  # Get first batch, first sample
            video = vid_tensor.detach().cpu()
            video = torch.clamp(video.float(), -1.0, 1.0)
            
            # Rearrange to [t, c, h, w]
            video = video.permute(1, 0, 2, 3)  # c,t,h,w -> t,c,h,w
            
            # Normalize to [0, 1]
            video = (video + 1.0) / 2.0
            
            # Convert to uint8 [t, h, w, c]
            video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)
            
            # Save video using torchvision
            torchvision.io.write_video(
                str(output_path),
                video,
                fps=save_fps,
                video_codec="h264",
                options={"crf": "10"}
            )
        
        success = output_path.exists() and output_path.stat().st_size > 0
        duration = time.time() - start_time
        
        return {
            "success": success,
            "video_path": str(output_path) if success else None,
            "error": error_msg,
            "duration_seconds": duration,
            "generation_id": f"videocrafter_{timestamp}",
            "model": self.model_id,
            "status": "success" if success else "failed",
            "metadata": {
                "text_prompt": text_prompt,
                "image_path": str(image_path),
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "fps": fps,
                "save_fps": save_fps,
                "seed": seed,
                "ddim_steps": ddim_steps,
                "ddim_eta": ddim_eta,
                "cfg_scale": cfg_scale,
            }
        }

    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        height: int = 320,
        width: int = 512,
        seed: Optional[int] = None,
        output_filename: Optional[str] = None,
        ddim_steps: int = 50,
        cfg_scale: float = 12.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video from image and text prompt using VideoCrafter diffusion.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (used to calculate frames)
            height: Video height in pixels (default 320 for i2v_512 model)
            width: Video width in pixels (default 512 for i2v_512 model)
            seed: Random seed for reproducibility
            output_filename: Optional output filename (auto-generated if None)
            ddim_steps: Number of DDIM sampling steps (higher = better quality, slower)
            cfg_scale: Classifier-free guidance scale (higher = stronger prompt following)
            **kwargs: Additional parameters (fps, ddim_eta, save_fps, etc.)
            
        Returns:
            Dictionary with generation results and metadata
        """
        # VideoCrafter i2v_512 defaults: 320x512
        # Calculate frames from duration
        fps = kwargs.get('fps', 8)
        num_frames = kwargs.get('num_frames')
        if num_frames is None:
            # Use model's temporal length, ignore duration for now
            # VideoCrafter models have fixed temporal length
            num_frames = None  # Will use self.temporal_length
        
        # Validate inputs
        image_path = Path(image_path)
        if not image_path.exists():
            return {
                "success": False,
                "video_path": None,
                "error": f"Input image not found: {image_path}",
                "duration_seconds": 0,
                "generation_id": f"videocrafter_error_{int(time.time())}",
                "model": self.model_id,
                "status": "failed",
                "metadata": {"text_prompt": text_prompt, "image_path": str(image_path)},
            }
        
        # Run real diffusion inference
        result = self._run_videocrafter_inference(
            image_path=image_path,
            text_prompt=text_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            ddim_steps=ddim_steps,
            cfg_scale=cfg_scale,
            **kwargs
        )
        
        # Handle custom output filename
        if output_filename and result["success"] and result["video_path"]:
            old_path = Path(result["video_path"])
            new_path = self.output_dir / output_filename
            if old_path.exists():
                old_path.rename(new_path)
                result["video_path"] = str(new_path)
        
        return result


# Wrapper class to match VMEvalKit's interface pattern
class VideoCrafterWrapper(ModelWrapper):
    """
    Wrapper for VideoCrafterService to match VMEvalKit's standard interface.
    
    This wrapper loads the VideoCrafter model once and performs real diffusion-based
    video generation for each request, not subprocess-based placeholder generation.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./outputs",
        **kwargs
    ):
        """
        Initialize VideoCrafter wrapper.
        
        Args:
            model: Model identifier (e.g., "videocrafter2-512")
            output_dir: Directory to save generated videos
            **kwargs: Additional parameters passed to VideoCrafterService
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create VideoCrafterService instance (loads model)
        self.videocrafter_service = VideoCrafterService(
            model_id=model, output_dir=output_dir, **kwargs
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
        Generate video using VideoCrafter diffusion model (matches VMEvalKit interface).
        
        Performs real text-guided image-to-video generation using the VideoCrafter
        diffusion model. This is NOT a placeholder or frame duplication.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (note: VideoCrafter has fixed temporal length)
            output_filename: Optional output filename
            **kwargs: Additional parameters (height, width, ddim_steps, cfg_scale, seed, etc.)
            
        Returns:
            Dictionary with generation results including:
                - success: bool
                - video_path: str (path to generated video)
                - error: Optional[str]
                - duration_seconds: float (inference time)
                - metadata: dict (full generation parameters)
        """
        # Sync service output_dir with wrapper output_dir before each generation
        # This ensures videos are saved to the correct location when wrapper is cached
        self.videocrafter_service.output_dir = self.output_dir
        
        return self.videocrafter_service.generate(
            image_path=image_path,
            text_prompt=text_prompt,
            duration=duration,
            output_filename=output_filename,
            **kwargs
        )
