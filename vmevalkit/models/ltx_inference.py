"""
LTX-Video Inference Service for VMEvalKit

Wrapper for the LTX-Video model (submodules/LTX-Video) to integrate with VMEvalKit's
unified inference interface. Supports image-to-video generation with text prompts.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .base import ModelWrapper
import json
import time

# Add LTX-Video submodule to path
LTXV_PATH = Path(__file__).parent.parent.parent / "submodules" / "LTX-Video"
sys.path.insert(0, str(LTXV_PATH))

try:
    from ltx_video.inference import LTXVInference
except ImportError:
    print(f"import ltx_video.inference failed, please check if the module is installed")
    LTXVInference = None


class LTXVideoService:
    """
    Service class for LTX-Video inference integration.
    """
    
    def __init__(
        self,
        model_id: str = "ltxv-13b-0.9.8-distilled",
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """
        Initialize LTX-Video service.
        
        Args:
            model_id: LTX-Video model variant to use
            output_dir: Directory to save generated videos
            **kwargs: Additional parameters
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Map model IDs to config files
        self.config_mapping = {
            "ltxv-13b-0.9.8-distilled": "configs/ltxv-13b-0.9.8-distilled.yaml",
            "ltxv-13b-0.9.8-dev": "configs/ltxv-13b-0.9.8-dev.yaml", 
            "ltxv-2b-0.9.8-distilled": "configs/ltxv-2b-0.9.8-distilled.yaml",
            "ltxv-2b-0.9.6-distilled": "configs/ltxv-2b-0.9.6-distilled.yaml",
        }
        
        self.config_path = LTXV_PATH / self.config_mapping.get(
            model_id, "configs/ltxv-13b-0.9.8-distilled.yaml"
        )
        
        # Check if LTX-Video is available
        if LTXVInference is None:
            raise ImportError(
                f"LTX-Video not available. Please initialize submodule:\n"
                f"cd {LTXV_PATH.parent} && git submodule update --init LTX-Video"
            )

    def _run_ltx_inference(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        height: int = 512,
        width: int = 512, 
        num_frames: int = 16,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run LTX-Video inference using subprocess to avoid dependency conflicts.
        
        Returns:
            Dictionary with inference results and metadata
        """
        start_time = time.time()
        
        # Generate output filename
        timestamp = int(time.time())
        output_filename = f"ltxv_{self.model_id}_{timestamp}.mp4"
        output_path = self.output_dir / output_filename
        
        # Prepare inference command
        cmd = [
            sys.executable,
            str(LTXV_PATH / "inference.py"),
            "--prompt", text_prompt,
            "--conditioning_media_paths", str(image_path),
            "--conditioning_start_frames", "0",
            "--height", str(height),
            "--width", str(width),
            "--num_frames", str(num_frames),
            "--pipeline_config", str(self.config_path),
            "--output_path", str(output_path)
        ]
        
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
            
        # Add any additional parameters
        for key, value in kwargs.items():
            if value is not None:
                cmd.extend([f"--{key}", str(value)])
        
        try:
            # Change to LTX-Video directory and run inference
            result = subprocess.run(
                cmd,
                cwd=str(LTXV_PATH),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            success = result.returncode == 0
            error_msg = result.stderr if result.returncode != 0 else None
            
        except subprocess.TimeoutExpired:
            success = False
            error_msg = "LTX-Video inference timed out"
        except Exception as e:
            success = False
            error_msg = f"LTX-Video inference failed: {str(e)}"
        
        duration = time.time() - start_time
        
        return {
            "success": success,
            "video_path": str(output_path) if success and output_path.exists() else None,
            "error": error_msg,
            "duration_seconds": duration,
            "generation_id": f"ltx_{int(time.time())}",
            "model": self.model_id,
            "status": "success" if success else "failed",
            "metadata": {
                "text_prompt": text_prompt,
                "image_path": str(image_path),
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "seed": seed,
                "stdout": result.stdout if 'result' in locals() else None,
                "stderr": result.stderr if 'result' in locals() else None,
            }
        }

    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video from image and text prompt.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (converted to num_frames)
            height: Video height in pixels
            width: Video width in pixels  
            seed: Random seed for reproducibility
            output_filename: Optional output filename (auto-generated if None)
            **kwargs: Additional parameters passed to LTX-Video
            
        Returns:
            Dictionary with generation results and metadata
        """
        # Convert duration to frames (assuming 8 FPS)
        fps = kwargs.get('fps', 8)
        num_frames = max(1, int(duration * fps))
        
        # Validate inputs
        image_path = Path(image_path)
        if not image_path.exists():
            return {
                "success": False,
                "video_path": None,
                "error": f"Input image not found: {image_path}",
                "duration_seconds": 0,
                "generation_id": f"ltx_error_{int(time.time())}",
                "model": self.model_id,
                "status": "failed",
                "metadata": {"text_prompt": text_prompt, "image_path": str(image_path)},
            }
        
        # Run inference
        result = self._run_ltx_inference(
            image_path=image_path,
            text_prompt=text_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
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
class LTXVideoWrapper(ModelWrapper):
    """
    Wrapper for LTXVideoService to match VMEvalKit's standard interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs", 
        **kwargs
    ):
        """Initialize LTX-Video wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create LTXVideoService instance
        self.ltx_service = LTXVideoService(model_id=model, output_dir=output_dir, **kwargs)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using LTX-Video (matches VMEvalKit interface).
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds
            output_filename: Optional output filename
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results
        """
        return self.ltx_service.generate(
            image_path=image_path,
            text_prompt=text_prompt,
            duration=duration,
            output_filename=output_filename,
            **kwargs
        )
