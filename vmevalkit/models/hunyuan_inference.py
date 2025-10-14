"""
HunyuanVideo-I2V Inference Service for VMEvalKit

Wrapper for the HunyuanVideo-I2V model (submodules/HunyuanVideo-I2V) to integrate with VMEvalKit's
unified inference interface. Supports high-quality image-to-video generation up to 720p.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
import json
import time

# Add HunyuanVideo-I2V submodule to path
HUNYUAN_PATH = Path(__file__).parent.parent.parent / "submodules" / "HunyuanVideo-I2V"
sys.path.insert(0, str(HUNYUAN_PATH))


class HunyuanVideoService:
    """
    Service class for HunyuanVideo-I2V inference integration.
    """
    
    def __init__(
        self,
        model_id: str = "hunyuan-video-i2v",
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """
        Initialize HunyuanVideo-I2V service.
        
        Args:
            model_id: HunyuanVideo model variant (currently only one available)
            output_dir: Directory to save generated videos
            **kwargs: Additional parameters
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Check if HunyuanVideo-I2V is available
        self.inference_script = HUNYUAN_PATH / "sample_image2video.py"
        if not self.inference_script.exists():
            raise FileNotFoundError(
                f"HunyuanVideo-I2V inference script not found at {self.inference_script}.\n"
                f"Please initialize submodule:\n"
                f"cd {HUNYUAN_PATH.parent} && git submodule update --init HunyuanVideo-I2V"
            )

    def _run_hunyuan_inference(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        height: int = 720,
        width: int = 1280,
        video_length: int = 129,  # HunyuanVideo uses frame counts
        seed: Optional[int] = None,
        use_i2v_stability: bool = True,
        flow_shift: float = 7.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run HunyuanVideo-I2V inference using subprocess.
        
        Returns:
            Dictionary with inference results and metadata
        """
        start_time = time.time()
        
        # Generate output filename
        timestamp = int(time.time())
        output_filename = f"hunyuan_{timestamp}.mp4"
        output_path = self.output_dir / output_filename
        
        # Prepare inference command
        cmd = [
            sys.executable,
            str(self.inference_script),
            "--prompt", text_prompt,
            "--image-path", str(image_path),
            "--height", str(height),
            "--width", str(width),
            "--video-length", str(video_length),
            "--output-path", str(output_path),
        ]
        
        # Add stability and flow shift parameters
        if use_i2v_stability:
            cmd.append("--i2v-stability")
        if flow_shift:
            cmd.extend(["--flow-shift", str(flow_shift)])
        
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
            
        # Add any additional parameters
        for key, value in kwargs.items():
            if value is not None and key not in ['use_i2v_stability', 'flow_shift']:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        try:
            # Change to HunyuanVideo directory and run inference
            result = subprocess.run(
                cmd,
                cwd=str(HUNYUAN_PATH),
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout for large model
            )
            
            success = result.returncode == 0
            error_msg = result.stderr if result.returncode != 0 else None
            
        except subprocess.TimeoutExpired:
            success = False
            error_msg = "HunyuanVideo inference timed out"
        except Exception as e:
            success = False
            error_msg = f"HunyuanVideo inference failed: {str(e)}"
        
        duration = time.time() - start_time
        
        return {
            "success": success,
            "video_path": str(output_path) if success and output_path.exists() else None,
            "error": error_msg,
            "duration_seconds": duration,
            "generation_id": f"hunyuan_{int(time.time())}",
            "model": self.model_id,
            "status": "success" if success else "failed",
            "metadata": {
                "text_prompt": text_prompt,
                "image_path": str(image_path),
                "height": height,
                "width": width,
                "video_length": video_length,
                "seed": seed,
                "use_i2v_stability": use_i2v_stability,
                "flow_shift": flow_shift,
                "stdout": result.stdout if 'result' in locals() else None,
                "stderr": result.stderr if 'result' in locals() else None,
            }
        }

    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        height: int = 720,
        width: int = 1280,
        seed: Optional[int] = None,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video from image and text prompt.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (converted to frames)
            height: Video height in pixels (720p recommended)
            width: Video width in pixels (1280 for 720p)
            seed: Random seed for reproducibility
            output_filename: Optional output filename (auto-generated if None)
            **kwargs: Additional parameters passed to HunyuanVideo
            
        Returns:
            Dictionary with generation results and metadata
        """
        # Convert duration to frames (HunyuanVideo uses ~25 FPS)
        fps = kwargs.get('fps', 25)
        video_length = max(1, int(duration * fps))
        # Ensure odd number of frames (HunyuanVideo requirement)
        if video_length % 2 == 0:
            video_length += 1
        
        # Validate inputs
        image_path = Path(image_path)
        if not image_path.exists():
            return {
                "success": False,
                "video_path": None,
                "error": f"Input image not found: {image_path}",
                "duration_seconds": 0,
                "generation_id": f"hunyuan_error_{int(time.time())}",
                "model": self.model_id,
                "status": "failed",
                "metadata": {"text_prompt": text_prompt, "image_path": str(image_path)},
            }
        
        # Check GPU memory requirements
        if height >= 720:
            print(f"Warning: HunyuanVideo requires 60-80GB GPU memory for {height}p generation")
        
        # Run inference
        result = self._run_hunyuan_inference(
            image_path=image_path,
            text_prompt=text_prompt,
            height=height,
            width=width,
            video_length=video_length,
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
class HunyuanVideoWrapper:
    """
    Wrapper for HunyuanVideoService to match VMEvalKit's standard interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        api_key: Optional[str] = None,  # Not used for local inference
        **kwargs
    ):
        """Initialize HunyuanVideo wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create HunyuanVideoService instance
        self.hunyuan_service = HunyuanVideoService(
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
        Generate video using HunyuanVideo-I2V (matches VMEvalKit interface).
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds
            output_filename: Optional output filename
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results
        """
        return self.hunyuan_service.generate(
            image_path=image_path,
            text_prompt=text_prompt,
            duration=duration,
            output_filename=output_filename,
            **kwargs
        )
