"""
HunyuanVideo-I2V Inference Service for VMEvalKit

Wrapper for the HunyuanVideo-I2V model (submodules/HunyuanVideo-I2V) to integrate with VMEvalKit's
unified inference interface. Supports high-quality image-to-video generation up to 720p.
"""

import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .base import ModelWrapper
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
        output_dir: str = "./outputs",
        model_python_interpreter: str = None,
        **kwargs
    ):
        """
        Initialize HunyuanVideo-I2V service.
        
        Args:
            model_id: HunyuanVideo model variant (currently only one available)
            output_dir: Directory to save generated videos
            model_python_interpreter: Python interpreter to use (defaults to sys.executable)
            **kwargs: Additional parameters
        """
        self.model_id = model_id
        # Resolve to an absolute path so the subprocess (running from the submodule dir)
        # writes to the same folder we later read from.
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.model_python_interpreter = model_python_interpreter or sys.executable
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
        timestamp = int(start_time)
        
        # Use output directory directly (no timestamp subfolder)
        # The InferenceRunner already creates a properly structured directory
        output_dir = self.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Determine i2v resolution based on height
        if height >= 720:
            i2v_resolution = "720p"
        elif height >= 540:
            i2v_resolution = "540p"
        else:
            i2v_resolution = "360p"
        
        # Get model base path from kwargs or use default
        model_base = kwargs.get('model_base', str(HUNYUAN_PATH / "ckpts" / "hunyuan-video-i2v-720p"))
        model_type = kwargs.get('model_type', 'HYVideo-T/2')
        infer_steps = kwargs.get('infer_steps', 50)
        embedded_cfg_scale = kwargs.get('embedded_cfg_scale', 6.0)
        
        # Single GPU mode (default) - xfuser not required
        # For multi-GPU, set num_gpus > 1 and ensure xfuser is installed
        num_gpus = kwargs.get('num_gpus', 1)  # Default to single GPU
        use_cpu_offload = kwargs.get('use_cpu_offload', True)  # Recommended for single GPU
        
        # Prepare inference command
        cmd = [
            self.model_python_interpreter,
            str(HUNYUAN_PATH / "sample_image2video.py"),
        ]
        
        # Add common arguments
        cmd.extend([
            "--prompt", text_prompt,
            "--i2v-image-path", str(image_path),
            "--video-size", str(width), str(height),
            "--video-length", str(video_length),
            "--save-path", str(output_dir),
            "--model", model_type,
            "--model-base", model_base,
            "--i2v-mode",
            "--i2v-resolution", i2v_resolution,
            "--infer-steps", str(infer_steps),
            "--embedded-cfg-scale", str(embedded_cfg_scale),
        ])
        
        # Add CPU offload for single GPU (reduces memory usage)
        if use_cpu_offload and num_gpus == 1:
            cmd.append("--use-cpu-offload")
        
        # Add stability parameter
        if use_i2v_stability:
            cmd.append("--i2v-stability")
        
        # Add flow shift parameter
        if flow_shift:
            cmd.extend(["--flow-shift", str(flow_shift)])
        
        # Add flow-reverse (recommended by docs)
        cmd.append("--flow-reverse")
        
        # Add seed
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        else:
            cmd.extend(["--seed", "0"])  # Use deterministic seed by default
            
        # Add any additional parameters (skip already handled ones and metadata)
        skip_keys = ['use_i2v_stability', 'flow_shift', 'model_base', 'model_type', 
                     'infer_steps', 'embedded_cfg_scale', 'question_data', 'duration', 
                     'output_filename', 'fps', 'num_gpus', 'use_cpu_offload']
        for key, value in kwargs.items():
            if value is not None and key not in skip_keys:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        # Preflight: HunyuanVideo also expects a CLIP-L text encoder/tokenizer at ckpts/text_encoder_2.
        # If it's missing, inference will fail deep inside Transformers with a confusing error.
        clip_text_encoder_dir = HUNYUAN_PATH / "ckpts" / "text_encoder_2"
        if not clip_text_encoder_dir.exists() or not any(clip_text_encoder_dir.iterdir()):
            raise FileNotFoundError(
                f"Missing required checkpoint directory: {clip_text_encoder_dir}\n"
                f"Run: /home/ubuntu/Hokin/VMEvalKit/setup/models/hunyuan-video-i2v/setup.sh"
            )
        
        try:
            # Ensure HunyuanVideo resolves ckpt-relative paths (e.g. ./ckpts/text_encoder_i2v)
            # regardless of the caller's working directory, without modifying the submodule.
            env = os.environ.copy()
            env.setdefault("MODEL_BASE", str(HUNYUAN_PATH / "ckpts"))
            
            # Enable xDiT parallel inference resizing (per docs)
            if num_gpus > 1:
                env["ALLOW_RESIZE_FOR_SP"] = "1"


            # Change to HunyuanVideo directory and run inference
            result = subprocess.run(
                cmd,
                cwd=str(HUNYUAN_PATH),
                env=env,
                capture_output=True,
                text=True,
                timeout=7200  # 120 minute timeout for large model
            )
            
            success = result.returncode == 0
            error_msg = result.stderr if result.returncode != 0 else None
            
            # Find the generated video file in the output directory
            output_video = None
            if success and output_dir.exists():
                # HunyuanVideo may create videos in nested directories
                # Search recursively and flatten to video.mp4
                video_files = list(output_dir.glob("**/*.mp4"))
                if video_files:
                    source_video = video_files[0]
                    final_video_path = output_dir / "video.mp4"
                    
                    # Move/rename to simple path
                    if source_video != final_video_path:
                        shutil.move(str(source_video), str(final_video_path))
                    
                    # Clean up any nested directories created by the model
                    for item in output_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                    
                    output_video = str(final_video_path)
                else:
                    success = False
                    error_msg = f"Video generation succeeded but no .mp4 file found in {output_dir}"
            
        except subprocess.TimeoutExpired:
            success = False
            error_msg = "HunyuanVideo inference timed out"
            output_video = None
        except Exception as e:
            success = False
            error_msg = f"HunyuanVideo inference failed: {str(e)}"
            output_video = None
        
        duration = time.time() - start_time
        
        return {
            "success": success,
            "video_path": output_video,
            "error": error_msg,
            "duration_seconds": duration,
            "generation_id": f"hunyuan_{timestamp}",
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
        # Per docs: max supported is 129 frames (5 seconds)
        fps = kwargs.get('fps', 25)
        video_length = max(1, int(duration * fps))
        # Ensure odd number of frames (HunyuanVideo requirement)
        if video_length % 2 == 0:
            video_length += 1
        # Cap to max supported length per documentation
        video_length = min(video_length, 129)
        
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
class HunyuanVideoWrapper(ModelWrapper):
    """
    Wrapper for HunyuanVideoService to match VMEvalKit's standard interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./outputs",
        **kwargs
    ):
        """Initialize HunyuanVideo wrapper."""
        # Properly initialize the base class
        super().__init__(model, output_dir, **kwargs)
        
        # Create HunyuanVideoService instance with model-specific Python interpreter
        self.hunyuan_service = HunyuanVideoService(
            model_id=model, 
            output_dir=str(self.output_dir), 
            model_python_interpreter=self.get_model_python_interpreter(),
            **kwargs
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
        # Sync service output_dir with wrapper output_dir before each generation
        # This ensures videos are saved to the correct location when wrapper is cached
        self.hunyuan_service.output_dir = self.output_dir
        
        return self.hunyuan_service.generate(
            image_path=image_path,
            text_prompt=text_prompt,
            duration=duration,
            output_filename=output_filename,
            **kwargs
        )
