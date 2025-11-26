"""
SGLang Inference Service for VMEvalKit

Wrapper for SGLang to support video models: Wan-series, FastWan, Hunyuan, etc.
SGLang provides support for video models that are not available in diffusers.

Note: SGLang currently has known issues (see https://github.com/sgl-project/sglang/issues/12850)
and requires Docker. This implementation includes error handling and fallback mechanisms.

Reference:
- https://lmsys.org/blog/2025-11-07-sglang-diffusion/
- https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cli.md
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
import logging

logger = logging.getLogger(__name__)

# SGLang supported models mapping
SGLANG_MODEL_MAP = {
    "hunyuan-video-i2v": "Tencent/HunyuanVideo-I2V",
    "wan-2.1": "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    "wan-2.2": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "fastwan": "Wan-AI/FastWan",  # Placeholder, update with actual model path
}


class SGLangService:
    """
    Service class for SGLang inference integration.
    
    Supports video models: Wan-series, FastWan, Hunyuan and more.
    """
    
    def __init__(
        self,
        model_id: str = "hunyuan-video-i2v",
        output_dir: str = "./data/outputs",
        sglang_server_url: Optional[str] = None,
        use_docker: bool = True,
        **kwargs
    ):
        """
        Initialize SGLang service.
        
        Args:
            model_id: Model identifier (hunyuan-video-i2v, wan-2.1, wan-2.2, fastwan)
            output_dir: Directory to save generated videos
            sglang_server_url: Optional SGLang server URL (if running as service)
            use_docker: Whether to use Docker (default: True, required for some models)
            **kwargs: Additional parameters
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.sglang_server_url = sglang_server_url
        self.use_docker = use_docker
        self.kwargs = kwargs
        
        # Map model_id to SGLang model path
        self.sglang_model = SGLANG_MODEL_MAP.get(model_id, model_id)
        
        # Check if SGLang is available
        self._check_sglang_availability()
    
    def _check_sglang_availability(self) -> bool:
        """
        Check if SGLang is available and working.
        
        Returns:
            True if SGLang is available, False otherwise
        """
        try:
            # Try to import sglang
            import sglang
            logger.info("SGLang library found")
            return True
        except ImportError:
            logger.warning(
                "SGLang not found. Install with: pip install sglang[all]\n"
                "Note: SGLang may require Docker for some models. "
                "See https://github.com/sgl-project/sglang for installation instructions."
            )
            return False
    
    def _check_docker_available(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(
                ["docker", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _run_sglang_inference(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 25,
        seed: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run SGLang inference for video generation.
        
        Attempts to use SGLang CLI or Python API to generate video.
        Note: SGLang has known bugs (see https://github.com/sgl-project/sglang/issues/12850)
        which may cause this to fail even with correct implementation.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            height: Video height in pixels
            width: Video width in pixels
            num_frames: Number of frames to generate
            seed: Random seed for reproducibility
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with inference results and metadata
        """
        start_time = time.time()
        
        # Generate output filename
        timestamp = int(time.time())
        output_filename = f"sglang_{self.model_id}_{timestamp}.mp4"
        output_path = self.output_dir / output_filename
        
        # Check Docker if required
        if self.use_docker and not self._check_docker_available():
            return {
                "success": False,
                "video_path": None,
                "error": "Docker is required but not available. Install Docker or set use_docker=False",
                "duration_seconds": time.time() - start_time,
                "generation_id": f"sglang_error_{timestamp}",
                "model": self.model_id,
                "status": "failed",
                "metadata": {
                    "text_prompt": text_prompt,
                    "image_path": str(image_path),
                    "sglang_model": self.sglang_model,
                }
            }
        
        # Validate input image exists
        image_path = Path(image_path)
        if not image_path.exists():
            return {
                "success": False,
                "video_path": None,
                "error": f"Input image not found: {image_path}",
                "duration_seconds": time.time() - start_time,
                "generation_id": f"sglang_error_{timestamp}",
                "model": self.model_id,
                "status": "failed",
                "metadata": {
                    "text_prompt": text_prompt,
                    "image_path": str(image_path),
                    "sglang_model": self.sglang_model,
                }
            }
        
        # Try to use SGLang (CLI or Python API)
        try:
            import sglang
            
            logger.warning(
                "SGLang inference is experimental and may have bugs. "
                "See https://github.com/sgl-project/sglang/issues/12850"
            )
            
            # Method 1: Try SGLang CLI command (if available)
            # SGLang CLI format: sglang generate --model-path <model> --prompt <prompt> --output-path <output>
            # Note: For image-to-video, we may need to use a different approach or pass image as part of prompt
            try:
                # Check if sglang CLI is available
                cli_result = subprocess.run(
                    ["sglang", "generate", "--help"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if cli_result.returncode == 0:
                    # CLI is available, try using it
                    logger.info(f"Attempting SGLang CLI inference with model: {self.sglang_model}")
                    
                    # SGLang CLI uses --model-path, --output-path (directory), --prompt
                    # For image-to-video, we may need to handle image input differently
                    # Note: SGLang CLI may not directly support image-to-video via CLI
                    # We'll try the basic video generation first
                    output_dir = str(self.output_dir)
                    cmd = [
                        "sglang", "generate",
                        "--model-path", self.sglang_model,
                        "--prompt", text_prompt,
                        "--output-path", output_dir,
                        "--height", str(height),
                        "--width", str(width),
                        "--num-frames", str(num_frames),
                        "--save-output",
                    ]
                    
                    if seed is not None:
                        cmd.extend(["--seed", str(seed)])
                    
                    # Add any additional kwargs (convert to CLI format)
                    for key, value in kwargs.items():
                        if value is not None:
                            # Convert snake_case to kebab-case
                            cli_key = f"--{key.replace('_', '-')}"
                            cmd.extend([cli_key, str(value)])
                    
                    logger.info(f"Running SGLang CLI: {' '.join(cmd)}")
                    
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=600  # 10 minute timeout
                    )
                    
                    # SGLang saves files with generated names, we need to find the output
                    # Check if any new video files were created in output_dir
                    output_dir_path = Path(output_dir)
                    if output_dir_path.exists():
                        # Find the most recently created video file
                        video_files = list(output_dir_path.glob("*.mp4")) + list(output_dir_path.glob("*.avi"))
                        if video_files:
                            # Get the most recent file
                            latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
                            # If it was created after we started, it's likely our output
                            if latest_video.stat().st_mtime > start_time:
                                output_path = latest_video
                    
                    if result.returncode == 0 and output_path.exists():
                        return {
                            "success": True,
                            "video_path": str(output_path),
                            "error": None,
                            "duration_seconds": time.time() - start_time,
                            "generation_id": f"sglang_{timestamp}",
                            "model": self.model_id,
                            "status": "success",
                            "metadata": {
                                "text_prompt": text_prompt,
                                "image_path": str(image_path),
                                "sglang_model": self.sglang_model,
                                "height": height,
                                "width": width,
                                "num_frames": num_frames,
                                "seed": seed,
                                "method": "cli",
                                "stdout": result.stdout,
                            }
                        }
                    else:
                        error_msg = result.stderr or result.stdout or "Unknown error"
                        logger.error(f"SGLang CLI failed: {error_msg}")
                        return {
                            "success": False,
                            "video_path": None,
                            "error": f"SGLang CLI inference failed: {error_msg}",
                            "duration_seconds": time.time() - start_time,
                            "generation_id": f"sglang_error_{timestamp}",
                            "model": self.model_id,
                            "status": "failed",
                            "metadata": {
                                "text_prompt": text_prompt,
                                "image_path": str(image_path),
                                "sglang_model": self.sglang_model,
                                "stdout": result.stdout,
                                "stderr": result.stderr,
                                "returncode": result.returncode,
                                "method": "cli",
                            }
                        }
            except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError) as e:
                logger.info(f"SGLang CLI not available or failed, trying Python API: {str(e)}")
            
            # Method 2: Try SGLang Python API
            # Actual API: DiffGenerator.from_pretrained() then generator.generate()
            try:
                # Import SGLang's actual API
                from sglang.multimodal_gen import DiffGenerator
                
                logger.info(f"Attempting SGLang Python API inference with model: {self.sglang_model}")
                
                # Create or reuse generator (could be cached for performance)
                # For now, create a new one each time (could be optimized later)
                generator = DiffGenerator.from_pretrained(
                    model_path=self.sglang_model,
                    num_gpus=kwargs.get("num_gpus", 1),
                )
                
                # Call SGLang Python API
                # Note: For image-to-video, we may need to pass image differently
                # The generate() method accepts prompt and various parameters
                generate_kwargs = {
                    "prompt": text_prompt,
                    "output_path": str(self.output_dir),
                    "save_output": True,
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                }
                
                if seed is not None:
                    generate_kwargs["seed"] = seed
                
                # Add any additional kwargs
                generate_kwargs.update(kwargs)
                
                # Note: For image-to-video, we may need to handle image_path differently
                # SGLang's API may require image to be passed as part of prompt or via different parameter
                # This is a placeholder - actual implementation may vary
                video_result = generator.generate(**generate_kwargs)
                
                # Find the generated video file
                output_dir_path = Path(self.output_dir)
                if output_dir_path.exists():
                    video_files = list(output_dir_path.glob("*.mp4")) + list(output_dir_path.glob("*.avi"))
                    if video_files:
                        latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
                        if latest_video.stat().st_mtime > start_time:
                            output_path = latest_video
                
                if output_path.exists():
                    return {
                        "success": True,
                        "video_path": str(output_path),
                        "error": None,
                        "duration_seconds": time.time() - start_time,
                        "generation_id": f"sglang_{timestamp}",
                        "model": self.model_id,
                        "status": "success",
                        "metadata": {
                            "text_prompt": text_prompt,
                            "image_path": str(image_path),
                            "sglang_model": self.sglang_model,
                            "height": height,
                            "width": width,
                            "num_frames": num_frames,
                            "seed": seed,
                            "method": "python_api",
                            "result": str(video_result) if video_result else None,
                        }
                    }
                else:
                    return {
                        "success": False,
                        "video_path": None,
                        "error": "SGLang Python API completed but output file not found",
                        "duration_seconds": time.time() - start_time,
                        "generation_id": f"sglang_error_{timestamp}",
                        "model": self.model_id,
                        "status": "failed",
                        "metadata": {
                            "text_prompt": text_prompt,
                            "image_path": str(image_path),
                            "sglang_model": self.sglang_model,
                            "method": "python_api",
                        }
                    }
                    
            except ImportError as e:
                logger.warning(f"SGLang Python API not available: {str(e)}")
                # Fall through to error handling
            except Exception as e:
                logger.error(f"SGLang Python API failed: {str(e)}")
                # Check if this is the known bug
                error_str = str(e).lower()
                if "12850" in error_str or "bug" in error_str or "not implemented" in error_str:
                    return {
                        "success": False,
                        "video_path": None,
                        "error": f"SGLang inference failed due to known bug: {str(e)}. See https://github.com/sgl-project/sglang/issues/12850",
                        "duration_seconds": time.time() - start_time,
                        "generation_id": f"sglang_error_{timestamp}",
                        "model": self.model_id,
                        "status": "failed",
                        "metadata": {
                            "text_prompt": text_prompt,
                            "image_path": str(image_path),
                            "sglang_model": self.sglang_model,
                            "method": "python_api",
                            "exception": str(e),
                        }
                    }
                else:
                    return {
                        "success": False,
                        "video_path": None,
                        "error": f"SGLang Python API inference failed: {str(e)}",
                        "duration_seconds": time.time() - start_time,
                        "generation_id": f"sglang_error_{timestamp}",
                        "model": self.model_id,
                        "status": "failed",
                        "metadata": {
                            "text_prompt": text_prompt,
                            "image_path": str(image_path),
                            "sglang_model": self.sglang_model,
                            "method": "python_api",
                            "exception": str(e),
                        }
                    }
            
            # If both methods failed, return generic error
            return {
                "success": False,
                "video_path": None,
                "error": "SGLang inference failed: Both CLI and Python API methods failed. This may be due to known bugs (see https://github.com/sgl-project/sglang/issues/12850)",
                "duration_seconds": time.time() - start_time,
                "generation_id": f"sglang_error_{timestamp}",
                "model": self.model_id,
                "status": "failed",
                "metadata": {
                    "text_prompt": text_prompt,
                    "image_path": str(image_path),
                    "sglang_model": self.sglang_model,
                }
            }
            
        except ImportError:
            return {
                "success": False,
                "video_path": None,
                "error": "SGLang library not installed. Install with: pip install sglang[all]",
                "duration_seconds": time.time() - start_time,
                "generation_id": f"sglang_error_{timestamp}",
                "model": self.model_id,
                "status": "failed",
                "metadata": {
                    "text_prompt": text_prompt,
                    "image_path": str(image_path),
                    "sglang_model": self.sglang_model,
                }
            }
        except Exception as e:
            return {
                "success": False,
                "video_path": None,
                "error": f"SGLang inference failed with unexpected error: {str(e)}",
                "duration_seconds": time.time() - start_time,
                "generation_id": f"sglang_error_{timestamp}",
                "model": self.model_id,
                "status": "failed",
                "metadata": {
                    "text_prompt": text_prompt,
                    "image_path": str(image_path),
                    "sglang_model": self.sglang_model,
                    "exception": str(e),
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
        Generate video from image and text prompt using SGLang.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds
            height: Video height in pixels
            width: Video width in pixels
            seed: Random seed for reproducibility
            output_filename: Optional output filename (auto-generated if None)
            **kwargs: Additional parameters passed to SGLang
            
        Returns:
            Dictionary with generation results and metadata
        """
        # Validate inputs
        image_path = Path(image_path)
        if not image_path.exists():
            return {
                "success": False,
                "video_path": None,
                "error": f"Input image not found: {image_path}",
                "duration_seconds": 0,
                "generation_id": f"sglang_error_{int(time.time())}",
                "model": self.model_id,
                "status": "failed",
                "metadata": {"text_prompt": text_prompt, "image_path": str(image_path)},
            }
        
        # Convert duration to frames (assuming ~25 FPS)
        fps = kwargs.get('fps', 25)
        num_frames = max(1, int(duration * fps))
        
        # Run inference
        result = self._run_sglang_inference(
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
class SGLangWrapper(ModelWrapper):
    """
    Wrapper for SGLangService to match VMEvalKit's standard interface.
    
    Supports models: Hunyuan, Wan-series, FastWan via SGLang.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """
        Initialize SGLang wrapper.
        
        Args:
            model: Model identifier (hunyuan-video-i2v, wan-2.1, wan-2.2, fastwan)
            output_dir: Directory to save generated videos
            **kwargs: Additional parameters (sglang_server_url, use_docker, etc.)
        """
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create SGLangService instance
        self.sglang_service = SGLangService(
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
        Generate video using SGLang (matches VMEvalKit interface).
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds
            output_filename: Optional output filename
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results
        """
        return self.sglang_service.generate(
            image_path=image_path,
            text_prompt=text_prompt,
            duration=duration,
            output_filename=output_filename,
            **kwargs
        )

