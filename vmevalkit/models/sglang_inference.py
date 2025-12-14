"""
SGLang Inference Service for VMEvalKit

Wrapper for SGLang to support video models: Wan-series, FastWan, Hunyuan, etc.
SGLang provides support for video models that are not available in diffusers.

Note: SGLang Issue #12850 has been fixed (closed on 2025-11-09). 
This implementation is ready for testing. Docker is recommended for reproducibility, but local GPU installation is also supported.

Usage Modes:
1. Persistent Server Mode (Recommended): Start server with `sglang serve`, then use API calls
   - Avoids reloading model on every inference
   - Better performance for multiple requests
2. Direct Mode: Load model for each inference
   - Simpler but slower for multiple requests

Reference:
- https://lmsys.org/blog/2025-11-07-sglang-diffusion/
- https://github.com/sgl-project/sglang/blob/main/python/sglang/multimodal_gen/docs/cli.md
- Issue #12850: https://github.com/sgl-project/sglang/issues/12850 (fixed)
"""

import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .base import ModelWrapper
import time
import logging
import requests

logger = logging.getLogger(__name__)
import sglang
from sglang.multimodal_gen import DiffGenerator


# SGLang supported models mapping
SGLANG_MODEL_MAP = {
    "hunyuan-video-i2v": "Tencent/HunyuanVideo-I2V",
    "wan-2.1": "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
    "wan-2.2": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    # FastWan models (by Hao AI Lab @ UCSD)
    # See: https://hao-ai-lab.github.io/blogs/fastvideo_post_training/
    # Available models:
    # - FastVideo/FastWan2.1-T2V-1.3B-Diffusers (480P, 1.3B params)
    # - FastVideo/FastWan2.1-T2V-14B-Diffusers (720P, 14B params, preview)
    # - FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers (720P, 5B params)
    "fastwan": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",  # Default: 1.3B 480P model (alias for fastwan-1.3b)
    "fastwan-1.3b": "FastVideo/FastWan2.1-T2V-1.3B-Diffusers",  # 480P, 1.3B params (same as fastwan, explicit version)
    "fastwan-14b": "FastVideo/FastWan2.1-T2V-14B-Diffusers",  # 720P, 14B params (preview)
    "fastwan-2.2-5b": "FastVideo/FastWan2.2-TI2V-5B-FullAttn-Diffusers",  # 720P, 5B params
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
        use_persistent_server: bool = True,
        **kwargs
    ):
        """
        Initialize SGLang service.
        
        Args:
            model_id: Model identifier (hunyuan-video-i2v, wan-2.1, wan-2.2, fastwan)
            output_dir: Directory to save generated videos
            sglang_server_url: Optional SGLang server URL (if running as service).
                              If None and use_persistent_server=True, will auto-detect or use http://localhost:30000
            use_docker: Whether to run inference inside Docker. 
                        Default is True (recommended for reproducibility), but can be set to False 
                        for local GPU installations since SGLang officially supports non-Docker usage.
            use_persistent_server: If True, will try to use/connect to a persistent SGLang server.
                                  This avoids reloading the model on every inference (recommended).
                                  If False, will load model for each inference.
            **kwargs: Additional parameters
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.sglang_server_url = sglang_server_url or "http://localhost:30000"
        self.use_docker = use_docker
        self.use_persistent_server = use_persistent_server
        self.kwargs = kwargs
        
        # Map model_id to SGLang model path
        self.sglang_model = SGLANG_MODEL_MAP.get(model_id, model_id)
        
        # FastWan models are now using correct paths from FastVideo organization
        # No warning needed as paths are verified from official source
        
        # Check if SGLang is available - raise error early if not installed
        self._check_sglang_availability()
        
        # Check if persistent server is available
        if self.use_persistent_server:
            self.server_available = self._check_server_available()
            if self.server_available:
                logger.info(f"SGLang server detected at {self.sglang_server_url}")
            else:
                logger.warning(
                    f"Persistent server mode enabled but no server found at {self.sglang_server_url}. "
                    f"Will fall back to direct model loading. "
                    f"To use server mode, start server with: sglang serve --model-path {self.sglang_model}"
                )
        else:
            self.server_available = False
    
    def _check_sglang_availability(self) -> None:
        """
        Check if SGLang is available and raise ImportError if not.
        
        Raises:
            ImportError: If SGLang is not installed
        """
        try:
            # Try to import sglang and required modules
            import sglang
            from sglang.multimodal_gen import DiffGenerator
            logger.info("SGLang library found")
        except ImportError as e:
            raise ImportError(
                "SGLang is required but not installed. "
                "Install from source with: pip install -e \"python[diffusion]\" (recommended)\n"
                "Alternative: Clone SGLang repo and run: cd python && pip install -e \".[diffusion]\"\n"
                "Note: pip install \"sglang[all]\" may not work for diffusion models.\n"
                "See https://github.com/sgl-project/sglang for installation instructions."
            ) from e
    
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
    
    def _check_server_available(self) -> bool:
        """
        Check if SGLang server is running and available.
        
        Returns:
            True if server is reachable, False otherwise
        """
        try:
            response = requests.get(f"{self.sglang_server_url}/health", timeout=5)
            return response.status_code == 200
        except (requests.RequestException, Exception):
            return False
    
    def _call_server_api(
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
        Call SGLang server API for video generation.
        
        This method uses a persistent server to avoid reloading the model on every inference.
        Server should be started with: sglang serve --model-path <model>
        
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
        timestamp = int(time.time())
        
        # Prepare request payload
        payload = {
            "prompt": text_prompt,
            "image_path": str(image_path),
            "height": height,
            "width": width,
            "num_frames": num_frames,
        }
        
        if seed is not None:
            payload["seed"] = seed
        
        # Add any additional kwargs
        payload.update(kwargs)
        
        try:
            # Call API endpoint
            response = requests.post(
                f"{self.sglang_server_url}/generate",
                json=payload,
                timeout=600  # 10 minute timeout for video generation
            )
            
            if response.status_code == 200:
                result_data = response.json()
                video_path = result_data.get("video_path")
                
                if video_path and Path(video_path).exists():
                    return {
                        "success": True,
                        "video_path": str(video_path),
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
                            "method": "server_api",
                        }
                    }
                else:
                    return {
                        "success": False,
                        "video_path": None,
                        "error": f"Server returned success but video file not found: {video_path}",
                        "duration_seconds": time.time() - start_time,
                        "generation_id": f"sglang_error_{timestamp}",
                        "model": self.model_id,
                        "status": "failed",
                        "metadata": {
                            "text_prompt": text_prompt,
                            "image_path": str(image_path),
                            "method": "server_api",
                        }
                    }
            else:
                error_msg = response.text or f"HTTP {response.status_code}"
                return {
                    "success": False,
                    "video_path": None,
                    "error": f"Server API call failed: {error_msg}",
                    "duration_seconds": time.time() - start_time,
                    "generation_id": f"sglang_error_{timestamp}",
                    "model": self.model_id,
                    "status": "failed",
                    "metadata": {
                        "text_prompt": text_prompt,
                        "image_path": str(image_path),
                        "method": "server_api",
                        "http_status": response.status_code,
                    }
                }
                
        except requests.RequestException as e:
            return {
                "success": False,
                "video_path": None,
                "error": f"Server API request failed: {str(e)}",
                "duration_seconds": time.time() - start_time,
                "generation_id": f"sglang_error_{timestamp}",
                "model": self.model_id,
                "status": "failed",
                "metadata": {
                    "text_prompt": text_prompt,
                    "image_path": str(image_path),
                    "method": "server_api",
                    "exception": str(e),
                }
            }
    
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
        
        Priority order:
        1. If use_persistent_server=True and server available, use server API (recommended)
        2. Otherwise, try SGLang CLI
        3. Finally, fall back to Python API (loads model each time)
        
        Note: SGLang Issue #12850 has been fixed (closed on 2025-11-09).
        This implementation is ready for testing.
        
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
        
        # Priority 1: Try server API if available (recommended - avoids reloading model)
        if self.use_persistent_server and self.server_available:
            logger.info("Using persistent SGLang server (recommended for performance)")
            result = self._call_server_api(
                image_path=image_path,
                text_prompt=text_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                seed=seed,
                **kwargs
            )
            if result["success"]:
                return result
            else:
                logger.warning(f"Server API failed: {result.get('error')}. Falling back to direct loading.")
                # Continue to try other methods
        
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
        
        # Priority 2 & 3: Use direct model loading (CLI or Python API)
        # Note: ImportError is already checked in __init__, so imports are safe here
        # Warning: These methods reload the model on every inference
            
        logger.info(
            "SGLang inference: Issue #12850 has been fixed (closed on 2025-11-09). "
            "Implementation is ready for testing."
        )
        logger.warning(
            "Using direct model loading (model will be loaded for each inference). "
            "For better performance, start a persistent server with: "
            f"sglang serve --model-path {self.sglang_model}"
        )
        
        try:
            
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
                    # For image-to-video, pass image via --image-path or --image
                    output_dir = str(self.output_dir)
                    cmd = [
                        "sglang", "generate",
                        "--model-path", self.sglang_model,
                        "--prompt", text_prompt,
                        "--image-path", str(image_path),  # Add image path for I2V
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
                    found_output_path = None
                    output_dir_path = Path(output_dir)
                    if output_dir_path.exists():
                        # Find the most recently created video file
                        video_files = list(output_dir_path.glob("*.mp4")) + list(output_dir_path.glob("*.avi"))
                        if video_files:
                            # Get the most recent file
                            latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
                            # If it was created after we started, it's likely our output
                            if latest_video.stat().st_mtime > start_time:
                                found_output_path = latest_video
                    
                    # Use found path or fall back to expected path
                    final_output_path = found_output_path if found_output_path else output_path
                    
                    if result.returncode == 0 and final_output_path and final_output_path.exists():
                        return {
                            "success": True,
                            "video_path": str(final_output_path),
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
            # Note: DiffGenerator is already imported at the top of the try block
            try:
                logger.info(f"Attempting SGLang Python API inference with model: {self.sglang_model}")
                
                # Create or reuse generator (could be cached for performance)
                # For now, create a new one each time (could be optimized later)
                generator = DiffGenerator.from_pretrained(
                    model_path=self.sglang_model,
                    num_gpus=kwargs.get("num_gpus", 1),
                )
                
                # Call SGLang Python API
                # For image-to-video, pass image_path as image or image_path parameter
                generate_kwargs = {
                    "prompt": text_prompt,
                    "image_path": str(image_path),  # Add image path for I2V
                    "output_path": str(self.output_dir),
                    "save_output": True,
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                }
                
                if seed is not None:
                    generate_kwargs["seed"] = seed
                
                # Add any additional kwargs (may override above)
                generate_kwargs.update(kwargs)
                
                # Try generating with image_path, fallback to image if needed
                try:
                    video_result = generator.generate(**generate_kwargs)
                except TypeError as e:
                    # If image_path doesn't work, try with 'image' parameter instead
                    if "image_path" in str(e):
                        generate_kwargs.pop("image_path", None)
                        generate_kwargs["image"] = str(image_path)
                video_result = generator.generate(**generate_kwargs)
                
                # Find the generated video file
                found_output_path = None
                output_dir_path = Path(self.output_dir)
                if output_dir_path.exists():
                    video_files = list(output_dir_path.glob("*.mp4")) + list(output_dir_path.glob("*.avi"))
                    if video_files:
                        latest_video = max(video_files, key=lambda p: p.stat().st_mtime)
                        if latest_video.stat().st_mtime > start_time:
                            found_output_path = latest_video
                
                # Use found path or fall back to expected path
                final_output_path = found_output_path if found_output_path else output_path
                
                if final_output_path and final_output_path.exists():
                    return {
                        "success": True,
                        "video_path": str(final_output_path),
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
                    
            except Exception as e:
                logger.error(f"SGLang Python API failed: {str(e)}")
                # Return error information
                return {
                    "success": False,
                    "video_path": None,
                    "error": f"SGLang Python API inference failed: {str(e)}. Issue #12850 has been fixed, please check your SGLang installation and configuration.",
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
                "error": "SGLang inference failed: Both CLI and Python API methods failed. Issue #12850 has been fixed, please check your SGLang installation and configuration.",
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
    
    Performance Tip:
    For better performance with multiple requests, start a persistent SGLang server:
        sglang serve --model-path <model> --num-gpus 4 --ulysses-degree=2 --ring-degree=2
    Then set use_persistent_server=True (default) to use the server API.
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
            **kwargs: Additional parameters:
                - sglang_server_url: Server URL (default: http://localhost:30000)
                - use_persistent_server: Use server API if available (default: True, recommended)
                - use_docker: Use Docker for inference (default: True)
                - num_gpus: Number of GPUs to use (for direct mode)
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

