"""
OpenAI Sora Image-to-Video Generation Service.

Supports text + image → video generation using OpenAI's Sora-2 and Sora-2-Pro models.
"""

import os
import time
import asyncio
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from PIL import Image

logger = logging.getLogger(__name__)


class SoraService:
    """Service for image-to-video generation using OpenAI Sora models."""
    
    def __init__(self, model: str = "sora-2"):
        """
        Initialize Sora service.
        
        Args:
            model: Sora model to use ("sora-2" or "sora-2-pro")
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.model = model
        self.base_url = "https://api.openai.com/v1"
        
        # Validate model and set constraints
        self.model_constraints = self._get_model_constraints(model)
    
    def _get_model_constraints(self, model: str) -> Dict[str, Any]:
        """Get model-specific constraints."""
        constraints = {
            "sora-2": {
                "durations": [4, 8, 12],
                "sizes": ["1280x720", "720x1280", "1024x1024"],  # Standard resolutions
                "description": "OpenAI Sora-2 - High-quality video generation"
            },
            "sora-2-pro": {
                "durations": [4, 8, 12],
                "sizes": [
                    "1280x720", "720x1280", "1024x1024",  # Standard
                    "1920x1080", "1080x1920",  # HD
                    "1440x1080", "1080x1440",  # Custom aspect ratios
                    "1536x864", "864x1536"     # Pro-specific resolutions
                ],
                "description": "OpenAI Sora-2-Pro - Enhanced model with more resolution options"
            }
        }
        
        if model not in constraints:
            raise ValueError(f"Unknown Sora model: {model}. Available: {list(constraints.keys())}")
        
        return constraints[model]
    
    async def generate_video(
        self,
        prompt: str,
        image_path: Union[str, Path],
        duration: int = 8,
        size: str = "1280x720",
        output_path: Optional[Path] = None,
        auto_pad: bool = True
    ) -> Dict[str, Any]:
        """
        Generate video from text prompt and image.
        
        Args:
            prompt: Text description for video generation
            image_path: Path to input image file
            duration: Video duration in seconds (4, 8, or 12)
            size: Video resolution (e.g., "1280x720")
            output_path: Optional path to save video
            auto_pad: If True, automatically pad image to match target resolution exactly
            
        Returns:
            Dictionary with generation results
        """
        # Validate duration
        if duration not in self.model_constraints["durations"]:
            valid_durations = self.model_constraints["durations"]
            logger.warning(f"Duration {duration}s not supported for {self.model}. Using {valid_durations[1]}s")
            duration = valid_durations[1]  # Default to middle value (8s)
        
        # Validate size
        if size not in self.model_constraints["sizes"]:
            valid_sizes = self.model_constraints["sizes"]
            logger.warning(f"Size {size} not supported for {self.model}. Using {valid_sizes[0]}")
            size = valid_sizes[0]
        
        # Process image to match target resolution
        processed_image_path = await self._process_image_for_resolution(image_path, size, auto_pad)
        
        # Submit video generation job
        video_id = await self._create_video_job(prompt, processed_image_path, duration, size)
        logger.info(f"Sora video generation started. Job ID: {video_id}")
        
        # Poll for completion
        job_result = await self._poll_video_job(video_id)
        
        result = {
            "video_id": video_id,
            "model": self.model,
            "prompt": prompt,
            "image_path": str(image_path),
            "duration": duration,
            "size": size,
            "status": job_result.get("status")
        }
        
        # Download video if job completed successfully
        if job_result.get("status") in ["completed", "succeeded"] and output_path:
            saved_path = await self._download_video(video_id, output_path)
            result["video_path"] = str(saved_path)
            logger.info(f"Video saved to: {saved_path}")
        
        return result
    
    async def _process_image_for_resolution(
        self, 
        image_path: Union[str, Path], 
        target_size: str, 
        auto_pad: bool
    ) -> str:
        """
        Process image to match target resolution.
        
        Args:
            image_path: Path to input image
            target_size: Target resolution (e.g., "1280x720")
            auto_pad: If True, pad image to match resolution; if False, validate exact match
            
        Returns:
            Path to processed image (same as input if no processing needed)
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Parse target resolution
        target_w, target_h = map(int, target_size.split("x"))
        
        # Check current image resolution
        with Image.open(path) as img:
            current_w, current_h = img.size
        
        # If already matches exactly, no processing needed
        if (current_w, current_h) == (target_w, target_h):
            logger.debug(f"Image resolution {current_w}x{current_h} already matches target {target_size}")
            return str(image_path)
        
        if not auto_pad:
            # Strict validation mode - reject mismatched dimensions
            raise ValueError(
                f"Image resolution {current_w}x{current_h} does not match target size {target_size}. "
                "Sora requires exact resolution matching. Set auto_pad=True to enable automatic padding."
            )
        
        # Auto-pad mode - create padded version
        logger.info(f"Auto-padding image from {current_w}x{current_h} to {target_size}")
        padded_image = self._pad_image_to_exact_resolution(image_path, target_size)
        
        # Save padded image to temporary file
        temp_path = path.parent / f"{path.stem}_padded_{target_size.replace('x', '_')}{path.suffix}"
        padded_image.save(temp_path)
        logger.debug(f"Saved padded image to: {temp_path}")
        
        return str(temp_path)
    
    def _pad_image_to_exact_resolution(self, image_path: Union[str, Path], target_size: str) -> Image.Image:
        """
        Pad an image with white margins to match exact target resolution.
        
        Args:
            image_path: Path to input image
            target_size: Target resolution (e.g., "1280x720")
            
        Returns:
            PIL Image padded to exact target resolution
        """
        # Parse target resolution
        target_w, target_h = map(int, target_size.split("x"))
        
        # Load and convert image
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")
            
            current_w, current_h = img.size
            
            # If already correct size, return as-is
            if (current_w, current_h) == (target_w, target_h):
                return img.copy()
            
            # Create white canvas with exact target dimensions
            padded = Image.new("RGB", (target_w, target_h), color="white")
            
            # Calculate scaling to fit image inside target while maintaining aspect ratio
            scale_w = target_w / current_w
            scale_h = target_h / current_h
            scale = min(scale_w, scale_h)
            
            # Resize image to fit within target dimensions
            new_w = int(current_w * scale)
            new_h = int(current_h * scale)
            resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Calculate position to center the resized image
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            
            # Paste resized image onto white canvas
            padded.paste(resized_img, (x_offset, y_offset))
            
            logger.info(f"Padded and scaled image from {current_w}×{current_h} to {target_w}×{target_h}")
            return padded
    
    def _validate_image_resolution(self, image_path: Union[str, Path], target_size: str) -> None:
        """Validate that image resolution exactly matches target video size."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Parse target resolution
        target_w, target_h = map(int, target_size.split("x"))
        
        # Get actual image resolution
        with Image.open(path) as img:
            actual_w, actual_h = img.size
        
        if (actual_w, actual_h) != (target_w, target_h):
            raise ValueError(
                f"Image resolution {actual_w}x{actual_h} does not match target size {target_size}. "
                "Sora requires exact resolution matching."
            )
        
        logger.debug(f"Image resolution {actual_w}x{actual_h} matches target size {target_size}")
    
    async def _create_video_job(
        self,
        prompt: str,
        image_path: Union[str, Path],
        duration: int,
        size: str
    ) -> str:
        """Create a video generation job using OpenAI Videos API."""
        import httpx
        import mimetypes
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # Prepare multipart form data with filename and MIME type
        path = Path(image_path)
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        data = {
            "model": self.model,
            "prompt": prompt,
            "size": size,
            "seconds": str(duration),
        }
        
        async with httpx.AsyncClient(timeout=300.0) as client:
            with open(path, "rb") as f:
                files = {"input_reference": (path.name, f, mime)}
                response = await client.post(
                    f"{self.base_url}/videos",
                    headers=headers,
                    files=files,
                    data=data,
                )
            
            if response.status_code != 200:
                error_msg = f"Failed to create video job: {response.status_code} {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            job = response.json()
            return job["id"]
    
    async def _poll_video_job(self, video_id: str, max_wait_time: int = 600) -> Dict[str, Any]:
        """Poll video generation job until completion."""
        import httpx
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        terminal_statuses = {"completed", "succeeded", "failed", "cancelled", "rejected"}
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=60.0) as client:
            while time.time() - start_time < max_wait_time:
                try:
                    response = await client.get(
                        f"{self.base_url}/videos/{video_id}",
                        headers=headers
                    )
                    
                    if response.status_code != 200:
                        logger.warning(f"Poll request failed: {response.status_code}")
                        await asyncio.sleep(5)
                        continue
                    
                    job = response.json()
                    status = job.get("status", "unknown")
                    progress = job.get("progress")
                    
                    if progress is not None:
                        logger.info(f"Status: {status} | Progress: {progress}%")
                    else:
                        logger.info(f"Status: {status}")
                    
                    if status in terminal_statuses:
                        if status not in {"completed", "succeeded"}:
                            raise Exception(f"Video generation failed or was cancelled: {job}")
                        return job
                    
                    await asyncio.sleep(2)
                    
                except httpx.TimeoutException:
                    logger.warning("Poll request timed out, retrying...")
                    await asyncio.sleep(5)
        
        raise TimeoutError(f"Video generation timed out after {max_wait_time} seconds")
    
    async def _download_video(self, video_id: str, output_path: Path) -> Path:
        """Download completed video from OpenAI."""
        import httpx
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            async with client.stream(
                "GET",
                f"{self.base_url}/videos/{video_id}/content",
                headers=headers
            ) as response:
                if response.status_code != 200:
                    raise Exception(f"Failed to download video: {response.status_code}")
                
                with open(output_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        
        return output_path
