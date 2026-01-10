"""
Runway ML Image-to-Video Generation Service.

Supports text + image → video generation using Runway's Gen-3 and Gen-4 models.
"""

import os
import time
import asyncio
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
import io
from PIL import Image
from dotenv import load_dotenv
from .base import ModelWrapper

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class RunwayService:
    """Service for image-to-video generation using Runway ML models."""
    
    def __init__(self, model: str = "gen4_turbo"):
        """
        Initialize Runway service.
        
        Args:
            model: Runway model to use (gen4_turbo, gen4_aleph, gen3a_turbo)
        """
        self.api_secret = os.getenv("RUNWAYML_API_SECRET")
        if not self.api_secret:
            raise ValueError("RUNWAYML_API_SECRET environment variable is required")
        
        self.model = model
        
        # Validate model and set constraints
        self.model_constraints = self._get_model_constraints(model)
    
    def _get_model_constraints(self, model: str) -> Dict[str, Any]:
        """Get model-specific constraints."""
        constraints = {
            "gen4_turbo": {
                "durations": [5, 10],
                # Use actual pixel dimensions required by Runway API
                "ratios": ["1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"],
                "description": "Runway Gen-4 Turbo - Fast high-quality generation"
            },
            "gen4_aleph": {
                "durations": [5],
                # Use actual pixel dimensions required by Runway API
                "ratios": ["1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"],
                "description": "Runway Gen-4 Aleph - Premium quality"
            },
            "gen3a_turbo": {
                "durations": [5, 10],
                "ratios": ["1280:768", "768:1280"],  # Gen-3 specific pixel dimensions
                "description": "Runway Gen-3A Turbo - Proven performance"
            }
        }
        
        if model not in constraints:
            raise ValueError(f"Unknown Runway model: {model}. Available: {list(constraints.keys())}")
        
        return constraints[model]
    
    def _determine_best_aspect_ratio(self, image_width: int, image_height: int) -> str:
        """
        Determine the best aspect ratio match from supported ratios.
        
        Args:
            image_width: Original image width
            image_height: Original image height
            
        Returns:
            Best matching aspect ratio string (e.g., "960:960", "1280:720")
        """
        input_ratio = image_width / image_height
        supported_ratios = self.model_constraints["ratios"]
        
        # Check for square images and use 960:960 if available
        if 0.9 <= input_ratio <= 1.1:
            # Look for square format in supported ratios
            for ratio_str in supported_ratios:
                if ':' in ratio_str:
                    parts = ratio_str.split(':')
                    w, h = map(int, parts)
                    if w == h:  # Square format found
                        logger.info(f"Square image detected ({image_width}×{image_height}) -> using {ratio_str}")
                        return ratio_str
        
        best_ratio = None
        min_diff = float('inf')
        
        for ratio_str in supported_ratios:
            # Skip 1:1 if we already handled it above
            if ratio_str == "1:1":
                continue
                
            # Handle both aspect ratio strings (e.g., "16:9") and pixel dimensions (e.g., "1280:768")
            if ':' in ratio_str:
                parts = ratio_str.split(':')
                w, h = map(float, parts)  # Use float to handle both "16:9" and "1280:768"
                ratio = w / h
            else:
                # Fallback for any other format
                continue
            
            diff = abs(input_ratio - ratio)
            
            if diff < min_diff:
                min_diff = diff
                best_ratio = ratio_str
        
        logger.info(f"Input aspect ratio {input_ratio:.3f} ({image_width}×{image_height}) -> Best match: {best_ratio}")
        return best_ratio

    def _resize_and_pad_image(self, image_path: Union[str, Path], target_ratio: str) -> Path:
        """
        Resize and pad image to match target dimensions exactly.
        
        Args:
            image_path: Path to input image
            target_ratio: Target aspect ratio (e.g., "16:9" or "1280:768")
            
        Returns:
            Path to processed image file
        """
        # Parse target dimensions - handle both aspect ratios and pixel dimensions
        if ':' in target_ratio:
            parts = target_ratio.split(':')
            if '.' not in parts[0] and len(parts[0]) > 2:  # Likely pixel dimensions like "1280:768"
                target_w, target_h = map(int, parts)
            else:  # Aspect ratio like "16:9"
                # Convert aspect ratio to target dimensions (use standard 720p resolution)
                aspect_w, aspect_h = map(float, parts)
                aspect_ratio = aspect_w / aspect_h
                if aspect_ratio > 1:  # Landscape
                    target_w, target_h = 1280, int(1280 / aspect_ratio)
                else:  # Portrait
                    target_h, target_w = 1280, int(1280 * aspect_ratio)
        else:
            # Default to 720p 16:9 if format not recognized
            target_w, target_h = 1280, 720
        
        # Load and convert image
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        original_w, original_h = image.size
        logger.info(f"Original image size: {original_w}×{original_h}")
        
        # Calculate scaling to fit image within target dimensions while preserving aspect ratio
        scale_w = target_w / original_w
        scale_h = target_h / original_h
        scale = min(scale_w, scale_h)  # Use smaller scale to ensure image fits
        
        # Resize image
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)
        resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # Create canvas with exact target dimensions and white background
        padded_image = Image.new("RGB", (target_w, target_h), color="white")
        
        # Calculate position to center the resized image
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # Paste resized image onto padded canvas
        padded_image.paste(resized_image, (x_offset, y_offset))
        
        # Save processed image
        processed_path = Path(image_path).parent / f"runway_processed_{Path(image_path).name}"
        padded_image.save(processed_path, "PNG", quality=95)
        
        logger.info(f"Processed image: {original_w}×{original_h} -> {new_w}×{new_h} -> {target_w}×{target_h}")
        logger.info(f"Saved to: {processed_path}")
        
        return processed_path
    
    async def generate_video(
        self,
        prompt: str,
        image_path: Union[str, Path],
        duration: int = 5,
        ratio: Optional[str] = None,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate video from text prompt and image.
        
        Args:
            prompt: Text description for video generation
            image_path: Path to input image file (will be uploaded to get URL)
            duration: Video duration in seconds
            ratio: Video aspect ratio (if not provided, uses first available)
            output_path: Optional path to save video
            
        Returns:
            Dictionary with generation results
        """
        # Validate duration
        if duration not in self.model_constraints["durations"]:
            valid_durations = self.model_constraints["durations"]
            logger.warning(f"Duration {duration}s not supported for {self.model}. Using {valid_durations[0]}s")
            duration = valid_durations[0]
        
        # Determine best aspect ratio if not provided
        if not ratio:
            # Load image to get dimensions for aspect ratio detection
            with Image.open(image_path) as img:
                ratio = self._determine_best_aspect_ratio(img.width, img.height)
        elif ratio not in self.model_constraints["ratios"]:
            valid_ratios = self.model_constraints["ratios"]
            logger.warning(f"Ratio {ratio} not supported for {self.model}. Using {valid_ratios[0]}")
            ratio = valid_ratios[0]
        
        # Process image to match target dimensions
        processed_image_path = self._resize_and_pad_image(image_path, ratio)
        
        # Upload processed image to get URL (Runway requires image URLs)
        image_url = await self._upload_image(processed_image_path)
        
        try:
            # Generate video using Runway SDK
            result = await self._generate_with_runway(prompt, image_url, duration, ratio)
            
            # Download video if output path provided
            if output_path and result.get("video_url"):
                saved_path = await self._download_video(result["video_url"], output_path)
                result["video_path"] = str(saved_path)
                logger.info(f"Video saved to: {saved_path}")
            
            result.update({
                "model": self.model,
                "prompt": prompt,
                "image_path": str(image_path),
                "processed_image_path": str(processed_image_path),
                "duration": duration,
                "ratio": ratio
            })
            
            return result
            
        finally:
            # Clean up temporary processed image
            try:
                if processed_image_path.exists():
                    processed_image_path.unlink()
                    logger.debug(f"Cleaned up processed image: {processed_image_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up processed image: {e}")
    
    async def _upload_image(self, image_path: Union[str, Path]) -> str:
        """
        Upload image and return URL. 
        For now, this is a placeholder - in practice you'd need to upload to a CDN/S3.
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # TODO: Implement actual image upload to CDN/S3
        # For now, we'll assume the user provides a publicly accessible image URL
        # or we could integrate with the existing S3 uploader from Luma
        from ..utils.s3_uploader import S3ImageUploader
        
        try:
            s3_uploader = S3ImageUploader()
            image_url = s3_uploader.upload(str(image_path))
            if not image_url:
                raise Exception("Failed to upload image to S3")
            logger.info(f"Uploaded image to: {image_url}")
            return image_url
        except Exception as e:
            logger.error(f"Failed to upload image: {e}")
            raise Exception(f"Image upload failed: {e}")
    
    async def _generate_with_runway(
        self, 
        prompt: str, 
        image_url: str, 
        duration: int, 
        ratio: str
    ) -> Dict[str, Any]:
        """Generate video using Runway SDK."""
        try:
            from runwayml import RunwayML, TaskFailedError
        except ImportError:
            raise ImportError("runwayml package not installed. Run: pip install runwayml")
        
        # Run in thread pool since Runway SDK is synchronous
        def _sync_generate():
            client = RunwayML()
            
            try:
                task = client.image_to_video.create(
                    model=self.model,
                    prompt_image=image_url,
                    prompt_text=prompt,
                    ratio=ratio,
                    duration=duration
                ).wait_for_task_output()
                
                # Handle case where task.output is a list instead of string
                video_url = None
                if hasattr(task, 'output') and task.output:
                    if isinstance(task.output, list):
                        video_url = task.output[0] if task.output else None
                    else:
                        video_url = task.output
                
                return {
                    "task_id": task.id if hasattr(task, 'id') else 'unknown',
                    "video_url": video_url,
                    "status": "success"
                }
                
            except TaskFailedError as e:
                logger.error(f"Runway task failed: {e.task_details}")
                raise Exception(f"Runway generation failed: {e.task_details}")
            except Exception as e:
                logger.error(f"Runway SDK error: {e}")
                raise Exception(f"Runway generation error: {e}")
        
        # Run synchronous Runway call in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_generate)
    
    async def _download_video(self, video_url: str, output_path: Path) -> Path:
        """Download video from URL to local file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        import httpx
        async with httpx.AsyncClient(timeout=600.0) as client:  # 10 minute timeout for download
            response = await client.get(video_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download video: {response.status_code}")
            
            with open(output_path, "wb") as f:
                f.write(response.content)
        
        return output_path


# ========================================
# VMEVALKIT WRAPPER CLASS
# ========================================

class RunwayWrapper(ModelWrapper):
    """
    VMEvalKit wrapper for RunwayService to match the standard interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./outputs",
        **kwargs
    ):
        """Initialize Runway wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create RunwayService instance
        self.runway_service = RunwayService(model=model)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        ratio: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using Runway (synchronous wrapper).
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (5 or 10 depending on model)
            output_filename: Optional output filename
            ratio: Video aspect ratio (model-specific)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results
        """
        import time
        start_time = time.time()
        
        # Convert duration to int (Runway expects int)
        duration_int = int(duration)
        
        # Generate output path
        if not output_filename:
            output_filename = "video.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Run async generation in sync context
        # Filter kwargs - RunwayService.generate_video only accepts:
        # prompt, image_path, duration, ratio, output_path
        # All parameters are already passed explicitly, so no additional kwargs
        result = asyncio.run(
            self.runway_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                duration=duration_int,
                ratio=ratio,
                output_path=output_path
            )
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "success": bool(result.get("video_path")),
            "video_path": result.get("video_path"),
            "error": None,
            "duration_seconds": duration_taken,
            "generation_id": result.get("task_id", 'unknown'),
            "model": self.model,
            "status": "success" if result.get("video_path") else "failed",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "video_url": result.get("video_url"),
                "duration": duration_int,
                "ratio": result.get("ratio"),
                "runway_result": result
            }
        }
