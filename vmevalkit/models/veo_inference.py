"""
Google Veo (Gemini API) — Text + Image → Video
- Auth: uses GEMINI_API_KEY environment variable
- Models: veo-2.0-generate-001, veo-3.0-generate-preview, veo-3.0-fast-generate-preview, veo-3.1-generate-001
- Input image: local file (PNG/JPEG)
- Output: saves video file directly using the Gemini client
"""

from __future__ import annotations

import os
import base64
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

from google import genai
from google.genai import types
from PIL import Image
import io
from .base import ModelWrapper

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# Attempt to load environment variables from a .env file if present
try:
    from dotenv import load_dotenv, find_dotenv
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)
    else:
        load_dotenv(override=False)
except Exception:
    pass


def _hydrate_env_from_nearby_dotenv() -> None:
    """Best-effort manual .env loader if python-dotenv did not populate env."""
    candidate_paths = []
    try:
        candidate_paths.append(Path.cwd() / ".env")
    except Exception:
        pass
    try:
        here = Path(__file__).resolve()
        for i in range(1, 6):
            candidate_paths.append(here.parents[i] / ".env")
    except Exception:
        pass

    for env_path in candidate_paths:
        try:
            if env_path.exists():
                for raw_line in env_path.read_text().splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
                break
        except Exception:
            continue


def get_gemini_api_key() -> str:
    """Get GEMINI_API_KEY from environment."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        _hydrate_env_from_nearby_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. "
            "Set it in your environment or add to .env file."
        )
    return api_key


# Model name mapping from catalog names to Gemini API model IDs
MODEL_NAME_MAPPING = {
    "veo-2.0-generate-001": "veo-2.0-generate-001",
    "veo-3.0-generate-preview": "veo-3.0-generate-preview",
    "veo-3.0-fast-generate-preview": "veo-3.0-fast-generate-preview",
    "veo-3.1-generate-001": "veo-3.1-generate-001",
}


class VeoService:
    """
    Google Veo video generation client using the google-genai library.
    """

    def __init__(
        self,
        model_id: str = "veo-3.0-generate-preview",
        poll_interval_s: float = 5.0,
        poll_timeout_s: float = 1800.0,
    ):
        """
        Initialize the Veo service.
        
        Args:
            model_id: Model ID to use for generation
            poll_interval_s: Seconds between polling attempts
            poll_timeout_s: Maximum time to wait for generation
        """
        self.model_id = MODEL_NAME_MAPPING.get(model_id, model_id)
        self.poll_interval_s = poll_interval_s
        self.poll_timeout_s = poll_timeout_s
        
        # Initialize client
        api_key = get_gemini_api_key()
        self.client = genai.Client(api_key=api_key)
        
        logger.info(f"VeoService initialized with model: {self.model_id}")

    def _determine_best_aspect_ratio(
        self, 
        image_width: int, 
        image_height: int, 
        preferred_ratio: Optional[str] = None
    ) -> str:
        """
        Determine the best aspect ratio for Veo based on input image dimensions.
        
        Args:
            image_width: Original image width
            image_height: Original image height  
            preferred_ratio: User-specified preferred ratio ("16:9" or "9:16")
            
        Returns:
            Best aspect ratio ("16:9" or "9:16")
        """
        if preferred_ratio and preferred_ratio in ("16:9", "9:16"):
            return preferred_ratio
            
        input_ratio = image_width / image_height
        
        # Default to landscape (16:9) for square images
        if 0.9 <= input_ratio <= 1.1:
            best_ratio = "16:9"
            logger.info(f"Square image detected ({image_width}×{image_height}) -> defaulting to landscape {best_ratio}")
            return best_ratio
        
        # For non-square images, choose based on orientation
        if input_ratio > 1:
            best_ratio = "16:9"
        else:
            best_ratio = "9:16"
        
        logger.info(f"Input aspect ratio {input_ratio:.3f} ({image_width}×{image_height}) -> Best Veo match: {best_ratio}")
        return best_ratio

    def _pad_image_to_aspect_ratio(self, image: Image.Image, target_aspect_ratio: str) -> Image.Image:
        """
        Pad an image with white margins to match the target aspect ratio.
        
        Args:
            image: PIL Image to pad
            target_aspect_ratio: "16:9" or "9:16"
            
        Returns:
            PIL Image with white padding to match target aspect ratio
        """
        if target_aspect_ratio == "16:9":
            target_ratio = 16 / 9
        elif target_aspect_ratio == "9:16":
            target_ratio = 9 / 16
        else:
            raise ValueError(f"Unsupported aspect ratio: {target_aspect_ratio}")
        
        width, height = image.size
        current_ratio = width / height
        
        if abs(current_ratio - target_ratio) < 0.01:
            return image
        
        if current_ratio > target_ratio:
            new_width = width
            new_height = int(width / target_ratio)
        else:
            new_height = height
            new_width = int(height * target_ratio)
        
        padded = Image.new("RGB", (new_width, new_height), color="white")
        x_offset = (new_width - width) // 2
        y_offset = (new_height - height) // 2
        padded.paste(image, (x_offset, y_offset))
        
        logger.info(f"Padded image from {width}×{height} to {new_width}×{new_height} for {target_aspect_ratio} ratio")
        return padded

    def _prepare_image(
        self,
        image_path: Optional[str],
        aspect_ratio: str = "16:9"
    ) -> Optional[types.Image]:
        """
        Prepare an image for the Veo API.
        
        Args:
            image_path: Path to the input image
            aspect_ratio: Target aspect ratio for padding
            
        Returns:
            Prepared Image object or None
        """
        if not image_path:
            return None
            
        p = Path(image_path)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(p)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Pad image to match video aspect ratio
        padded_image = self._pad_image_to_aspect_ratio(image, aspect_ratio)
        
        # Convert to bytes
        buffer = io.BytesIO()
        padded_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
        # Create Gemini Image object
        return types.Image(image_bytes=image_bytes, mime_type="image/png")

    async def generate_video(
        self,
        prompt: str,
        *,
        image_path: Optional[str] = None,
        duration_seconds: int = 8,
        aspect_ratio: str = "16:9",
        negative_prompt: Optional[str] = None,
        generate_audio: bool = True,
        seed: Optional[int] = None,
        person_generation: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> Tuple[Optional[bytes], Dict[str, Any]]:
        """
        Generate a video using the Veo model.
        
        Args:
            prompt: Text prompt describing the desired video
            image_path: Optional path to input image for image-to-video
            duration_seconds: Duration of the video (4, 6, or 8 seconds)
            aspect_ratio: Aspect ratio ("16:9" or "9:16")
            negative_prompt: Optional negative prompt
            generate_audio: Whether to generate audio
            seed: Optional random seed for reproducibility
            person_generation: Person generation setting ("disallow" or "allow_adult")
            output_path: Optional path to save the video
            
        Returns:
            Tuple of (video_bytes, metadata_dict)
        """
        # Auto-detect aspect ratio from image if using default
        if image_path and aspect_ratio == "16:9":
            with Image.open(image_path) as img:
                aspect_ratio = self._determine_best_aspect_ratio(img.width, img.height, aspect_ratio)
        
        # Validate parameters
        if duration_seconds not in (4, 6, 8):
            logger.warning(f"Veo supports durations 4, 6, or 8 seconds; got {duration_seconds}. Using 8.")
            duration_seconds = 8
        if aspect_ratio not in ("16:9", "9:16"):
            raise ValueError('aspect_ratio must be "16:9" or "9:16"')
        
        # Prepare image if provided
        image = self._prepare_image(image_path, aspect_ratio)
        
        # Build generation config
        config_kwargs: Dict[str, Any] = {
            "aspect_ratio": aspect_ratio,
        }
        
        if negative_prompt:
            config_kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            config_kwargs["seed"] = seed
        if person_generation:
            config_kwargs["person_generation"] = person_generation
            
        config = types.GenerateVideosConfig(**config_kwargs)
        
        # Start video generation
        logger.info(f"Starting Veo video generation with model: {self.model_id}")
        logger.info(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
        
        # Build the request - include image if provided
        generate_kwargs: Dict[str, Any] = {
            "model": self.model_id,
            "prompt": prompt,
            "config": config,
        }
        if image:
            generate_kwargs["image"] = image
            
        operation = self.client.models.generate_videos(**generate_kwargs)
        
        logger.info(f"Started operation: {operation.name}")
        
        # Poll for completion
        start_time = time.time()
        max_attempts = int(self.poll_timeout_s / self.poll_interval_s)
        attempt = 0
        
        operation_name = operation.name
        while not operation.done:
            if attempt >= max_attempts:
                raise TimeoutError(f"Video generation timed out after {self.poll_timeout_s}s")
            
            elapsed = time.time() - start_time
            logger.info(f"Still running... attempt {attempt + 1}, elapsed: {elapsed:.1f}s")
            await asyncio.sleep(self.poll_interval_s)
            operation = self.client.operations.get(operation_name)
            attempt += 1
        
        elapsed_total = time.time() - start_time
        logger.info(f"Operation completed in {elapsed_total:.1f}s")
        
        # Extract video from response
        metadata: Dict[str, Any] = {
            "operation_name": operation.name,
            "elapsed_seconds": elapsed_total,
            "model": self.model_id,
            "prompt": prompt,
        }
        
        if not operation.response or not operation.response.generated_videos:
            logger.warning("Operation completed but no videos found in response.")
            return None, metadata
        
        video = operation.response.generated_videos[0]
        
        # Download the video
        self.client.files.download(file=video.video)
        
        # Save to file if output path provided
        if output_path:
            video.video.save(output_path)
            logger.info(f"Video saved to {output_path}")
            metadata["output_path"] = output_path
            
            # Read the saved file to return bytes
            video_bytes = Path(output_path).read_bytes()
        else:
            # Save to temp location and read bytes
            temp_path = Path(f"/tmp/veo_video_{int(time.time())}.mp4")
            video.video.save(str(temp_path))
            video_bytes = temp_path.read_bytes()
            temp_path.unlink()  # Clean up temp file
        
        logger.info(f"Video bytes received: {len(video_bytes)} bytes")
        return video_bytes, metadata

    async def save_video(self, video_bytes: bytes, output_path: Path) -> Path:
        """Save video bytes to a file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(video_bytes)
        logger.info(f"Video saved to {output_path}")
        return output_path


class VeoWrapper(ModelWrapper):
    """
    VMEvalKit wrapper for VeoService to match the standard interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """Initialize Veo wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create VeoService instance
        self.veo_service = VeoService(model_id=model, **kwargs)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using Veo (synchronous wrapper).
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (4, 6, or 8 for Veo)
            output_filename: Optional output filename
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results
        """
        start_time = time.time()
        
        # Convert duration to int (Veo requires int)
        duration_seconds = int(duration)
        
        # Filter kwargs to only include supported parameters
        veo_kwargs: Dict[str, Any] = {}
        allowed_params = {
            'aspect_ratio', 'negative_prompt', 'generate_audio',
            'seed', 'person_generation'
        }
        for key, value in kwargs.items():
            if key in allowed_params:
                veo_kwargs[key] = value
        
        # Determine output path
        if not output_filename:
            output_filename = f"veo_{int(time.time())}.mp4"
        output_path = self.output_dir / output_filename
        
        # Run async generation in sync context
        video_bytes, metadata = asyncio.run(
            self.veo_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                duration_seconds=duration_seconds,
                output_path=str(output_path),
                **veo_kwargs
            )
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "success": video_bytes is not None,
            "video_path": str(output_path) if video_bytes else None,
            "error": None,
            "duration_seconds": duration_taken,
            "generation_id": metadata.get('operation_name', 'unknown'),
            "model": self.model,
            "status": "success" if video_bytes else "failed",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "veo_metadata": metadata
            }
        }
