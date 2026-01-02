"""
WaveSpeedAI Image-to-Video Generation Service.

Supports text + image → video generation using WaveSpeedAI's platform:
- Google Veo 3.1 models  
- WaveSpeedAI WAN 2.1 and WAN 2.2 models
"""

import os
import httpx
import json
import asyncio
import base64
from typing import Optional, Dict, Any, Union
from pathlib import Path
import logging
from enum import Enum
import io
from PIL import Image
from dotenv import load_dotenv
from .base import ModelWrapper

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()


class WaveSpeedModel(str, Enum):
    """WaveSpeedAI model endpoints for I2V generation."""
    
    # Google Veo Models
    VEO_3_1_I2V = "veo3.1/image-to-video"  # Veo 3.1 standard image-to-video endpoint
    VEO_3_1_FAST_I2V = "veo3.1-fast/image-to-video"  # Veo 3.1 fast image-to-video endpoint
    
    # WAN 2.2 Models
    WAN_2_2_I2V_480P = "wan-2.2/i2v-480p"
    WAN_2_2_I2V_480P_ULTRA_FAST = "wan-2.2/i2v-480p-ultra-fast"
    WAN_2_2_I2V_480P_LORA = "wan-2.2/i2v-480p-lora"
    WAN_2_2_I2V_480P_LORA_ULTRA_FAST = "wan-2.2/i2v-480p-lora-ultra-fast"
    WAN_2_2_I2V_5B_720P = "wan-2.2/i2v-5b-720p"
    WAN_2_2_I2V_5B_720P_LORA = "wan-2.2/i2v-5b-720p-lora"
    WAN_2_2_I2V_720P = "wan-2.2/i2v-720p"
    WAN_2_2_I2V_720P_ULTRA_FAST = "wan-2.2/i2v-720p-ultra-fast"
    WAN_2_2_I2V_720P_LORA = "wan-2.2/i2v-720p-lora"
    WAN_2_2_I2V_720P_LORA_ULTRA_FAST = "wan-2.2/i2v-720p-lora-ultra-fast"
    
    # WAN 2.1 Models
    WAN_2_1_I2V_480P = "wan-2.1/i2v-480p"
    WAN_2_1_I2V_480P_ULTRA_FAST = "wan-2.1/i2v-480p-ultra-fast"
    WAN_2_1_I2V_480P_LORA = "wan-2.1/i2v-480p-lora"
    WAN_2_1_I2V_480P_LORA_ULTRA_FAST = "wan-2.1/i2v-480p-lora-ultra-fast"
    WAN_2_1_I2V_720P = "wan-2.1/i2v-720p"
    WAN_2_1_I2V_720P_ULTRA_FAST = "wan-2.1/i2v-720p-ultra-fast"
    WAN_2_1_I2V_720P_LORA = "wan-2.1/i2v-720p-lora"
    WAN_2_1_I2V_720P_LORA_ULTRA_FAST = "wan-2.1/i2v-720p-lora-ultra-fast"
    
# Keep backward compatibility
WANModel = WaveSpeedModel


class WaveSpeedService:
    """Service for image-to-video generation using WaveSpeedAI models (WAN and Veo)."""
    
    def __init__(self, model: str = WaveSpeedModel.WAN_2_2_I2V_720P):
        """
        Initialize WaveSpeed service.
        
        Args:
            model: Model variant to use (WAN or Veo)
        """
        self.api_key = os.getenv("WAVESPEED_API_KEY")
        if not self.api_key:
            raise ValueError("WAVESPEED_API_KEY environment variable is required")
        
        self.model = model
        self.base_url = "https://api.wavespeed.ai"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def _is_veo_model(self) -> bool:
        """Check if the current model is a VEO 3.1 model."""
        veo_models = [WaveSpeedModel.VEO_3_1_I2V, WaveSpeedModel.VEO_3_1_FAST_I2V]
        veo_values = [WaveSpeedModel.VEO_3_1_I2V.value, WaveSpeedModel.VEO_3_1_FAST_I2V.value]
        return self.model in veo_models or self.model in veo_values

    def _determine_best_veo_aspect_ratio(self, image_width: int, image_height: int, preferred_ratio: Optional[str] = None) -> str:
        """
        Determine the best aspect ratio for VEO 3.1 based on input image dimensions.
        
        Args:
            image_width: Original image width
            image_height: Original image height  
            preferred_ratio: User-specified preferred ratio
            
        Returns:
            Best aspect ratio string ("16:9" or "9:16" only)
        """
        # VEO 3.1 via WaveSpeed API only supports these two aspect ratios
        supported_ratios = {
            "16:9": 16/9,   # ~1.78 (landscape)
            "9:16": 9/16    # ~0.56 (portrait)
        }
        
        # If user specified a ratio and it's supported, use it
        if preferred_ratio and preferred_ratio in supported_ratios:
            return preferred_ratio
            
        input_ratio = image_width / image_height
        
        # Special handling for square images (1:1)
        # Default to landscape (16:9) for square images since it's more common
        if 0.9 <= input_ratio <= 1.1:  # Close to square
            best_ratio = "16:9"
            logger.info(f"Square image detected ({image_width}×{image_height}) -> defaulting to landscape {best_ratio}")
            return best_ratio
        
        # For non-square images, find the closest matching aspect ratio
        # If input is wider than tall (>1), use landscape; otherwise use portrait
        if input_ratio > 1:
            best_ratio = "16:9"  # Landscape
        else:
            best_ratio = "9:16"  # Portrait
        
        logger.info(f"Input aspect ratio {input_ratio:.3f} ({image_width}×{image_height}) -> Best VEO match: {best_ratio}")
        return best_ratio

    def _pad_image_for_veo(self, image: Image.Image, target_aspect_ratio: str) -> Image.Image:
        """
        Pad an image with white margins to match VEO's target aspect ratio.
        
        Args:
            image: PIL Image to pad
            target_aspect_ratio: Target aspect ratio ("16:9" or "9:16")
            
        Returns:
            PIL Image with white padding to match target aspect ratio
        """
        # Parse target aspect ratio - VEO 3.1 via WaveSpeed only supports these
        aspect_ratios = {
            "16:9": 16/9,
            "9:16": 9/16
        }
        
        if target_aspect_ratio not in aspect_ratios:
            raise ValueError(f"Unsupported VEO aspect ratio: {target_aspect_ratio}")
        
        target_ratio = aspect_ratios[target_aspect_ratio]
        width, height = image.size
        current_ratio = width / height
        
        # If already correct ratio, return as-is
        if abs(current_ratio - target_ratio) < 0.01:
            return image
        
        # Calculate target dimensions  
        if current_ratio > target_ratio:
            # Image is wider than target - add height padding
            new_width = width
            new_height = int(width / target_ratio)
        else:
            # Image is taller than target - add width padding
            new_height = height
            new_width = int(height * target_ratio)
        
        # Create white canvas with target dimensions
        padded = Image.new("RGB", (new_width, new_height), color="white")
        
        # Calculate position to center the original image
        x_offset = (new_width - width) // 2
        y_offset = (new_height - height) // 2
        
        # Paste original image onto white canvas
        padded.paste(image, (x_offset, y_offset))
        
        logger.info(f"VEO padding: {width}×{height} -> {new_width}×{new_height} for {target_aspect_ratio} ratio")
        return padded

    async def generate_video(
        self,
        prompt: str,
        image_path: Union[str, Path],
        seed: int = -1,
        output_path: Optional[Path] = None,
        poll_timeout_s: float = 1800.0,  # 30 minute timeout
        poll_interval_s: float = 2.0,
        # Veo 3.1 specific parameters
        aspect_ratio: Optional[str] = None,
        duration: Optional[float] = None,
        resolution: Optional[str] = None,
        generate_audio: Optional[bool] = None,
        negative_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate video from text prompt and image.
        
        Args:
            prompt: Text description for video generation
            image_path: Path to input image file
            seed: Random seed for reproducibility (-1 for random)
            output_path: Optional path to save video (if not provided, returns URL)
            poll_timeout_s: Maximum time to wait for completion
            poll_interval_s: Time between polling attempts
            aspect_ratio: Video aspect ratio (Veo 3.1 only)
            duration: Video duration in seconds (Veo 3.1 only, up to 8s)
            resolution: Output resolution e.g. "1080p", "720p" (Veo 3.1 only)
            generate_audio: Whether to generate audio (Veo 3.1 only)
            negative_prompt: Negative prompt to avoid certain content (Veo 3.1 only)
            
        Returns:
            Dictionary with generation results including video URL/path
        """
        # Auto-detect best aspect ratio for VEO models if not specified
        if self._is_veo_model() and not aspect_ratio:
            with Image.open(image_path) as img:
                aspect_ratio = self._determine_best_veo_aspect_ratio(img.width, img.height)
                logger.info(f"Auto-detected aspect ratio for VEO: {aspect_ratio}")

        # Encode image to base64 (with VEO padding if needed)
        image_b64 = self._encode_image(image_path, aspect_ratio)
        
        # Submit generation request
        request_id = await self._submit_generation(
            prompt=prompt,
            image_b64=image_b64,
            seed=seed,
            aspect_ratio=aspect_ratio,
            duration=duration,
            resolution=resolution,
            generate_audio=generate_audio,
            negative_prompt=negative_prompt
        )
        logger.info(f"WaveSpeed generation started. Request ID: {request_id}")
        
        # Poll for completion
        result_url = await self._poll_generation(
            request_id, 
            poll_timeout_s, 
            poll_interval_s
        )
        
        result = {
            "video_url": result_url,
            "request_id": request_id,
            "model": self.model,
            "prompt": prompt,
            "image_path": str(image_path),
            "seed": seed
        }
        
        # Add Veo 3.1 parameters to result if used
        # Check both enum and string value forms
        veo_models = [WaveSpeedModel.VEO_3_1_I2V, WaveSpeedModel.VEO_3_1_FAST_I2V]
        veo_values = [WaveSpeedModel.VEO_3_1_I2V.value, WaveSpeedModel.VEO_3_1_FAST_I2V.value]
        
        if self.model in veo_models or self.model in veo_values:
            if aspect_ratio: result["aspect_ratio"] = aspect_ratio
            if duration: result["duration"] = duration
            if resolution: result["resolution"] = resolution
            if generate_audio is not None: result["generate_audio"] = generate_audio
            if negative_prompt: result["negative_prompt"] = negative_prompt
        
        # Download video if output path provided
        if output_path and result_url:
            saved_path = await self._download_video(result_url, output_path)
            result["video_path"] = str(saved_path)
            logger.info(f"Video saved to: {saved_path}")
        
        return result
    
    def _encode_image(self, image_path: Union[str, Path], aspect_ratio: Optional[str] = None) -> str:
        """
        Encode image file to base64, with optional VEO padding.
        
        Args:
            image_path: Path to input image
            aspect_ratio: Target aspect ratio for VEO models
            
        Returns:
            Base64 encoded image string
        """
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # For VEO models, process image with padding if needed
        if self._is_veo_model() and aspect_ratio:
            # Load and process image
            image = Image.open(path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            
            # Pad image to match target aspect ratio
            padded_image = self._pad_image_for_veo(image, aspect_ratio)
            
            # Convert processed image to bytes
            buffer = io.BytesIO()
            padded_image.save(buffer, format="PNG", quality=95)
            image_data = buffer.getvalue()
            
            logger.debug(f"Processed VEO image for {aspect_ratio} aspect ratio")
        else:
            # For WAN models or when no aspect ratio specified, use original image
            with open(path, "rb") as f:
                image_data = f.read()
        
        return base64.b64encode(image_data).decode("utf-8")
    
    async def _submit_generation(
        self,
        prompt: str,
        image_b64: str,
        seed: int,
        aspect_ratio: Optional[str] = None,
        duration: Optional[float] = None,
        resolution: Optional[str] = None,
        generate_audio: Optional[bool] = None,
        negative_prompt: Optional[str] = None
    ) -> str:
        """Submit I2V generation request."""
        # Build endpoint URL based on model type
        # Handle both enum and string model inputs
        model_value = self.model.value if hasattr(self.model, 'value') else self.model
        
        # Check if this is a Veo 3.1 model (enum or string value)
        veo_models = [WaveSpeedModel.VEO_3_1_I2V, WaveSpeedModel.VEO_3_1_FAST_I2V]
        veo_values = [WaveSpeedModel.VEO_3_1_I2V.value, WaveSpeedModel.VEO_3_1_FAST_I2V.value]
        
        if self.model in veo_models or self.model in veo_values:
            # For Veo 3.1 models, use the Google prefix path
            submit_url = f"{self.base_url}/api/v3/google/{model_value}"
        else:
            # For WAN models, use wavespeed-ai prefix
            submit_url = f"{self.base_url}/api/v3/wavespeed-ai/{model_value}"
        
        payload = {
            "prompt": prompt,
            "image": image_b64,
            "seed": seed
        }
        
        # Add Veo 3.1 specific parameters if using Veo models
        # Check both enum and string value forms
        veo_models = [WaveSpeedModel.VEO_3_1_I2V, WaveSpeedModel.VEO_3_1_FAST_I2V]
        veo_values = [WaveSpeedModel.VEO_3_1_I2V.value, WaveSpeedModel.VEO_3_1_FAST_I2V.value]
        
        if self.model in veo_models or self.model in veo_values:
            if aspect_ratio:
                payload["aspect_ratio"] = aspect_ratio
            if duration:
                payload["duration"] = duration
            if resolution:
                payload["resolution"] = resolution
            if generate_audio is not None:
                payload["generate_audio"] = generate_audio
            if negative_prompt:
                payload["negative_prompt"] = negative_prompt
        
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for download  # 2 minute timeout
            response = await client.post(
                submit_url,
                headers=self.headers,
                json=payload
            )
            
            if response.status_code != 200:
                error_msg = f"Failed to submit generation: {response.status_code} {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
            result_data = response.json()
            
            # Extract request ID from response
            if "data" in result_data:
                data = result_data["data"]
                if "id" in data:
                    return data["id"]
            
            # Fallback: try direct access
            if "id" in result_data:
                return result_data["id"]
                
            raise Exception(f"Invalid response format - no request ID found: {result_data}")
    
    async def _poll_generation(
        self, 
        request_id: str, 
        timeout_s: float, 
        interval_s: float
    ) -> str:
        """Poll for generation completion."""
        poll_url = f"{self.base_url}/api/v3/predictions/{request_id}/result"
        poll_headers = {"Authorization": f"Bearer {self.api_key}"}
        
        start_time = asyncio.get_event_loop().time()
        max_attempts = int(timeout_s / interval_s)
        
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for download  # 2 minute timeout
            for attempt in range(max_attempts):
                if asyncio.get_event_loop().time() - start_time > timeout_s:
                    raise TimeoutError(f"Generation timed out after {timeout_s}s")
                
                try:
                    response = await client.get(poll_url, headers=poll_headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Handle data wrapper
                        if "data" in data:
                            data = data["data"]
                        
                        status = data.get("status", "unknown")
                        
                        if status == "completed":
                            outputs = data.get("outputs")
                            if outputs:
                                # Return first output URL
                                output_url = outputs[0] if isinstance(outputs, list) else outputs
                                logger.info(f"Generation completed successfully")
                                return output_url
                            else:
                                raise Exception("Generation completed but no output found")
                        
                        elif status == "failed":
                            error_msg = data.get("error", "Unknown error")
                            raise Exception(f"Generation failed: {error_msg}")
                        
                        elif status in ["starting", "processing", "pending", "queued"]:
                            logger.debug(f"Generation in progress... Status: {status}")
                        else:
                            logger.debug(f"Unknown status: {status}")
                    
                    else:
                        logger.warning(f"Poll request failed: {response.status_code}")
                
                except httpx.TimeoutException:
                    logger.warning(f"Poll request timed out on attempt {attempt + 1}")
                
                await asyncio.sleep(interval_s)
        
        raise TimeoutError(f"Generation timed out after {max_attempts} attempts")
    
    async def _download_video(self, video_url: str, output_path: Path) -> Path:
        """Download video from URL to local file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with httpx.AsyncClient(timeout=300.0) as client:  # 5 minute timeout for download
            response = await client.get(video_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download video: {response.status_code}")
            
            with open(output_path, "wb") as f:
                f.write(response.content)
        
        return output_path


class Veo31Service(WaveSpeedService):
    """Convenience service class specifically for Google Veo 3.1 image-to-video generation."""
    
    def __init__(self):
        """Initialize Veo 3.1 service."""
        super().__init__(model=WaveSpeedModel.VEO_3_1_I2V)
    
    async def generate_video(
        self,
        prompt: str,
        image_path: Union[str, Path],
        seed: int = -1,
        output_path: Optional[Path] = None,
        poll_timeout_s: float = 1800.0,  # 30 minute timeout
        poll_interval_s: float = 2.0,
        aspect_ratio: Optional[str] = None,  # Let auto-detection work by default
        duration: Optional[float] = 8.0,
        resolution: Optional[str] = "1080p",  # Default to 1080p for Veo 3.1
        generate_audio: bool = True,  # Default to generating audio
        negative_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate video using Google Veo 3.1 model with automatic aspect ratio detection.
        
        Args:
            prompt: Text description for video generation
            image_path: Path to input image file
            seed: Random seed for reproducibility (-1 for random)
            output_path: Optional path to save video (if not provided, returns URL)
            poll_timeout_s: Maximum time to wait for completion
            poll_interval_s: Time between polling attempts
            aspect_ratio: Video aspect ratio ("16:9" or "9:16" only). If None, automatically detects best ratio.
            duration: Video duration in seconds (up to 8 seconds)
            resolution: Output resolution ("720p" or "1080p", defaults to "1080p")
            generate_audio: Whether to generate audio (defaults to True)
            negative_prompt: Negative prompt to avoid certain content
            
        Returns:
            Dictionary with generation results including video URL/path
        """
        return await super().generate_video(
            prompt=prompt,
            image_path=image_path,
            seed=seed,
            output_path=output_path,
            poll_timeout_s=poll_timeout_s,
            poll_interval_s=poll_interval_s,
            aspect_ratio=aspect_ratio,
            duration=duration,
            resolution=resolution,
            generate_audio=generate_audio,
            negative_prompt=negative_prompt
        )


# ========================================
# VMEVALKIT WRAPPER CLASSES
# ========================================

class WaveSpeedWrapper(ModelWrapper):
    """
    VMEvalKit wrapper for WaveSpeedService to match the standard interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./outputs",
        **kwargs
    ):
        """Initialize WaveSpeed wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create WaveSpeedService instance
        self.wavespeed_service = WaveSpeedService(model=model)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,  # Not used by WaveSpeed but kept for interface compatibility
        output_filename: Optional[str] = None,
        seed: int = -1,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using WaveSpeed (synchronous wrapper).
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Not used (kept for compatibility)
            output_filename: Optional output filename
            seed: Random seed for reproducibility
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results
        """
        import time
        start_time = time.time()
        
        # Generate output path
        if not output_filename:
            # Create filename from model and timestamp
            safe_model = self.model.replace('/', '_').replace('-', '_')
            timestamp = int(time.time())
            output_filename = f"wavespeed_{safe_model}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Run async generation in sync context
        # Filter kwargs to only include parameters supported by WaveSpeedService.generate_video
        wavespeed_kwargs = {}
        # WaveSpeedService accepts: seed, output_path, poll_timeout_s, poll_interval_s,
        # aspect_ratio, duration, resolution, generate_audio, negative_prompt
        allowed_params = {
            'poll_timeout_s', 'poll_interval_s', 'aspect_ratio', 'duration',
            'resolution', 'generate_audio', 'negative_prompt'
        }
        for key, value in kwargs.items():
            if key in allowed_params:
                wavespeed_kwargs[key] = value
        
        result = asyncio.run(
            self.wavespeed_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                seed=seed,
                output_path=output_path,
                **wavespeed_kwargs
            )
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "success": bool(result.get("video_path")),
            "video_path": result.get("video_path"),
            "error": None,
            "duration_seconds": duration_taken,
            "generation_id": result.get("request_id", 'unknown'),
            "model": self.model,
            "status": "success" if result.get("video_path") else "failed",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "video_url": result.get("video_url"),
                "seed": seed,
                "wavespeed_result": result
            }
        }


class Veo31Wrapper(ModelWrapper):
    """
    VMEvalKit wrapper for Veo31Service (Google Veo 3.1 via WaveSpeed) to match the standard interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./outputs",
        **kwargs
    ):
        """Initialize Veo 3.1 wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Veo 3.1 uses WaveSpeed API key from environment
        self.veo_service = Veo31Service()
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using Veo 3.1 (synchronous wrapper).
        
        Args:
            image_path: Path to input image
            text_prompt: Text description for video generation
            duration: Video duration (up to 8 seconds for Veo 3.1)
            output_filename: Optional output filename
            **kwargs: Additional parameters for Veo 3.1
        """
        from datetime import datetime
        # Create output path with timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_filename:
            output_filename = f"veo31_{timestamp}.mp4"
        output_path = self.output_dir / output_filename
        
        # Extract Veo 3.1 specific parameters
        resolution = kwargs.get('resolution', '1080p')
        aspect_ratio = kwargs.get('aspect_ratio', None)
        generate_audio = kwargs.get('generate_audio', True)
        negative_prompt = kwargs.get('negative_prompt', None)
        seed = kwargs.get('seed', -1)
        
        # Run async function synchronously
        result = asyncio.run(
            self.veo_service.generate_video(
                prompt=text_prompt,
                image_path=image_path,
                output_path=output_path,
                duration=duration,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                generate_audio=generate_audio,
                negative_prompt=negative_prompt,
                seed=seed,
                poll_timeout_s=kwargs.get('poll_timeout_s', 300.0),
                poll_interval_s=kwargs.get('poll_interval_s', 2.0)
            )
        )
        
        # Return standardized format
        return {
            "success": bool(result.get("video_path")),
            "video_path": result.get("video_path", str(output_path)),
            "error": None,
            "duration_seconds": result.get("duration_seconds", 0),
            "generation_id": result.get("request_id", 'unknown'),
            "model": "google/veo3.1/image-to-video",
            "status": "success" if result.get("video_path") else "failed",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "video_url": result.get("video_url"),
                "wavespeed_result": result
            }
        }


class Veo31FastWrapper(ModelWrapper):
    """
    VMEvalKit wrapper for Veo31 Fast Service (Google Veo 3.1 Fast via WaveSpeed).
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./outputs",
        **kwargs
    ):
        """Initialize Veo 3.1 Fast wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Veo 3.1 Fast uses WaveSpeed API key from environment
        self.veo_service = WaveSpeedService(model=WaveSpeedModel.VEO_3_1_FAST_I2V)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using Veo 3.1 Fast (synchronous wrapper).
        
        Args:
            image_path: Path to input image
            text_prompt: Text description for video generation
            duration: Video duration (up to 8 seconds for Veo 3.1)
            output_filename: Optional output filename
            **kwargs: Additional parameters for Veo 3.1
        """
        from datetime import datetime
        # Create output path with timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_filename:
            output_filename = f"veo31_fast_{timestamp}.mp4"
        output_path = self.output_dir / output_filename
        
        # Extract Veo 3.1 specific parameters
        resolution = kwargs.get('resolution', '1080p')
        aspect_ratio = kwargs.get('aspect_ratio', None)
        generate_audio = kwargs.get('generate_audio', True)
        negative_prompt = kwargs.get('negative_prompt', None)
        seed = kwargs.get('seed', -1)
        
        # Run async function synchronously
        result = asyncio.run(
            self.veo_service.generate_video(
                prompt=text_prompt,
                image_path=image_path,
                output_path=output_path,
                duration=duration,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                generate_audio=generate_audio,
                negative_prompt=negative_prompt,
                seed=seed,
                poll_timeout_s=kwargs.get('poll_timeout_s', 300.0),
                poll_interval_s=kwargs.get('poll_interval_s', 2.0)
            )
        )
        
        # Return standardized format
        return {
            "success": bool(result.get("video_path")),
            "video_path": result.get("video_path", str(output_path)),
            "error": None,
            "duration_seconds": result.get("duration_seconds", 0),
            "generation_id": result.get("request_id", 'unknown'),
            "model": "google/veo3.1-fast/image-to-video",
            "status": "success" if result.get("video_path") else "failed",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "video_url": result.get("video_url"),
                "wavespeed_result": result
            }
        }