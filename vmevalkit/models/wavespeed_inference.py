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

logger = logging.getLogger(__name__)


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
    
    async def generate_video(
        self,
        prompt: str,
        image_path: Union[str, Path],
        seed: int = -1,
        output_path: Optional[Path] = None,
        poll_timeout_s: float = 300.0,
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
        # Encode image to base64
        image_b64 = self._encode_image(image_path)
        
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
    
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image file to base64."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
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
        
        async with httpx.AsyncClient(timeout=30.0) as client:
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
        
        async with httpx.AsyncClient(timeout=30.0) as client:
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
        
        async with httpx.AsyncClient(timeout=120.0) as client:
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
        poll_timeout_s: float = 300.0,
        poll_interval_s: float = 2.0,
        aspect_ratio: Optional[str] = "16:9",
        duration: Optional[float] = 8.0,
        resolution: Optional[str] = "1080p",  # Default to 1080p for Veo 3.1
        generate_audio: bool = True,  # Default to generating audio
        negative_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate video using Google Veo 3.1 model.
        
        Args:
            prompt: Text description for video generation
            image_path: Path to input image file
            seed: Random seed for reproducibility (-1 for random)
            output_path: Optional path to save video (if not provided, returns URL)
            poll_timeout_s: Maximum time to wait for completion
            poll_interval_s: Time between polling attempts
            aspect_ratio: Video aspect ratio (e.g., "16:9", "9:16", "1:1")
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