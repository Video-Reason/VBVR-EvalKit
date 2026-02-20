"""
Kling AI Image-to-Video Generation Service.

API Documentation: https://app.klingai.com
- Auth: uses KLING_API_KEY environment variable (JWT token)
- Models:
  - kling-v1: Original Kling model
  - kling-v1-6: Kling v1.6
  - kling-v2-master: Kling 2.0 Master
  - kling-v2-1-master: Kling 2.1 Master
  - kling-v2-5-turbo: Kling 2.5 Turbo (faster)
  - kling-v2-6: Kling 2.6 (latest)
- Endpoints:
  - Text-to-Video: POST /v1/videos/text2video
  - Image-to-Video: POST /v1/videos/image2video
  - Query task: GET /v1/videos/image2video/{task_id}
"""

from __future__ import annotations

import os
import base64
import time
import asyncio
import logging
import httpx
import jwt
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime, timedelta

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


def get_kling_api_key() -> str:
    """Get KLING_API_KEY from environment."""
    _hydrate_env_from_nearby_dotenv()
    key = os.environ.get("KLING_API_KEY", "")
    if not key:
        raise ValueError(
            "KLING_API_KEY not found. Set it in your environment or .env file."
        )
    return key


def get_kling_access_key() -> str:
    """Get KLING_ACCESS_KEY from environment (for JWT generation)."""
    _hydrate_env_from_nearby_dotenv()
    key = os.environ.get("KLING_ACCESS_KEY", "")
    return key


def get_kling_secret_key() -> str:
    """Get KLING_SECRET_KEY from environment (for JWT generation)."""
    _hydrate_env_from_nearby_dotenv()
    key = os.environ.get("KLING_SECRET_KEY", "")
    return key


def generate_jwt_token(access_key: str, secret_key: str) -> str:
    """Generate JWT token for Kling API authentication."""
    headers = {
        "alg": "HS256",
        "typ": "JWT"
    }
    payload = {
        "iss": access_key,
        "exp": int(time.time()) + 1800,  # Valid for 30 minutes
        "nbf": int(time.time()) - 5  # Starts 5 seconds ago to account for clock skew
    }
    return jwt.encode(payload, secret_key, headers=headers)


# Valid model identifiers
SUPPORTED_MODELS = {
    "kling-v1", "kling-v1-6", "kling-v2-master",
    "kling-v2-1-master", "kling-v2-5-turbo", "kling-v2-6",
}


class KlingService:
    """
    Kling AI video generation service using their REST API.
    
    Supports image-to-video generation with various Kling model versions.
    """
    
    BASE_URL = "https://api-singapore.klingai.com"
    
    def __init__(
        self,
        model: str = "kling-v2-6",
        duration: float = 5.0,
        aspect_ratio: str = "16:9",
        mode: str = "std",  # "std" or "pro"
        **kwargs  # Accept and ignore extra kwargs from runner
    ):
        self.model = model
        self.duration = duration
        self.aspect_ratio = aspect_ratio
        self.mode = mode
        
        # Try to get API key or generate JWT
        self._api_key = None
        self._access_key = None
        self._secret_key = None
        
        try:
            self._api_key = get_kling_api_key()
        except ValueError:
            # Try JWT authentication
            self._access_key = get_kling_access_key()
            self._secret_key = get_kling_secret_key()
            if not self._access_key or not self._secret_key:
                raise ValueError(
                    "Kling API authentication required. Set either:\n"
                    "  - KLING_API_KEY (direct API key), or\n"
                    "  - KLING_ACCESS_KEY and KLING_SECRET_KEY (for JWT generation)"
                )
        
        logger.info(f"Initialized KlingService with model={self.model}")
    
    def _get_auth_token(self) -> str:
        """Get authorization token (either direct API key or generated JWT)."""
        if self._api_key:
            return self._api_key
        return generate_jwt_token(self._access_key, self._secret_key)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authorization."""
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._get_auth_token()}"
        }
    
    def _get_model_constraints(self, model: str) -> Dict[str, Any]:
        """Get constraints for a specific model."""
        constraints = {
            "kling-v1": {
                "durations": [5],
                "aspect_ratios": ["16:9", "9:16", "1:1"],
                "modes": ["std"],
            },
            "kling-v1-6": {
                "durations": [5],
                "aspect_ratios": ["16:9", "9:16", "1:1"],
                "modes": ["std", "pro"],
            },
            "kling-v2-master": {
                "durations": [5, 10],
                "aspect_ratios": ["16:9", "9:16", "1:1"],
                "modes": ["std", "pro"],
            },
            "kling-v2-1-master": {
                "durations": [5, 10],
                "aspect_ratios": ["16:9", "9:16", "1:1"],
                "modes": ["std", "pro"],
            },
            "kling-v2-5-turbo": {
                "durations": [5, 10],
                "aspect_ratios": ["16:9", "9:16", "1:1"],
                "modes": ["std"],
            },
            "kling-v2-6": {
                "durations": [5, 10],
                "aspect_ratios": ["16:9", "9:16", "1:1"],
                "modes": ["std", "pro"],
            },
        }
        return constraints.get(model, constraints["kling-v2-6"])
    
    def _encode_image_to_base64(self, image_path: Union[str, Path]) -> str:
        """Encode an image file to base64 string."""
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        
        with open(path, "rb") as f:
            image_data = f.read()
        
        return base64.b64encode(image_data).decode("utf-8")
    
    async def generate_video(
        self,
        prompt: str,
        image_path: Optional[Union[str, Path]] = None,
        duration: Optional[float] = None,
        aspect_ratio: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        cfg_scale: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Generate a video from text prompt and optional reference image.
        
        Args:
            prompt: Text description for video generation (max 2500 chars)
            image_path: Optional path to reference image for image-to-video
            duration: Video duration in seconds (5 or 10)
            aspect_ratio: Aspect ratio (16:9, 9:16, 1:1)
            negative_prompt: Optional negative prompt
            cfg_scale: CFG scale for generation (0.0-1.0)
            
        Returns:
            Dictionary with video_url and metadata
        """
        duration = duration or self.duration
        aspect_ratio = aspect_ratio or self.aspect_ratio
        
        # Validate constraints
        constraints = self._get_model_constraints(self.model)
        
        # Use image-to-video if image provided, else text-to-video
        if image_path:
            endpoint = f"{self.BASE_URL}/v1/videos/image2video"
            image_base64 = self._encode_image_to_base64(image_path)
            
            payload = {
                "model_name": self.model,
                "prompt": prompt[:2500],  # Max 2500 chars
                "image": image_base64,
                "duration": str(int(duration)),
                "aspect_ratio": aspect_ratio,
                "cfg_scale": cfg_scale,
            }
        else:
            endpoint = f"{self.BASE_URL}/v1/videos/text2video"
            payload = {
                "model_name": self.model,
                "prompt": prompt[:2500],
                "duration": str(int(duration)),
                "aspect_ratio": aspect_ratio,
                "cfg_scale": cfg_scale,
            }
        
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt[:2500]
        
        if self.mode in constraints.get("modes", ["std"]):
            payload["mode"] = self.mode
        
        logger.info(f"Submitting Kling generation request: model={self.model}, duration={duration}s")
        
        # Submit generation request
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                endpoint,
                headers=self._get_headers(),
                json=payload
            )
            
            if response.status_code != 200:
                error_text = response.text
                logger.error(f"Kling API error: {response.status_code} - {error_text}")
                raise Exception(f"Kling API error: {response.status_code} - {error_text}")
            
            result = response.json()
            
            if result.get("code") != 0:
                raise Exception(f"Kling API error: {result.get('message', 'Unknown error')}")
            
            task_id = result.get("data", {}).get("task_id")
            if not task_id:
                raise Exception(f"No task_id in response: {result}")
            
            logger.info(f"Kling task submitted: {task_id}")
        
        # Poll for completion
        video_result = await self._poll_task(task_id, image_path is not None)
        return video_result
    
    async def _poll_task(
        self,
        task_id: str,
        is_image2video: bool = True,
        max_wait: int = 600,
        poll_interval: int = 5
    ) -> Dict[str, Any]:
        """Poll for task completion."""
        endpoint_type = "image2video" if is_image2video else "text2video"
        url = f"{self.BASE_URL}/v1/videos/{endpoint_type}/{task_id}"
        
        start_time = time.time()
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            while time.time() - start_time < max_wait:
                response = await client.get(url, headers=self._get_headers())
                
                if response.status_code != 200:
                    logger.warning(f"Poll request failed: {response.status_code}")
                    await asyncio.sleep(poll_interval)
                    continue
                
                result = response.json()
                
                if result.get("code") != 0:
                    raise Exception(f"Kling poll error: {result.get('message')}")
                
                data = result.get("data", {})
                status = data.get("task_status")
                
                if status == "succeed":
                    videos = data.get("task_result", {}).get("videos", [])
                    if videos:
                        video_url = videos[0].get("url")
                        return {
                            "video_url": video_url,
                            "task_id": task_id,
                            "status": "completed",
                            "duration": data.get("task_result", {}).get("videos", [{}])[0].get("duration"),
                        }
                    raise Exception("No video URL in completed task")
                
                elif status == "failed":
                    error_msg = data.get("task_status_msg", "Unknown error")
                    raise Exception(f"Kling generation failed: {error_msg}")
                
                elif status in ["submitted", "processing"]:
                    progress = data.get("task_progress", 0)
                    logger.info(f"Kling task {task_id}: {status} ({progress}%)")
                    await asyncio.sleep(poll_interval)
                
                else:
                    logger.warning(f"Unknown status: {status}")
                    await asyncio.sleep(poll_interval)
        
        raise TimeoutError(f"Kling generation timed out after {max_wait}s")
    
    async def download_video(self, video_url: str, output_path: Path) -> Path:
        """Download video from URL to local file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.get(video_url)
            if response.status_code != 200:
                raise Exception(f"Failed to download video: {response.status_code}")
            
            with open(output_path, "wb") as f:
                f.write(response.content)
        
        logger.info(f"Downloaded video to: {output_path}")
        return output_path


class KlingWrapper(ModelWrapper):
    """
    ModelWrapper implementation for Kling AI video generation.
    """
    
    def __init__(self, model: str = "kling-v2-6", **kwargs):
        self.model = model
        self.kwargs = kwargs
        self.kling_service = KlingService(model=model, **kwargs)
        logger.info(f"Initialized KlingWrapper with model={model}")
    
    def generate(
        self,
        image_path: str,
        text_prompt: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video from image and text prompt.
        
        Args:
            image_path: Path to input image
            text_prompt: Text prompt for generation
            output_dir: Directory to save output video
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results
        """
        if output_dir:
            output_path = Path(output_dir) / "video.mp4"
        else:
            output_path = Path("output_video.mp4")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Filter kwargs to only include parameters supported by KlingService.generate_video()
        allowed_params = {'duration', 'aspect_ratio', 'negative_prompt', 'cfg_scale'}
        kling_kwargs = {k: v for k, v in kwargs.items() if k in allowed_params}
        
        try:
            # Run async generation
            result = asyncio.run(
                self.kling_service.generate_video(
                    prompt=text_prompt,
                    image_path=image_path,
                    **kling_kwargs
                )
            )
            
            # Download the video
            video_url = result.get("video_url")
            if video_url:
                asyncio.run(
                    self.kling_service.download_video(video_url, output_path)
                )
            
            return {
                "status": "completed",
                "video_path": str(output_path),
                "video_url": video_url,
                "task_id": result.get("task_id"),
                "model": self.model,
            }
            
        except Exception as e:
            logger.error(f"Kling generation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "model": self.model,
            }
