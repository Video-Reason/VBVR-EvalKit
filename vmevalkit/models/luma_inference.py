"""
Luma Dream Machine inference implementation.

Direct inference API for Luma's text+image→video generation.
"""

import os
import time
import base64
import requests
from typing import Optional, Dict, Any, Union
from pathlib import Path
import json
from io import BytesIO
from PIL import Image
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from ..utils.s3_uploader import S3ImageUploader


class LumaAPIError(Exception):
    """Custom exception for Luma API errors."""
    pass


class LumaInference:
    """
    Direct inference API for Luma Dream Machine.
    
    Supports text+image→video generation for reasoning tasks.
    """
    
    BASE_URL = "https://api.lumalabs.ai/dream-machine/v1"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        enhance_prompt: bool = True,
        loop_video: bool = False,
        aspect_ratio: str = "16:9",
        model: str = "ray-2",
        verbose: bool = True,
        output_dir: str = "./data/outputs"
    ):
        """
        Initialize Luma inference client.
        
        Args:
            api_key: API key for authentication
            enhance_prompt: Whether to enhance the prompt automatically
            loop_video: Whether to create looping videos
            aspect_ratio: Video aspect ratio (e.g., "16:9", "1:1", "9:16")
            model: Luma model to use (e.g., "ray-2")
            verbose: Print progress messages
            output_dir: Directory to save generated videos
        """
        self.api_key = api_key or os.environ.get("LUMA_API_KEY")
        if not self.api_key:
            raise ValueError("API key required. Set LUMA_API_KEY environment variable.")
        
        self.enhance_prompt = enhance_prompt
        self.loop_video = loop_video
        self.aspect_ratio = aspect_ratio
        self.model = model
        self.verbose = verbose
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Headers for API requests
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Initialize S3 uploader for serving images to Luma
        self._s3_uploader = None
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 5.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate a video from text prompt and image.
        
        Args:
            image_path: Path to input image
            text_prompt: Text instructions for video generation
            duration: Video duration in seconds (Luma typically generates 5s videos)
            output_filename: Optional output filename (auto-generated if not provided)
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            Dictionary with:
            - video_path: Path to generated video
            - generation_id: Luma generation ID
            - status: Generation status
            - duration_seconds: Time taken to generate
        """
        start_time = time.time()
        
        # Convert image to URL (Luma requires image URLs)
        image_url = self._get_image_url(image_path)
        
        # Create generation
        generation_id = self._create_generation(image_url, text_prompt)
        
        if self.verbose:
            print(f"Started generation: {generation_id}")
        
        # Poll for completion
        video_url = self._poll_generation(generation_id)
        
        # Download video
        if not output_filename:
            output_filename = f"luma_{generation_id}.mp4"
        
        video_path = self.output_dir / output_filename
        self._download_video(video_url, video_path)
        
        duration_seconds = time.time() - start_time
        
        result = {
            "video_path": str(video_path),
            "generation_id": generation_id,
            "status": "success",
            "duration_seconds": duration_seconds,
            "model": self.model,
            "prompt": text_prompt,
            "image_path": str(image_path)
        }
        
        if self.verbose:
            print(f"✅ Generated video: {video_path}")
            print(f"   Time taken: {duration_seconds:.1f}s")
        
        # Clean up S3 resources
        if self._s3_uploader:
            self._s3_uploader.cleanup()
            if self.verbose:
                print("   Cleaned up temporary S3 resources")
        
        return result
    
    def _get_image_url(self, image_path: Union[str, Path]) -> str:
        """Convert local image to URL using S3."""
        # Initialize S3 uploader if needed
        if self._s3_uploader is None:
            self._s3_uploader = S3ImageUploader()
        
        # Upload image and get URL
        image_url = self._s3_uploader.upload(str(image_path))
        if not image_url:
            raise LumaAPIError("Failed to upload image to S3")
        
        if self.verbose:
            print(f"Serving image at: {image_url}")
        
        return image_url
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def _create_generation(self, image_url: str, text_prompt: str) -> str:
        """Create a new video generation."""
        payload = {
            "prompt": text_prompt,
            "keyframes": {
                "frame0": {
                    "type": "image",
                    "url": image_url
                }
            },
            "enhance_prompt": self.enhance_prompt,
            "loop": self.loop_video,
            "aspect_ratio": self.aspect_ratio,
            "model": self.model
        }
        
        response = requests.post(
            f"{self.BASE_URL}/generations",
            headers=self.headers,
            json=payload
        )
        
        if response.status_code != 201:
            raise LumaAPIError(f"Failed to create generation: {response.text}")
        
        data = response.json()
        return data["id"]
    
    def _poll_generation(self, generation_id: str, timeout: int = 300) -> str:
        """Poll for generation completion."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check status
            response = requests.get(
                f"{self.BASE_URL}/generations/{generation_id}",
                headers=self.headers
            )
            
            if response.status_code != 200:
                raise LumaAPIError(f"Failed to check generation: {response.text}")
            
            data = response.json()
            state = data.get("state")
            
            if state == "completed":
                # Get video URL from assets
                if "assets" in data and "video" in data["assets"]:
                    return data["assets"]["video"]
                else:
                    raise LumaAPIError("Generation completed but no video URL found")
            
            elif state == "failed":
                failure_reason = data.get("failure_reason", "Unknown error")
                raise LumaAPIError(f"Generation failed: {failure_reason}")
            
            # Still processing
            if self.verbose:
                elapsed = int(time.time() - start_time)
                print(f"   Generating... ({elapsed}s)", end='\r')
            
            time.sleep(5)  # Poll every 5 seconds
        
        raise LumaAPIError(f"Generation timed out after {timeout} seconds")
    
    def _download_video(self, video_url: str, output_path: Path):
        """Download video from URL."""
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)


def generate_video(
    image_path: str,
    text_prompt: str,
    api_key: Optional[str] = None,
    output_dir: str = "./data/outputs",
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for one-shot video generation.
    
    Args:
        image_path: Path to input image
        text_prompt: Text instructions
        api_key: Optional API key (uses env var if not provided)
        output_dir: Where to save the video
        **kwargs: Additional parameters passed to LumaInference
    
    Returns:
        Dictionary with generation results
    """
    client = LumaInference(api_key=api_key, output_dir=output_dir, **kwargs)
    return client.generate(image_path, text_prompt)
