"""
VMEvalKit Inference Runner - Multi-Provider Video Generation

Unified interface for 37+ text+image→video models across 9 major providers:
- Luma Dream Machine (2 models)
- Google Veo (3 models) 
- Google Veo 3.1 via WaveSpeed (2 models)
- WaveSpeed WAN 2.x (18 models)
- Runway ML (3 models)
- OpenAI Sora (2 models)
- LTX-Video (3 models) - Open Source
- HunyuanVideo (1 model) - Open Source  
- VideoCrafter (1 model) - Open Source
- DynamiCrafter (3 models) - Open Source

Organized by families for easy scaling and management.
"""

import os
import asyncio
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime
import json

from ..models import (
    LumaInference, VeoService, Veo31Service, WaveSpeedService, RunwayService, SoraService,
    LTXVideoWrapper, HunyuanVideoWrapper, VideoCrafterWrapper, DynamiCrafterWrapper
)


class VeoWrapper:
    """
    Wrapper for VeoService to match the LumaInference interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """Initialize Veo wrapper."""
        # Veo uses Google Cloud authentication (no API key needed here)
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
        import time
        start_time = time.time()
        
        # Convert duration to int (Veo requires int)
        duration_seconds = int(duration)
        
        # Run async generation in sync context
        # Filter kwargs to only include parameters supported by VeoService.generate_video
        veo_kwargs = {}
        # VeoService accepts: image_path, image_gcs_uri, duration_seconds, aspect_ratio, resolution,
        # negative_prompt, enhance_prompt, generate_audio, sample_count, seed, person_generation,
        # storage_uri, poll_interval_s, poll_timeout_s, download_from_gcs
        allowed_params = {
            'image_gcs_uri', 'aspect_ratio', 'resolution', 'negative_prompt', 'enhance_prompt',
            'generate_audio', 'sample_count', 'seed', 'person_generation', 'storage_uri',
            'poll_interval_s', 'poll_timeout_s', 'download_from_gcs'
        }
        for key, value in kwargs.items():
            if key in allowed_params:
                veo_kwargs[key] = value
        
        video_bytes, metadata = asyncio.run(
            self.veo_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                duration_seconds=duration_seconds,
                **veo_kwargs
            )
        )
        
        # Save video if we got bytes
        video_path = None
        if video_bytes:
            if not output_filename:
                # Generate filename from operation name if available
                op_name = metadata.get('operation_name', 'veo_output')
                if isinstance(op_name, str) and '/' in op_name:
                    op_id = op_name.split('/')[-1]
                    output_filename = f"veo_{op_id}.mp4"
                else:
                    output_filename = f"veo_{int(time.time())}.mp4"
            
            video_path = self.output_dir / output_filename
            asyncio.run(self.veo_service.save_video(video_bytes, video_path))
        
        duration_taken = time.time() - start_time
        
        return {
            "success": video_bytes is not None,
            "video_path": str(video_path) if video_path else None,
            "error": None,
            "duration_seconds": duration_taken,
            "generation_id": metadata.get('operation_name', 'unknown'),
            "model": self.model,
            "status": "success" if video_bytes else "completed_no_download",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "veo_metadata": metadata
            }
        }


class Veo31FastWrapper:
    """
    Wrapper for Veo31 Fast Service (Google Veo 3.1 Fast via WaveSpeed).
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """Initialize Veo 3.1 Fast wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Veo 3.1 Fast uses WaveSpeed API key from environment
        from vmevalkit.models.wavespeed_inference import WaveSpeedService, WaveSpeedModel
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


class Veo31Wrapper:
    """
    Wrapper for Veo31Service (Google Veo 3.1 via WaveSpeed) to match the LumaInference interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """Initialize Veo 3.1 wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Veo 3.1 uses WaveSpeed API key from environment
        from vmevalkit.models.wavespeed_inference import Veo31Service
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


class WaveSpeedWrapper:
    """
    Wrapper for WaveSpeedService to match the LumaInference interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
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


class RunwayWrapper:
    """
    Wrapper for RunwayService to match the LumaInference interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
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
            # Create filename from model and timestamp
            safe_model = self.model.replace('_', '-')
            timestamp = int(time.time())
            output_filename = f"runway_{safe_model}_{timestamp}.mp4"
        
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


class OpenAIWrapper:
    """
    Wrapper for SoraService to match the LumaInference interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """Initialize OpenAI Sora wrapper."""
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create SoraService instance
        self.sora_service = SoraService(model=model)
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        size: str = "1280x720",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using Sora (synchronous wrapper).
        
        Args:
            image_path: Path to input image (must match size exactly)
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (4, 8, or 12)
            output_filename: Optional output filename
            size: Video resolution (must match image dimensions exactly)
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with generation results
        """
        import time
        start_time = time.time()
        
        # Convert duration to int (Sora expects int)
        duration_int = int(duration)
        
        # Generate output path
        if not output_filename:
            # Create filename from model and timestamp
            safe_model = self.model.replace('-', '_')
            timestamp = int(time.time())
            output_filename = f"sora_{safe_model}_{timestamp}.mp4"
        
        output_path = self.output_dir / output_filename
        
        # Run async generation in sync context
        # Filter kwargs - SoraService.generate_video only accepts:
        # prompt, image_path, duration, size, output_path, auto_pad
        # All parameters are already passed explicitly, so no additional kwargs
        result = asyncio.run(
            self.sora_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                duration=duration_int,
                size=size,
                output_path=output_path,
                auto_pad=True  # Enable auto-padding by default
            )
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "success": bool(result.get("video_path")),
            "video_path": result.get("video_path"),
            "error": None,
            "duration_seconds": duration_taken,
            "generation_id": result.get("video_id", 'unknown'),
            "model": self.model,
            "status": "success" if result.get("video_path") else "failed",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(image_path),
                "duration": duration_int,
                "size": result.get("size"),
                "sora_result": result
            }
        }


# ========================================
# MODEL FAMILY DEFINITIONS
# ========================================

# Luma Dream Machine Models
LUMA_MODELS = {
    "luma-ray-2": {
        "class": LumaInference,
        "model": "ray-2",
        "description": "Luma Ray 2 - Latest model with best quality",
        "family": "Luma Dream Machine"
    },
    "luma-ray-flash-2": {
        "class": LumaInference,
        "model": "ray-flash-2", 
        "description": "Luma Ray Flash 2 - Faster generation",
        "family": "Luma Dream Machine"
    }
}

# Google Veo Models (Vertex AI)
VEO_MODELS = {
    "veo-2.0-generate": {
        "class": VeoWrapper,
        "model": "veo-2.0-generate-001",
        "description": "Google Veo 2.0 - GA model for text+image→video",
        "family": "Google Veo"
    },
    "veo-3.0-generate": {
        "class": VeoWrapper,
        "model": "veo-3.0-generate-preview",
        "description": "Google Veo 3.0 - Preview model with advanced capabilities",
        "family": "Google Veo"
    },
    "veo-3.0-fast-generate": {
        "class": VeoWrapper,
        "model": "veo-3.0-fast-generate-preview",
        "description": "Google Veo 3.0 Fast - Preview model for faster generation",
        "family": "Google Veo"
    }
}

# WaveSpeedAI WAN Models
WAVESPEED_WAN_22_MODELS = {
    "wavespeed-wan-2.2-i2v-480p": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-480p",
        "description": "WaveSpeed WAN 2.2 I2V 480p - Standard quality",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-480p-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-480p-ultra-fast",
        "description": "WaveSpeed WAN 2.2 I2V 480p Ultra Fast - Speed optimized",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-480p-lora": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-480p-lora",
        "description": "WaveSpeed WAN 2.2 I2V 480p LoRA - Enhanced with LoRA",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-480p-lora-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-480p-lora-ultra-fast",
        "description": "WaveSpeed WAN 2.2 I2V 480p LoRA Ultra Fast - Best speed with LoRA",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-5b-720p": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-5b-720p",
        "description": "WaveSpeed WAN 2.2 I2V 5B 720p - High resolution 5B model",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-5b-720p-lora": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-5b-720p-lora",
        "description": "WaveSpeed WAN 2.2 I2V 5B 720p LoRA - High-res with LoRA",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-720p": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-720p",
        "description": "WaveSpeed WAN 2.2 I2V 720p - High resolution",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-720p-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-720p-ultra-fast",
        "description": "WaveSpeed WAN 2.2 I2V 720p Ultra Fast - High-res speed optimized",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-720p-lora": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-720p-lora",
        "description": "WaveSpeed WAN 2.2 I2V 720p LoRA - High-res with LoRA",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-720p-lora-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.2/i2v-720p-lora-ultra-fast",
        "description": "WaveSpeed WAN 2.2 I2V 720p LoRA Ultra Fast - Fastest high-res LoRA",
        "family": "WaveSpeed WAN 2.2"
    }
}

WAVESPEED_WAN_21_MODELS = {
    "wavespeed-wan-2.1-i2v-480p": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-480p",
        "description": "WaveSpeed WAN 2.1 I2V 480p - Standard quality",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-480p-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-480p-ultra-fast",
        "description": "WaveSpeed WAN 2.1 I2V 480p Ultra Fast - Speed optimized",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-480p-lora": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-480p-lora",
        "description": "WaveSpeed WAN 2.1 I2V 480p LoRA - Enhanced with LoRA",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-480p-lora-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-480p-lora-ultra-fast",
        "description": "WaveSpeed WAN 2.1 I2V 480p LoRA Ultra Fast - Best speed with LoRA",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-720p": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-720p",
        "description": "WaveSpeed WAN 2.1 I2V 720p - High resolution",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-720p-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-720p-ultra-fast",
        "description": "WaveSpeed WAN 2.1 I2V 720p Ultra Fast - High-res speed optimized",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-720p-lora": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-720p-lora",
        "description": "WaveSpeed WAN 2.1 I2V 720p LoRA - High-res with LoRA",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-720p-lora-ultra-fast": {
        "class": WaveSpeedWrapper,
        "model": "wan-2.1/i2v-720p-lora-ultra-fast",
        "description": "WaveSpeed WAN 2.1 I2V 720p LoRA Ultra Fast - Fastest high-res LoRA",
        "family": "WaveSpeed WAN 2.1"
    }
}

# Runway ML Models
RUNWAY_MODELS = {
    "runway-gen4-turbo": {
        "class": RunwayWrapper,
        "model": "gen4_turbo",
        "description": "Runway Gen-4 Turbo - Fast high-quality generation (5s or 10s)",
        "family": "Runway ML"
    },
    "runway-gen4-aleph": {
        "class": RunwayWrapper,
        "model": "gen4_aleph",
        "description": "Runway Gen-4 Aleph - Premium quality (5s)",
        "family": "Runway ML"
    },
    "runway-gen3a-turbo": {
        "class": RunwayWrapper,
        "model": "gen3a_turbo",
        "description": "Runway Gen-3A Turbo - Proven performance (5s or 10s)",
        "family": "Runway ML"
    }
}

# OpenAI Sora Models
OPENAI_SORA_MODELS = {
    "openai-sora-2": {
        "class": OpenAIWrapper,
        "model": "sora-2",
        "description": "OpenAI Sora-2 - High-quality video generation (4s/8s/12s)",
        "family": "OpenAI Sora"
    },
    "openai-sora-2-pro": {
        "class": OpenAIWrapper,
        "model": "sora-2-pro",
        "description": "OpenAI Sora-2-Pro - Enhanced model with more resolution options",
        "family": "OpenAI Sora"
    }
}

# Google Veo 3.1 Models (via WaveSpeed)
GOOGLE_VEO31_MODELS = {
    "veo-3.1": {
        "class": Veo31Wrapper,
        "model": "veo-3.1",
        "args": {},
        "description": "Google Veo 3.1 - Native 1080p with audio generation (via WaveSpeed)",
        "family": "Google Veo 3.1"
    },
    "veo-3.1-720p": {
        "class": Veo31Wrapper,
        "model": "veo-3.1-720p",
        "args": {"resolution": "720p"},
        "description": "Google Veo 3.1 - 720p with audio generation (via WaveSpeed)",
        "family": "Google Veo 3.1"
    },
    "veo-3.1-fast": {
        "class": Veo31FastWrapper,
        "model": "veo-3.1-fast",
        "args": {},
        "description": "Google Veo 3.1 Fast - 1080p, 30% faster generation (via WaveSpeed)",
        "family": "Google Veo 3.1"
    },
    "veo-3.1-fast-720p": {
        "class": Veo31FastWrapper,
        "model": "veo-3.1-fast-720p",
        "args": {"resolution": "720p"},
        "description": "Google Veo 3.1 Fast - 720p, 30% faster generation (via WaveSpeed)",
        "family": "Google Veo 3.1"
    }
}

# ========================================
# OPEN-SOURCE MODELS (SUBMODULES)  
# ========================================

# LTX-Video Models (Lightricks)
LTX_VIDEO_MODELS = {
    "ltx-video-13b-distilled": {
        "class": LTXVideoWrapper,
        "model": "ltxv-13b-0.9.8-distilled",
        "description": "LTX-Video 13B Distilled - Real-time video generation, high quality",
        "family": "LTX-Video"
    },
    "ltx-video-13b-dev": {
        "class": LTXVideoWrapper,
        "model": "ltxv-13b-0.9.8-dev",
        "description": "LTX-Video 13B Dev - Development version with latest features",
        "family": "LTX-Video"
    },
    "ltx-video-2b-distilled": {
        "class": LTXVideoWrapper,
        "model": "ltxv-2b-0.9.8-distilled",
        "description": "LTX-Video 2B Distilled - Smaller, faster model",
        "family": "LTX-Video"
    }
}

# HunyuanVideo-I2V Models (Tencent)
HUNYUAN_VIDEO_MODELS = {
    "hunyuan-video-i2v": {
        "class": HunyuanVideoWrapper,
        "model": "hunyuan-video-i2v",
        "description": "HunyuanVideo-I2V - High-quality image-to-video up to 720p",
        "family": "HunyuanVideo"
    }
}

# VideoCrafter Models (AILab-CVC)
VIDEOCRAFTER_MODELS = {
    "videocrafter2-512": {
        "class": VideoCrafterWrapper,
        "model": "videocrafter2",
        "description": "VideoCrafter2 - High-quality text-guided video generation",
        "family": "VideoCrafter"
    }
}

# DynamiCrafter Models (Doubiiu)
DYNAMICRAFTER_MODELS = {
    "dynamicrafter-512": {
        "class": DynamiCrafterWrapper,
        "model": "dynamicrafter-512",
        "description": "DynamiCrafter 512p - Image animation with video diffusion",
        "family": "DynamiCrafter"
    },
    "dynamicrafter-256": {
        "class": DynamiCrafterWrapper,
        "model": "dynamicrafter-256",
        "description": "DynamiCrafter 256p - Faster image animation",
        "family": "DynamiCrafter"
    },
    "dynamicrafter-1024": {
        "class": DynamiCrafterWrapper,
        "model": "dynamicrafter-1024",
        "description": "DynamiCrafter 1024p - High-resolution image animation",
        "family": "DynamiCrafter"
    }
}

# ========================================
# COMBINED MODEL REGISTRY
# ========================================

# Combine all model families into unified registry
AVAILABLE_MODELS = {
    **LUMA_MODELS,
    **VEO_MODELS,
    **GOOGLE_VEO31_MODELS,
    **WAVESPEED_WAN_22_MODELS,
    **WAVESPEED_WAN_21_MODELS,
    **RUNWAY_MODELS,
    **OPENAI_SORA_MODELS,
    **LTX_VIDEO_MODELS,
    **HUNYUAN_VIDEO_MODELS,
    **VIDEOCRAFTER_MODELS,
    **DYNAMICRAFTER_MODELS
}

# Model families metadata for easier management
MODEL_FAMILIES = {
    "Luma Dream Machine": LUMA_MODELS,
    "Google Veo": VEO_MODELS,
    "Google Veo 3.1": GOOGLE_VEO31_MODELS,
    "WaveSpeed WAN 2.2": WAVESPEED_WAN_22_MODELS,
    "WaveSpeed WAN 2.1": WAVESPEED_WAN_21_MODELS,
    "Runway ML": RUNWAY_MODELS,
    "OpenAI Sora": OPENAI_SORA_MODELS,
    "LTX-Video": LTX_VIDEO_MODELS,
    "HunyuanVideo": HUNYUAN_VIDEO_MODELS,
    "VideoCrafter": VIDEOCRAFTER_MODELS,
    "DynamiCrafter": DYNAMICRAFTER_MODELS
}


# ========================================
# UTILITY FUNCTIONS
# ========================================

def get_models_by_family(family_name: str) -> Dict[str, Dict[str, Any]]:
    """Get all models from a specific family."""
    if family_name not in MODEL_FAMILIES:
        raise ValueError(f"Unknown family: {family_name}. Available: {list(MODEL_FAMILIES.keys())}")
    return MODEL_FAMILIES[family_name]


def get_model_family(model_name: str) -> str:
    """Get the family name for a specific model."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    return AVAILABLE_MODELS[model_name]["family"]


def list_all_families() -> Dict[str, int]:
    """List all model families and their counts."""
    return {
        family_name: len(family_models)
        for family_name, family_models in MODEL_FAMILIES.items()
    }


def add_model_family(family_name: str, models: Dict[str, Dict[str, Any]]) -> None:
    """
    Add a new model family to the registry.
    
    Args:
        family_name: Name of the model family
        models: Dictionary of model configurations
    """
    # Add family info to each model
    for model_config in models.values():
        model_config["family"] = family_name
    
    # Add to registries
    MODEL_FAMILIES[family_name] = models
    AVAILABLE_MODELS.update(models)


def run_inference(
    model_name: str,
    image_path: Union[str, Path],
    text_prompt: str,
    output_dir: str = "./data/outputs",
    question_data: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run inference with specified model.
    
    Args:
        model_name: Name of model to use (e.g., "luma-ray-2", "luma-ray-flash-2")
        image_path: Path to input image
        text_prompt: Text instructions for video generation
        output_dir: Directory to save outputs
        question_data: Optional question metadata including final_image_path
        **kwargs: Additional model-specific parameters
        
    Returns:
        Dictionary with inference results
    """
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(AVAILABLE_MODELS.keys())}"
        )
    
    model_config = AVAILABLE_MODELS[model_name]
    model_class = model_config["class"]
    
    # Create structured output directory for this inference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    inference_id = kwargs.pop('inference_id', f"{model_name}_{timestamp}")  # Remove from kwargs
    inference_dir = Path(output_dir) / inference_id
    video_dir = inference_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Create model instance - no API key needed! Models handle their own config
    init_kwargs = {
        "model": model_config["model"],
        "output_dir": str(video_dir),  # Save video to the video subdirectory
    }

    model = model_class(**init_kwargs)
    
    # Run inference - clean kwargs, no filtering needed!
    result = model.generate(image_path, text_prompt, **kwargs)
    
    # Add structured output directory to result
    result["inference_dir"] = str(inference_dir)
    result["question_data"] = question_data
    
    return result


class InferenceRunner:
    """
    Enhanced inference runner for managing video generation with structured output folders.
    
    Each inference creates a self-contained folder with:
    - video/: Generated video file(s)
    - question/: Input images and prompt
    - metadata.json: Complete inference metadata
    """
    
    def __init__(self, output_dir: str = "./data/outputs"):
        """
        Initialize runner.
        
        Args:
            output_dir: Directory to save generated videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Logging disabled: keep in-memory only
        self.runs = []
    
    def run(
        self,
        model_name: str,
        image_path: Union[str, Path],
        text_prompt: str,
        run_id: Optional[str] = None,
        question_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference and create structured output folder.
        
        Args:
            model_name: Model to use
            image_path: Input image (first frame)
            text_prompt: Text instructions
            run_id: Optional run identifier
            question_data: Optional question metadata including final_image_path
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results
        """
        start_time = datetime.now()
        
        # Generate run ID if not provided
        if not run_id:
            question_id = question_data.get('id', 'unknown') if question_data else 'unknown'
            run_id = f"{model_name}_{question_id}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        # Create structured output directory mirroring questions structure:
        # <model_base>/<domain>_task/<task_id>/<run_id>/
        domain = None
        task_id = "unknown"
        if question_data:
            domain = question_data.get("domain") or question_data.get("task_category")
            task_id = question_data.get("id", task_id)
        domain_dir_name = f"{domain}_task" if domain else "unknown_task"

        task_base_dir = self.output_dir / domain_dir_name / task_id
        inference_dir = task_base_dir / run_id
        inference_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Run inference with inference_id for structured output
            result = run_inference(
                model_name=model_name,
                image_path=image_path,
                text_prompt=text_prompt,
                # Ensure inner runner also writes into the mirrored task directory
                output_dir=str(task_base_dir),
                question_data=question_data,
                inference_id=run_id,
                **kwargs  # Clean! No filtering needed
            )
            
            # Add metadata
            result["run_id"] = run_id
            result["timestamp"] = start_time.isoformat()
            
            # Create question folder and copy images
            self._setup_question_folder(inference_dir, image_path, text_prompt, question_data)
            
            # Save metadata
            self._save_metadata(inference_dir, result, question_data)
            
            # Logging disabled
            
            print(f"\n✅ Inference complete! Output saved to: {inference_dir}")
            print(f"   - Video: {inference_dir}/video/")
            print(f"   - Question data: {inference_dir}/question/")
            print(f"   - Metadata: {inference_dir}/metadata.json")
            
            return result
            
        except Exception as e:
            # Log failure
            error_result = {
                "run_id": run_id,
                "status": "failed",
                "error": str(e),
                "model": model_name,
                "timestamp": start_time.isoformat(),
                "inference_dir": str(inference_dir)
            }
            
            # Save error metadata
            self._save_metadata(inference_dir, error_result, question_data)
            # Logging disabled
            
            # Clean up folder if no video was generated
            self._cleanup_failed_folder(inference_dir)
            
            print(f"\n❌ Inference failed: {e}")
            print(f"   Folder cleaned up: {inference_dir}")
            
            return error_result
    
    def _setup_question_folder(self, inference_dir: Path, first_image: Union[str, Path], 
                               prompt: str, question_data: Optional[Dict[str, Any]]):
        """
        Create question folder with input images and prompt.
        
        Args:
            inference_dir: Directory for this inference
            first_image: Path to first image (input to model)
            prompt: Text prompt
            question_data: Optional question metadata
        """
        question_dir = inference_dir / "question"
        question_dir.mkdir(exist_ok=True)
        
        # Copy first image
        first_image_path = Path(first_image)
        if first_image_path.exists():
            dest_first = question_dir / f"first_frame{first_image_path.suffix}"
            shutil.copy2(first_image_path, dest_first)
        
        # Copy final image if available
        if question_data and 'final_image_path' in question_data:
            final_image_path = Path(question_data['final_image_path'])
            if final_image_path.exists():
                dest_final = question_dir / f"final_frame{final_image_path.suffix}"
                shutil.copy2(final_image_path, dest_final)
        
        # Save prompt to text file
        prompt_file = question_dir / "prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(prompt)
        
        # Save question metadata if available
        if question_data:
            question_metadata_file = question_dir / "question_metadata.json"
            with open(question_metadata_file, 'w') as f:
                json.dump(question_data, f, indent=2)
    
    def _save_metadata(self, inference_dir: Path, result: Dict[str, Any], 
                      question_data: Optional[Dict[str, Any]]):
        """
        Save complete metadata for the inference.
        
        Args:
            inference_dir: Directory for this inference
            result: Inference result
            question_data: Optional question metadata
        """
        metadata = {
            "inference": {
                "run_id": result.get("run_id"),
                "model": result.get("model"),
                "timestamp": result.get("timestamp"),
                "status": result.get("status", "unknown"),
                "duration_seconds": result.get("duration_seconds"),
                "error": result.get("error")
            },
            "input": {
                "prompt": result.get("prompt"),
                "image_path": result.get("image_path"),
                "question_id": question_data.get("id") if question_data else None,
                "task_category": question_data.get("task_category") if question_data else None
            },
            "output": {
                "video_path": result.get("video_path"),
                "video_url": result.get("video_url"),
                "generation_id": result.get("generation_id")
            },
            "paths": {
                "inference_dir": str(inference_dir),
                "video_dir": str(inference_dir / "video"),
                "question_dir": str(inference_dir / "question")
            },
            "question_data": question_data
        }
        
        # Remove None values for cleaner output
        metadata = self._remove_none_values(metadata)
        
        metadata_file = inference_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _remove_none_values(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively remove None values from a dictionary.
        
        Args:
            d: Dictionary to clean
            
        Returns:
            Dictionary without None values
        """
        if not isinstance(d, dict):
            return d
        
        clean = {}
        for key, value in d.items():
            if value is not None:
                if isinstance(value, dict):
                    nested = self._remove_none_values(value)
                    if nested:  # Only include non-empty dicts
                        clean[key] = nested
                else:
                    clean[key] = value
        return clean
    
    def _cleanup_failed_folder(self, inference_dir: Path):
        """
        Clean up folder if video generation failed.
        Only removes folder if video directory is empty or missing.
        
        Args:
            inference_dir: Path to inference directory
        """
        import shutil
        
        video_dir = inference_dir / "video"
        
        # Check if video directory exists and has content
        if video_dir.exists():
            video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm"))
            if video_files:
                # Keep folder if it has video files
                return
        
        # Remove the entire inference directory if no videos were generated
        if inference_dir.exists():
            shutil.rmtree(inference_dir)
            print(f"   Cleaned up empty folder: {inference_dir.name}")
    
    def _cleanup_all_failed_experiments(self, dry_run: bool = True) -> List[Path]:
        """
        Internal utility to clean up all failed experiment folders (those without video files).
        This is only needed for cleaning up old folders from before automatic cleanup was implemented.
        New failures are cleaned up automatically.
        
        Args:
            dry_run: If True, only list folders that would be deleted without actually deleting
            
        Returns:
            List of cleaned up folder paths
        """
        import shutil
        
        cleaned_folders = []
        
        # Check all subdirectories in output_dir
        for inference_dir in self.output_dir.iterdir():
            if not inference_dir.is_dir():
                continue
            
            # Skip special directories
            if inference_dir.name in ['logs', 'checkpoints', '.git']:
                continue
            
            video_dir = inference_dir / "video"
            
            # Check if this is an incomplete folder
            is_incomplete = False
            
            if not video_dir.exists():
                # No video directory at all
                is_incomplete = True
            else:
                # Check if video directory has any video files
                video_files = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.webm"))
                if not video_files:
                    is_incomplete = True
            
            if is_incomplete:
                cleaned_folders.append(inference_dir)
                if dry_run:
                    print(f"Would delete: {inference_dir.name}")
                else:
                    shutil.rmtree(inference_dir)
                    print(f"Deleted: {inference_dir.name}")
        
        if cleaned_folders:
            print(f"\n{'Would clean' if dry_run else 'Cleaned'} {len(cleaned_folders)} empty folders")
        else:
            print("\nNo empty folders found")
        
        return cleaned_folders
    
    def list_models(self) -> Dict[str, str]:
        """List available models and their descriptions."""
        return {
            name: config["description"]
            for name, config in AVAILABLE_MODELS.items()
        }
    
    def list_models_by_family(self) -> Dict[str, Dict[str, str]]:
        """List models organized by family."""
        return {
            family_name: {
                name: config["description"]
                for name, config in family_models.items()
            }
            for family_name, family_models in MODEL_FAMILIES.items()
        }
    
    def get_model_families(self) -> Dict[str, int]:
        """Get model family statistics."""
        return {
            family_name: len(family_models)
            for family_name, family_models in MODEL_FAMILIES.items()
        }
    
    def _load_log(self) -> list:
        """Load existing run log (disabled)."""
        return []
    
    def _log_run(self, run_id: str, result: Dict[str, Any]):
        """Log a run (disabled)."""
        return
