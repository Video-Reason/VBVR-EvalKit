"""
VMEvalKit Inference Runner - Multi-Provider Video Generation

Unified interface for 37+ text+image→video models across 9 major providers:
- Luma Dream Machine (2 models)
- Google Veo (3 models) 
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
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
import json

from ..models import (
    LumaInference, VeoService, WaveSpeedService, RunwayService, SoraService,
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
        api_key: Optional[str] = None,  # Not used for Veo (uses GCP auth)
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
        import time
        start_time = time.time()
        
        # Convert duration to int (Veo requires int)
        duration_seconds = int(duration)
        
        # Run async generation in sync context
        video_bytes, metadata = asyncio.run(
            self.veo_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                duration_seconds=duration_seconds,
                **kwargs
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
            "video_path": str(video_path) if video_path else None,
            "generation_id": metadata.get('operation_name', 'unknown'),
            "status": "success" if video_bytes else "completed_no_download",
            "duration_seconds": duration_taken,
            "model": self.model,
            "prompt": text_prompt,
            "image_path": str(image_path),
            "metadata": metadata
        }


class WaveSpeedWrapper:
    """
    Wrapper for WaveSpeedService to match the LumaInference interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        api_key: Optional[str] = None,  # Not used - WaveSpeed uses WAVESPEED_API_KEY env var
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
        result = asyncio.run(
            self.wavespeed_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                seed=seed,
                output_path=output_path,
                **kwargs
            )
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "video_path": result.get("video_path"),
            "generation_id": result.get("request_id", 'unknown'),
            "status": "success" if result.get("video_path") else "completed_no_download",
            "duration_seconds": duration_taken,
            "model": self.model,
            "prompt": text_prompt,
            "image_path": str(image_path),
            "video_url": result.get("video_url"),
            "seed": seed
        }


class RunwayWrapper:
    """
    Wrapper for RunwayService to match the LumaInference interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        api_key: Optional[str] = None,  # Not used - Runway uses RUNWAYML_API_SECRET env var
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
        result = asyncio.run(
            self.runway_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                duration=duration_int,
                ratio=ratio,
                output_path=output_path,
                **kwargs
            )
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "video_path": result.get("video_path"),
            "generation_id": result.get("task_id", 'unknown'),
            "status": "success" if result.get("video_path") else "completed_no_download",
            "duration_seconds": duration_taken,
            "model": self.model,
            "prompt": text_prompt,
            "image_path": str(image_path),
            "video_url": result.get("video_url"),
            "duration": duration_int,
            "ratio": result.get("ratio")
        }


class OpenAIWrapper:
    """
    Wrapper for SoraService to match the LumaInference interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        api_key: Optional[str] = None,  # Not used - Sora uses OPENAI_API_KEY env var
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
        result = asyncio.run(
            self.sora_service.generate_video(
                prompt=text_prompt,
                image_path=str(image_path),
                duration=duration_int,
                size=size,
                output_path=output_path,
                auto_pad=True,  # Enable auto-padding by default
                **kwargs
            )
        )
        
        duration_taken = time.time() - start_time
        
        return {
            "video_path": result.get("video_path"),
            "generation_id": result.get("video_id", 'unknown'),
            "status": "success" if result.get("video_path") else "completed_no_download",
            "duration_seconds": duration_taken,
            "model": self.model,
            "prompt": text_prompt,
            "image_path": str(image_path),
            "duration": duration_int,
            "size": result.get("size")
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
    api_key: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Run inference with specified model.
    
    Args:
        model_name: Name of model to use (e.g., "luma-ray-2", "luma-ray-flash-2")
        image_path: Path to input image
        text_prompt: Text instructions for video generation
        output_dir: Directory to save outputs
        api_key: Optional API key (uses env var if not provided)
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
    
    # Create model instance with only constructor-safe parameters
    init_kwargs = {
        "model": model_config["model"],
        "output_dir": output_dir,
    }
    if api_key is not None:
        init_kwargs["api_key"] = api_key

    model = model_class(**init_kwargs)
    
    # Run inference, forwarding runtime options (e.g., output_filename) to generate
    return model.generate(image_path, text_prompt, **kwargs)


class InferenceRunner:
    """
    Simple inference runner for managing video generation.
    """
    
    def __init__(self, output_dir: str = "./data/outputs"):
        """
        Initialize runner.
        
        Args:
            output_dir: Directory to save generated videos
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Simple logging to track runs
        self.log_file = self.output_dir / "inference_log.json"
        self.runs = self._load_log()
    
    def run(
        self,
        model_name: str,
        image_path: Union[str, Path],
        text_prompt: str,
        run_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference and log results.
        
        Args:
            model_name: Model to use
            image_path: Input image
            text_prompt: Text instructions
            run_id: Optional run identifier
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with results
        """
        start_time = datetime.now()
        
        # Generate run ID if not provided
        if not run_id:
            run_id = f"{model_name}_{start_time.strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Run inference
            result = run_inference(
                model_name=model_name,
                image_path=image_path,
                text_prompt=text_prompt,
                output_dir=self.output_dir,
                **kwargs
            )
            
            # Add metadata
            result["run_id"] = run_id
            result["timestamp"] = start_time.isoformat()
            
            # Log the run
            self._log_run(run_id, result)
            
            return result
            
        except Exception as e:
            # Log failure
            error_result = {
                "run_id": run_id,
                "status": "failed",
                "error": str(e),
                "model": model_name,
                "timestamp": start_time.isoformat()
            }
            self._log_run(run_id, error_result)
            return error_result
    
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
        """Load existing run log."""
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                return json.load(f)
        return []
    
    def _log_run(self, run_id: str, result: Dict[str, Any]):
        """Log a run to the log file."""
        self.runs.append({
            "run_id": run_id,
            **result
        })
        
        with open(self.log_file, 'w') as f:
            json.dump(self.runs, f, indent=2)
