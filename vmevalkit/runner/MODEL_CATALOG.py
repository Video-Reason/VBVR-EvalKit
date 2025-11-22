"""
Model Catalog for VMEvalKit - Registry of all available video generation models.

Pure registry with no imports or logic - just model definitions organized by families.
Uses string module paths for flexible dynamic loading.
"""

from typing import Dict, Any


# ========================================
# COMMERCIAL MODELS
# ========================================

# Luma Dream Machine Models
LUMA_MODELS = {
    "luma-ray-2": {
        "wrapper_module": "vmevalkit.models.luma_inference",
        "wrapper_class": "LumaWrapper",
        "service_class": "LumaInference",
        "model": "ray-2",
        "description": "Luma Ray 2 - Latest model with best quality",
        "family": "Luma Dream Machine"
    },
    "luma-ray-flash-2": {
        "wrapper_module": "vmevalkit.models.luma_inference",
        "wrapper_class": "LumaWrapper",
        "service_class": "LumaInference",
        "model": "ray-flash-2", 
        "description": "Luma Ray Flash 2 - Faster generation",
        "family": "Luma Dream Machine"
    }
}

# Google Veo Models (Vertex AI)
VEO_MODELS = {
    "veo-2.0-generate": {
        "wrapper_module": "vmevalkit.models.veo_inference",
        "wrapper_class": "VeoWrapper",
        "service_class": "VeoService",
        "model": "veo-2.0-generate-001",
        "description": "Google Veo 2.0 - GA model for text+image→video",
        "family": "Google Veo"
    },
    "veo-3.0-generate": {
        "wrapper_module": "vmevalkit.models.veo_inference",
        "wrapper_class": "VeoWrapper",
        "service_class": "VeoService",
        "model": "veo-3.0-generate-preview",
        "description": "Google Veo 3.0 - Preview model with advanced capabilities",
        "family": "Google Veo"
    },
    "veo-3.0-fast-generate": {
        "wrapper_module": "vmevalkit.models.veo_inference",
        "wrapper_class": "VeoWrapper",
        "service_class": "VeoService",
        "model": "veo-3.0-fast-generate-preview",
        "description": "Google Veo 3.0 Fast - Preview model for faster generation",
        "family": "Google Veo"
    }
}

# Google Veo 3.1 Models (via WaveSpeed)
GOOGLE_VEO31_MODELS = {
    "veo-3.1": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "Veo31Wrapper",
        "service_class": "Veo31Service",
        "model": "veo-3.1",
        "args": {},
        "description": "Google Veo 3.1 - Native 1080p with audio generation (via WaveSpeed)",
        "family": "Google Veo 3.1"
    },
    "veo-3.1-720p": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "Veo31Wrapper",
        "service_class": "Veo31Service",
        "model": "veo-3.1-720p",
        "args": {"resolution": "720p"},
        "description": "Google Veo 3.1 - 720p with audio generation (via WaveSpeed)",
        "family": "Google Veo 3.1"
    },
    "veo-3.1-fast": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "Veo31FastWrapper",
        "service_class": "WaveSpeedService", 
        "model": "veo-3.1-fast",
        "args": {},
        "description": "Google Veo 3.1 Fast - 1080p, 30% faster generation (via WaveSpeed)",
        "family": "Google Veo 3.1"
    },
    "veo-3.1-fast-720p": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "Veo31FastWrapper",
        "service_class": "WaveSpeedService",
        "model": "veo-3.1-fast-720p",
        "args": {"resolution": "720p"},
        "description": "Google Veo 3.1 Fast - 720p, 30% faster generation (via WaveSpeed)",
        "family": "Google Veo 3.1"
    }
}

# WaveSpeedAI WAN 2.2 Models
WAVESPEED_WAN_22_MODELS = {
    "wavespeed-wan-2.2-i2v-480p": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.2/i2v-480p",
        "description": "WaveSpeed WAN 2.2 I2V 480p - Standard quality",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-480p-ultra-fast": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.2/i2v-480p-ultra-fast",
        "description": "WaveSpeed WAN 2.2 I2V 480p Ultra Fast - Speed optimized",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-480p-lora": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.2/i2v-480p-lora",
        "description": "WaveSpeed WAN 2.2 I2V 480p LoRA - Enhanced with LoRA",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-480p-lora-ultra-fast": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.2/i2v-480p-lora-ultra-fast",
        "description": "WaveSpeed WAN 2.2 I2V 480p LoRA Ultra Fast - Best speed with LoRA",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-5b-720p": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.2/i2v-5b-720p",
        "description": "WaveSpeed WAN 2.2 I2V 5B 720p - High resolution 5B model",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-5b-720p-lora": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.2/i2v-5b-720p-lora",
        "description": "WaveSpeed WAN 2.2 I2V 5B 720p LoRA - High-res with LoRA",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-720p": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.2/i2v-720p",
        "description": "WaveSpeed WAN 2.2 I2V 720p - High resolution",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-720p-ultra-fast": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.2/i2v-720p-ultra-fast",
        "description": "WaveSpeed WAN 2.2 I2V 720p Ultra Fast - High-res speed optimized",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-720p-lora": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.2/i2v-720p-lora",
        "description": "WaveSpeed WAN 2.2 I2V 720p LoRA - High-res with LoRA",
        "family": "WaveSpeed WAN 2.2"
    },
    "wavespeed-wan-2.2-i2v-720p-lora-ultra-fast": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.2/i2v-720p-lora-ultra-fast",
        "description": "WaveSpeed WAN 2.2 I2V 720p LoRA Ultra Fast - Fastest high-res LoRA",
        "family": "WaveSpeed WAN 2.2"
    }
}

# WaveSpeedAI WAN 2.1 Models
WAVESPEED_WAN_21_MODELS = {
    "wavespeed-wan-2.1-i2v-480p": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.1/i2v-480p",
        "description": "WaveSpeed WAN 2.1 I2V 480p - Standard quality",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-480p-ultra-fast": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.1/i2v-480p-ultra-fast",
        "description": "WaveSpeed WAN 2.1 I2V 480p Ultra Fast - Speed optimized",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-480p-lora": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.1/i2v-480p-lora",
        "description": "WaveSpeed WAN 2.1 I2V 480p LoRA - Enhanced with LoRA",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-480p-lora-ultra-fast": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.1/i2v-480p-lora-ultra-fast",
        "description": "WaveSpeed WAN 2.1 I2V 480p LoRA Ultra Fast - Best speed with LoRA",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-720p": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.1/i2v-720p",
        "description": "WaveSpeed WAN 2.1 I2V 720p - High resolution",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-720p-ultra-fast": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.1/i2v-720p-ultra-fast",
        "description": "WaveSpeed WAN 2.1 I2V 720p Ultra Fast - High-res speed optimized",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-720p-lora": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.1/i2v-720p-lora",
        "description": "WaveSpeed WAN 2.1 I2V 720p LoRA - High-res with LoRA",
        "family": "WaveSpeed WAN 2.1"
    },
    "wavespeed-wan-2.1-i2v-720p-lora-ultra-fast": {
        "wrapper_module": "vmevalkit.models.wavespeed_inference",
        "wrapper_class": "WaveSpeedWrapper",
        "service_class": "WaveSpeedService",
        "model": "wan-2.1/i2v-720p-lora-ultra-fast",
        "description": "WaveSpeed WAN 2.1 I2V 720p LoRA Ultra Fast - Fastest high-res LoRA",
        "family": "WaveSpeed WAN 2.1"
    }
}

# Runway ML Models
RUNWAY_MODELS = {
    "runway-gen4-turbo": {
        "wrapper_module": "vmevalkit.models.runway_inference",
        "wrapper_class": "RunwayWrapper",
        "service_class": "RunwayService",
        "model": "gen4_turbo",
        "description": "Runway Gen-4 Turbo - Fast high-quality generation (5s or 10s)",
        "family": "Runway ML"
    },
    "runway-gen4-aleph": {
        "wrapper_module": "vmevalkit.models.runway_inference",
        "wrapper_class": "RunwayWrapper",
        "service_class": "RunwayService",
        "model": "gen4_aleph",
        "description": "Runway Gen-4 Aleph - Premium quality (5s)",
        "family": "Runway ML"
    },
    "runway-gen3a-turbo": {
        "wrapper_module": "vmevalkit.models.runway_inference",
        "wrapper_class": "RunwayWrapper",
        "service_class": "RunwayService",
        "model": "gen3a_turbo",
        "description": "Runway Gen-3A Turbo - Proven performance (5s or 10s)",
        "family": "Runway ML"
    }
}

# OpenAI Sora Models
OPENAI_SORA_MODELS = {
    "openai-sora-2": {
        "wrapper_module": "vmevalkit.models.openai_inference",
        "wrapper_class": "OpenAIWrapper",
        "service_class": "SoraService",
        "model": "sora-2",
        "description": "OpenAI Sora-2 - High-quality video generation (4s/8s/12s)",
        "family": "OpenAI Sora"
    },
    "openai-sora-2-pro": {
        "wrapper_module": "vmevalkit.models.openai_inference",
        "wrapper_class": "OpenAIWrapper",
        "service_class": "SoraService",
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
    "ltx-video": {
        "wrapper_module": "vmevalkit.models.ltx_inference",
        "wrapper_class": "LTXVideoWrapper",
        "service_class": "LTXVideoService",
        "model": "Lightricks/LTX-Video",
        "description": "LTX-Video - High-quality image-to-video generation (704x480, 24fps)",
        "family": "LTX-Video"
    },
    "ltx-video-13b-distilled": {
        "wrapper_module": "vmevalkit.models.ltx_inference",
        "wrapper_class": "LTXVideoWrapper",
        "service_class": "LTXVideoService",
        "model": "Lightricks/LTX-Video-0.9.8-13B-distilled",
        "description": "LTX-Video 0.9.8 13B Distilled - Distilled version with 13B parameters",
        "family": "LTX-Video"
    }
}

# HunyuanVideo-I2V Models (Tencent)
HUNYUAN_VIDEO_MODELS = {
    "hunyuan-video-i2v": {
        "wrapper_module": "vmevalkit.models.hunyuan_inference",
        "wrapper_class": "HunyuanVideoWrapper",
        "service_class": "HunyuanVideoService",
        "model": "hunyuan-video-i2v",
        "description": "HunyuanVideo-I2V - High-quality image-to-video up to 720p",
        "family": "HunyuanVideo"
    }
}

# VideoCrafter Models (AILab-CVC)
VIDEOCRAFTER_MODELS = {
    "videocrafter2-512": {
        "wrapper_module": "vmevalkit.models.videocrafter_inference",
        "wrapper_class": "VideoCrafterWrapper",
        "service_class": "VideoCrafterService",
        "model": "videocrafter2",
        "description": "VideoCrafter2 - High-quality text-guided video generation",
        "family": "VideoCrafter"
    }
}

# DynamiCrafter Models (Doubiiu)
DYNAMICRAFTER_MODELS = {
    "dynamicrafter-512": {
        "wrapper_module": "vmevalkit.models.dynamicrafter_inference",
        "wrapper_class": "DynamiCrafterWrapper",
        "service_class": "DynamiCrafterService",
        "model": "dynamicrafter-512",
        "description": "DynamiCrafter 512p - Image animation with video diffusion",
        "family": "DynamiCrafter"
    },
    "dynamicrafter-256": {
        "wrapper_module": "vmevalkit.models.dynamicrafter_inference",
        "wrapper_class": "DynamiCrafterWrapper",
        "service_class": "DynamiCrafterService",
        "model": "dynamicrafter-256",
        "description": "DynamiCrafter 256p - Faster image animation",
        "family": "DynamiCrafter"
    },
    "dynamicrafter-1024": {
        "wrapper_module": "vmevalkit.models.dynamicrafter_inference",
        "wrapper_class": "DynamiCrafterWrapper",
        "service_class": "DynamiCrafterService",
        "model": "dynamicrafter-1024",
        "description": "DynamiCrafter 1024p - High-resolution image animation",
        "family": "DynamiCrafter"
    }
}

# Morphic Frames-to-Video Models
MORPHIC_MODELS = {
    "morphic-frames-to-video": {
        "wrapper_module": "vmevalkit.models.morphic_inference",
        "wrapper_class": "MorphicWrapper",
        "service_class": "MorphicService",
        "model": "morphic-frames-to-video",
        "description": "Morphic Frames to Video - High-quality interpolation using Wan2.2",
        "family": "Morphic",
        "args": {
            "size": "1280*720",
            "frame_num": 81,
            "nproc_per_node": 8
        }
    }
}

# Stable Video Diffusion Models (Stability AI)
SVD_MODELS = {
    "svd": {
        "wrapper_module": "vmevalkit.models.svd_inference",
        "wrapper_class": "SVDWrapper",
        "service_class": "SVDService",
        "model": "stabilityai/stable-video-diffusion-img2vid-xt",
        "description": "Stable Video Diffusion XT - High-quality image-to-video generation",
        "family": "Stable Video Diffusion"
    },
}

# WAN Models (Wan-AI, local diffusers implementation)
WAN_MODELS = {
    "wan": {
        "wrapper_module": "vmevalkit.models.wan_inference",
        "wrapper_class": "WanWrapper",
        "service_class": "WanService",
        "model": "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
        "description": "WAN 2.1 FLF2V 14B 720P - First-Last Frame to Video generation",
        "family": "WAN (Wan-AI)"
    },
    "wan-2.1-flf2v-720p": {
        "wrapper_module": "vmevalkit.models.wan_inference",
        "wrapper_class": "WanWrapper",
        "service_class": "WanService",
        "model": "Wan-AI/Wan2.1-FLF2V-14B-720P-diffusers",
        "description": "WAN 2.1 FLF2V 14B 720P - First-Last Frame to Video generation",
        "family": "WAN (Wan-AI)"
    },
    "wan-2.2-i2v-a14b": {
        "wrapper_module": "vmevalkit.models.wan_inference",
        "wrapper_class": "WanWrapper",
        "service_class": "WanService",
        "model": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "description": "WAN 2.2 I2V A14B - Image to Video generation with 14B parameters",
        "family": "WAN (Wan-AI)"
    },
    "wan-2.1-i2v-480p": {
        "wrapper_module": "vmevalkit.models.wan_inference",
        "wrapper_class": "WanWrapper",
        "service_class": "WanService",
        "model": "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
        "description": "WAN 2.1 I2V 14B 480P - Image to Video generation at 480p resolution",
        "family": "WAN (Wan-AI)"
    },
    "wan-2.1-i2v-720p": {
        "wrapper_module": "vmevalkit.models.wan_inference",
        "wrapper_class": "WanWrapper",
        "service_class": "WanService",
        "model": "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
        "description": "WAN 2.1 I2V 14B 720P - Image to Video generation at 720p resolution",
        "family": "WAN (Wan-AI)"
    },
    "wan-2.2-ti2v-5b": {
        "wrapper_module": "vmevalkit.models.wan_inference",
        "wrapper_class": "WanWrapper",
        "service_class": "WanService",
        "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "description": "WAN 2.2 TI2V 5B - Text + Image to Video generation with 5B parameters",
        "family": "WAN (Wan-AI)"
    },
    "wan-2.1-vace-14b": {
        "wrapper_module": "vmevalkit.models.wan_inference",
        "wrapper_class": "WanWrapper",
        "service_class": "WanService",
        "model": "Wan-AI/Wan2.1-VACE-14B-diffusers",
        "description": "WAN 2.1 VACE 14B - Video generation with 14B parameters",
        "family": "WAN (Wan-AI)"
    },
    "wan-2.1-vace-1.3b": {
        "wrapper_module": "vmevalkit.models.wan_inference",
        "wrapper_class": "WanWrapper",
        "service_class": "WanService",
        "model": "Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        "description": "WAN 2.1 VACE 1.3B - Lightweight video generation with 1.3B parameters",
        "family": "WAN (Wan-AI)"
    }
}

# ========================================
# COMBINED REGISTRIES
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
    **DYNAMICRAFTER_MODELS,
    **MORPHIC_MODELS,
    **SVD_MODELS,
    **WAN_MODELS
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
    "DynamiCrafter": DYNAMICRAFTER_MODELS,
    "Morphic": MORPHIC_MODELS,
    "Stable Video Diffusion": SVD_MODELS,
    "WAN (Wan-AI)": WAN_MODELS
}

# ========================================
# CATALOG UTILITY FUNCTIONS
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
