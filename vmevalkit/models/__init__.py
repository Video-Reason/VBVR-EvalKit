"""Lazy import to avoid loading all dependencies."""

import importlib

__all__ = [
    # Commercial API services
    "LumaInference", "LumaWrapper", "luma_generate", 
    "VeoService", "VeoWrapper", 
    "RunwayService", "RunwayWrapper", 
    "SoraService", "OpenAIWrapper",
    
    # Open-source models
    "LTXVideoService", "LTXVideoWrapper",
    "HunyuanVideoService", "HunyuanVideoWrapper", 
    "VideoCrafterService", "VideoCrafterWrapper",
    "DynamiCrafterService", "DynamiCrafterWrapper",
    "MorphicService", "MorphicWrapper",
    "SVDService", "SVDWrapper",
    "WanService", "WanWrapper",
    "CogVideoXService", "CogVideoXWrapper",
    "SanaVideoService", "SanaVideoWrapper"
]

# Module name mapping
_MODULE_MAP = {
    "luma_inference": ["LumaInference", "LumaWrapper", "luma_generate"],
    "ltx_inference": ["LTXVideoService", "LTXVideoWrapper"],
    "hunyuan_inference": ["HunyuanVideoService", "HunyuanVideoWrapper"],
    "veo_inference": ["VeoService", "VeoWrapper"],
    "runway_inference": ["RunwayService", "RunwayWrapper"],
    "openai_inference": ["SoraService", "OpenAIWrapper"],
    "videocrafter_inference": ["VideoCrafterService", "VideoCrafterWrapper"],
    "dynamicrafter_inference": ["DynamiCrafterService", "DynamiCrafterWrapper"],
    "morphic_inference": ["MorphicService", "MorphicWrapper"],
    "svd_inference": ["SVDService", "SVDWrapper"],
    "wan_inference": ["WanService", "WanWrapper"],
    "cogvideox_inference": ["CogVideoXService", "CogVideoXWrapper"],
    "sana_inference": ["SanaVideoService", "SanaVideoWrapper"],
}


def __getattr__(name: str):
    """Lazy import of modules to avoid loading all dependencies at once."""
    for module_name, attrs in _MODULE_MAP.items():
        if name in attrs:
            module = importlib.import_module(f".{module_name}", package=__name__)
            return getattr(module, name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
