"""
Video generation models for VMEvalKit.
"""

# Commercial API services
from .luma_inference import LumaInference, LumaWrapper, generate_video as luma_generate
from .veo_inference import VeoService, VeoWrapper
from .wavespeed_inference import WaveSpeedService, Veo31Service, WaveSpeedModel, WaveSpeedWrapper, Veo31Wrapper, Veo31FastWrapper
from .runway_inference import RunwayService, RunwayWrapper
from .openai_inference import SoraService, OpenAIWrapper

# Open-source models (submodules)
from .ltx_inference import LTXVideoService, LTXVideoWrapper
from .hunyuan_inference import HunyuanVideoService, HunyuanVideoWrapper
from .videocrafter_inference import VideoCrafterService, VideoCrafterWrapper
from .dynamicrafter_inference import DynamiCrafterService, DynamiCrafterWrapper
from .morphic_inference import MorphicFramesToVideoWrapper, MorphicConfigurationError

__all__ = [
    # Commercial API services
    "LumaInference", "LumaWrapper", "luma_generate", 
    "VeoService", "VeoWrapper", 
    "WaveSpeedService", "Veo31Service", "WaveSpeedModel", "WaveSpeedWrapper", "Veo31Wrapper", "Veo31FastWrapper",
    "RunwayService", "RunwayWrapper", 
    "SoraService", "OpenAIWrapper",
    
    # Open-source models
    "LTXVideoService", "LTXVideoWrapper",
    "HunyuanVideoService", "HunyuanVideoWrapper", 
    "VideoCrafterService", "VideoCrafterWrapper",
    "DynamiCrafterService", "DynamiCrafterWrapper",
    "MorphicFramesToVideoWrapper", "MorphicConfigurationError"
]
