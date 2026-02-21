"""
VBVR-EvalKit - A unified framework for video model inference and evaluation.

This framework provides a unified interface for 40+ video generation models 
and comprehensive evaluation pipelines for reasoning capabilities.

Focus: Inference and evaluation only. Data generation is handled externally.
"""

__version__ = "0.1.0"

__all__ = [
    # Runner
    "run_inference", 
    "InferenceRunner",
    # Models
    "LumaInference",
    "luma_generate",
    "__version__"
]


def __getattr__(name: str):
    """Lazy import of modules to avoid loading all dependencies at once."""
    if name == "run_inference" or name == "InferenceRunner":
        from .runner import run_inference, InferenceRunner
        return locals()[name]
    
    elif name == "LumaInference" or name == "luma_generate":
        from .models import LumaInference, luma_generate
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")