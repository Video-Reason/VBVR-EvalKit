"""
VMEvalKit - A framework for video generation with text+image inputs.

This framework enables video generation models to process both text prompts 
and image inputs to generate videos.
"""

__version__ = "0.1.0"

# Loader and task system removed as unused

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