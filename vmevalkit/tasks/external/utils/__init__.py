"""
Shared utilities for external task loaders.

This module provides common functionality for downloading and processing
external datasets from various sources (HuggingFace, etc.).
"""

from .videothinkbench import create_videothinkbench_dataset

__all__ = ['create_videothinkbench_dataset']

