#!/usr/bin/env python3
"""
VideoThinkBench Text Centric Tasks for VMEvalKit

Downloads Text Centric Tasks from HuggingFace.

Author: VMEvalKit Team
"""

from typing import Dict, Any
from ..utils import create_videothinkbench_dataset


def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Download Text Centric Tasks dataset from HuggingFace.
    
    Args:
        num_samples: Not used for HuggingFace downloads (downloads all available)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    return create_videothinkbench_dataset(
        subset_name='Text_Centric_Tasks',
        task_id_prefix='text_centric_tasks',
        domain='text_centric_tasks',
        display_name='Text Centric Tasks',
        num_samples=num_samples
    )

