#!/usr/bin/env python3
"""
VideoThinkBench Eyeballing Puzzles Task for VMEvalKit

Downloads Eyeballing Puzzles tasks from HuggingFace.

Author: VMEvalKit Team
"""

from typing import Dict, Any
from ..utils import create_videothinkbench_dataset


def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Download Eyeballing Puzzles dataset from HuggingFace.
    
    Args:
        num_samples: Not used for HuggingFace downloads (downloads all available)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    return create_videothinkbench_dataset(
        subset_name='Eyeballing_Puzzles',
        task_id_prefix='eyeballing_puzzles',
        domain='eyeballing_puzzles',
        display_name='Eyeballing Puzzles',
        num_samples=num_samples
    )

