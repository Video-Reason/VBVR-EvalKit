#!/usr/bin/env python3
"""
VideoThinkBench Visual Puzzles Task for VMEvalKit

Downloads Visual Puzzles tasks from HuggingFace.

Author: VMEvalKit Team
"""

from typing import Dict, Any
from ..utils import create_videothinkbench_dataset


def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Download Visual Puzzles dataset from HuggingFace.
    
    Args:
        num_samples: Not used for HuggingFace downloads (downloads all available)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    return create_videothinkbench_dataset(
        subset_name='Visual_Puzzles',
        task_id_prefix='visual_puzzles',
        domain='visual_puzzles',
        display_name='Visual Puzzles',
        num_samples=num_samples
    )

