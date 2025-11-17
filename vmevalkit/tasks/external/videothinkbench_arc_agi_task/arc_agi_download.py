#!/usr/bin/env python3
"""
VideoThinkBench ARC AGI Task for VMEvalKit

Downloads ARC AGI reasoning tasks from HuggingFace.

Author: VMEvalKit Team
"""

from typing import Dict, Any
from ..utils import create_videothinkbench_dataset


def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Download ARC AGI dataset from HuggingFace.
    
    Args:
        num_samples: Not used for HuggingFace downloads (downloads all available)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    return create_videothinkbench_dataset(
        subset_name='ARC_AGI_2',
        task_id_prefix='arc_agi_2',
        domain='arc_agi_2',
        display_name='ARC AGI 2',
        num_samples=num_samples
    )

