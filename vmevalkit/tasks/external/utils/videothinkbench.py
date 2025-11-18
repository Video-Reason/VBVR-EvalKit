#!/usr/bin/env python3
"""
Shared utilities for VideoThinkBench tasks.

This module provides common functionality for downloading and processing
VideoThinkBench subsets from HuggingFace.

Author: VMEvalKit Team
"""

from typing import Dict, Any


def create_videothinkbench_dataset(
    subset_name: str,
    task_id_prefix: str,
    domain: str,
    display_name: str,
    num_samples: int = None
) -> Dict[str, Any]:
    """
    Download a VideoThinkBench subset from HuggingFace.
    
    This is a shared utility function used by all VideoThinkBench task loaders
    to avoid code duplication.
    
    Args:
        subset_name: HuggingFace subset name (e.g., 'ARC_AGI_2', 'Visual_Puzzles')
        task_id_prefix: Prefix for task IDs (e.g., 'arc_agi_2', 'visual_puzzles')
        domain: Domain identifier (same as task_id_prefix)
        display_name: Human-readable name for logging (e.g., 'ARC AGI 2')
        num_samples: Not used for HuggingFace downloads (downloads all available)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    from datasets import load_dataset
    
    print(f"📥 Downloading {display_name} from HuggingFace...")
    
    dataset = load_dataset('OpenMOSS-Team/VideoThinkBench', subset_name, split='test')
    
    pairs = []
    for idx, item in enumerate(dataset):
        task_id = f"{task_id_prefix}_{idx:04d}"
        
        prompt = item.get('prompt', '')
        first_image = item.get('image')
        solution_image = item.get('solution_image')
        
        if not prompt or first_image is None:
            continue
            
        pair = {
            'id': task_id,
            'domain': domain,
            'prompt': prompt,
            'first_image': first_image,
            'solution_image': solution_image,
        }
        
        pairs.append(pair)
    
    print(f"   ✅ Downloaded {len(pairs)} {display_name} tasks")
    
    return {
        'name': domain,
        'pairs': pairs,
        'source': 'huggingface',
        'hf_dataset': 'OpenMOSS-Team/VideoThinkBench',
        'hf_subset': subset_name
    }

