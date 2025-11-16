#!/usr/bin/env python3
"""
VideoThinkBench ARC AGI Task for VMEvalKit

Downloads ARC AGI reasoning tasks from HuggingFace.

Author: VMEvalKit Team
"""

from typing import Dict, Any, List


def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Download ARC AGI dataset from HuggingFace.
    
    Args:
        num_samples: Not used for HuggingFace downloads (downloads all available)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    from datasets import load_dataset
    
    print(f"📥 Downloading ARC AGI tasks from HuggingFace...")
    
    dataset = load_dataset('OpenMOSS-Team/VideoThinkBench', 'ARC_AGI_2', split='test')
    
    pairs = []
    for idx, item in enumerate(dataset):
        task_id = f"arc_agi_2_{idx:04d}"
        
        prompt = item.get('prompt', '')
        first_image = item.get('image')
        solution_image = item.get('solution_image')
        
        if not prompt or first_image is None:
            continue
            
        pair = {
            'id': task_id,
            'domain': 'arc_agi_2',
            'prompt': prompt,
            'first_image': first_image,
            'solution_image': solution_image,
        }
        
        pairs.append(pair)
    
    print(f"   ✅ Downloaded {len(pairs)} ARC AGI tasks")
    
    return {
        'name': 'arc_agi_2',
        'pairs': pairs,
        'source': 'huggingface',
        'hf_dataset': 'OpenMOSS-Team/VideoThinkBench',
        'hf_subset': 'ARC_AGI_2'
    }

