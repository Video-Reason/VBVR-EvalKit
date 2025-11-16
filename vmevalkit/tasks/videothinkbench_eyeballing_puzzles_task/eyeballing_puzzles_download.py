#!/usr/bin/env python3
"""
VideoThinkBench Eyeballing Puzzles Task for VMEvalKit

Downloads Eyeballing Puzzles tasks from HuggingFace.

Author: VMEvalKit Team
"""

from typing import Dict, Any, List


def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Download Eyeballing Puzzles dataset from HuggingFace.
    
    Args:
        num_samples: Not used for HuggingFace downloads (downloads all available)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    from datasets import load_dataset
    
    print(f"📥 Downloading Eyeballing Puzzles tasks from HuggingFace...")
    
    dataset = load_dataset('OpenMOSS-Team/VideoThinkBench', 'Eyeballing_Puzzles', split='test')
    
    pairs = []
    for idx, item in enumerate(dataset):
        task_id = f"eyeballing_puzzles_{idx:04d}"
        
        prompt = item.get('prompt', '')
        first_image = item.get('image')
        solution_image = item.get('solution_image')
        
        if not prompt or first_image is None:
            continue
            
        pair = {
            'id': task_id,
            'domain': 'eyeballing_puzzles',
            'prompt': prompt,
            'first_image': first_image,
            'solution_image': solution_image,
        }
        
        pairs.append(pair)
    
    print(f"   ✅ Downloaded {len(pairs)} Eyeballing Puzzles tasks")
    
    return {
        'name': 'eyeballing_puzzles',
        'pairs': pairs,
        'source': 'huggingface',
        'hf_dataset': 'OpenMOSS-Team/VideoThinkBench',
        'hf_subset': 'Eyeballing_Puzzles'
    }

