#!/usr/bin/env python3
"""
VideoThinkBench Visual Puzzles Task for VMEvalKit

Downloads Visual Puzzles tasks from HuggingFace.

Author: VMEvalKit Team
"""

from typing import Dict, Any, List


def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Download Visual Puzzles dataset from HuggingFace.
    
    Args:
        num_samples: Not used for HuggingFace downloads (downloads all available)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    from datasets import load_dataset
    
    print(f"📥 Downloading Visual Puzzles tasks from HuggingFace...")
    
    dataset = load_dataset('OpenMOSS-Team/VideoThinkBench', 'Visual_Puzzles', split='test')
    
    pairs = []
    for idx, item in enumerate(dataset):
        task_id = f"visual_puzzles_{idx:04d}"
        
        prompt = item.get('prompt', '')
        first_image = item.get('image')
        solution_image = item.get('solution_image')
        
        if not prompt or first_image is None:
            continue
            
        pair = {
            'id': task_id,
            'domain': 'visual_puzzles',
            'prompt': prompt,
            'first_image': first_image,
            'solution_image': solution_image,
        }
        
        pairs.append(pair)
    
    print(f"   ✅ Downloaded {len(pairs)} Visual Puzzles tasks")
    
    return {
        'name': 'visual_puzzles',
        'pairs': pairs,
        'source': 'huggingface',
        'hf_dataset': 'OpenMOSS-Team/VideoThinkBench',
        'hf_subset': 'Visual_Puzzles'
    }

