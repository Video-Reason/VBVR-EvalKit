#!/usr/bin/env python3
"""
MME-CoF Task for VMEvalKit

Downloads MME-CoF (Video Chain-of-Frame) tasks from HuggingFace.

Author: VMEvalKit Team
"""

from typing import Dict, Any, List


def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Download MME-CoF dataset from HuggingFace.
    
    MME-CoF provides video chain-of-frame reasoning evaluation across
    16 cognitive domains with 59 tasks.
    
    Args:
        num_samples: Not used for HuggingFace downloads (downloads all available)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    from datasets import load_dataset
    
    print(f"📥 Downloading MME-CoF tasks from HuggingFace...")
    
    dataset = load_dataset('VideoReason/MME-CoF-VMEval', split='train')
    
    pairs = []
    for idx, item in enumerate(dataset):
        task_id = f"mme_cof_{idx:04d}"
        
        prompt = item.get('prompt', '')
        first_image = item.get('image')
        solution_image = item.get('solution_image')
        
        if not prompt or first_image is None:
            continue
            
        pair = {
            'id': task_id,
            'domain': 'mme_cof',
            'prompt': prompt,
            'first_image': first_image,
            'solution_image': solution_image,
        }
        
        pairs.append(pair)
    
    print(f"   ✅ Downloaded {len(pairs)} MME-CoF tasks")
    
    return {
        'name': 'mme_cof',
        'pairs': pairs,
        'source': 'huggingface',
        'hf_dataset': 'VideoReason/MME-CoF-VMEval'
    }

