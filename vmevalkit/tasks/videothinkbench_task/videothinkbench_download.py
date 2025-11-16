#!/usr/bin/env python3
"""
VideoThinkBench Meta-Task for VMEvalKit

Downloads all VideoThinkBench subsets from HuggingFace.

Author: VMEvalKit Team
"""

from typing import Dict, Any, List


def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Download all VideoThinkBench subsets from HuggingFace.
    
    This is a meta-task that downloads all subsets:
    - ARC AGI 2
    - Eyeballing Puzzles
    - Visual Puzzles
    - Mazes
    - Text Centric Tasks
    
    Args:
        num_samples: Not used for HuggingFace downloads (downloads all available)
        
    Returns:
        Dataset dictionary with 'pairs' key containing all task data
    """
    from datasets import load_dataset
    
    print(f"📥 Downloading all VideoThinkBench tasks from HuggingFace...")
    print(f"   This includes all subsets: ARC AGI 2, Eyeballing Puzzles, Visual Puzzles, Mazes, Text Centric Tasks")
    
    subsets = [
        ('ARC_AGI_2', 'arc_agi_2'),
        ('Eyeballing_Puzzles', 'eyeballing_puzzles'),
        ('Visual_Puzzles', 'visual_puzzles'),
        ('Mazes', 'mazes'),
        ('Text_Centric_Tasks', 'text_centric_tasks')
    ]
    
    all_pairs = []
    
    for hf_subset_name, domain_name in subsets:
        print(f"\n   📦 Loading {hf_subset_name}...")
        dataset = load_dataset('OpenMOSS-Team/VideoThinkBench', hf_subset_name, split='test')
        
        for idx, item in enumerate(dataset):
            task_id = f"{domain_name}_{idx:04d}"
            
            prompt = item.get('prompt', '')
            first_image = item.get('image')
            solution_image = item.get('solution_image')
            
            if not prompt or first_image is None:
                continue
                
            pair = {
                'id': task_id,
                'domain': domain_name,
                'prompt': prompt,
                'first_image': first_image,
                'solution_image': solution_image,
            }
            
            all_pairs.append(pair)
        
        print(f"      ✅ Loaded {len([p for p in all_pairs if p['domain'] == domain_name])} tasks from {hf_subset_name}")
    
    print(f"\n   ✅ Total: Downloaded {len(all_pairs)} tasks from all VideoThinkBench subsets")
    
    return {
        'name': 'videothinkbench',
        'pairs': all_pairs,
        'source': 'huggingface',
        'hf_dataset': 'OpenMOSS-Team/VideoThinkBench',
        'subsets': [s[1] for s in subsets]
    }

