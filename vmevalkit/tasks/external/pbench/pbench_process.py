#!/usr/bin/env python3
"""
PBench Task for VMEvalKit

Downloads PBench (Physics Benchmark) tasks from HuggingFace.
PBench dataset contains physics simulation scenarios.

Author: VMEvalKit Team
"""

import json
import io
from pathlib import Path
from typing import Dict, Any
from datasets import load_dataset
from PIL import Image

def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Download PBench dataset from HuggingFace.
    
    PBench provides physics benchmark tasks with videos and questions.
    Each row can contain multiple QA pairs, generating multiple task pairs.
    
    Args:
        num_samples: Number of samples to process (None for all)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    print(f"📥 Downloading PBench tasks from HuggingFace...")
    
    print(f"   Downloading from: nvidia/PBench")
    # Use datasets library to properly load images
    dataset = load_dataset("nvidia/PBench", split="benchmark")
    
    # Limit samples if specified
    if num_samples:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    
    print(f"   Processing {len(dataset)} PBench rows...")
    
    pairs = []
    for idx, item in enumerate(dataset):
        # Extract fields according to requirements
        prompt = item.get('text_prompt', '')
        condition_image = item.get('condition_image', None)
        qa_pairs = item.get('qa_pairs', [])
        
        # Validate condition_image - must be a PIL Image object
        if condition_image is None:
            print(f"      ⚠️  Warning: condition_image is None for row {idx}, skipping")
            continue
        
        if not isinstance(condition_image, Image.Image):
            print(f"      ⚠️  Warning: condition_image is not a PIL Image for row {idx} (type: {type(condition_image)}), skipping")
            continue
        
        # Ensure RGB mode
        first_image = condition_image
        if first_image.mode != "RGB":
            first_image = first_image.convert("RGB")
        
        if isinstance(qa_pairs, str):
            qa_pairs = json.loads(qa_pairs)
        
        # Create one pair for each QA dict in qa_pairs list
        if not qa_pairs or not isinstance(qa_pairs, list):
            print(f"      ⚠️  Warning: no valid qa_pairs for row {idx}, skipping")
            continue
        
        for qa_idx, qa_dict in enumerate(qa_pairs):
            task_id = f"pbench_{idx:04d}_{qa_idx:02d}"
            
            question = qa_dict.get('question', '')
            answer = qa_dict.get('answer', '')
            category = qa_dict.get('category', '')
            subcategory = qa_dict.get('subcategory', '')
            
            goal = f"{question}, should be {answer}"
            
            pair = {
                'id': task_id,
                'domain': 'pbench',
                'prompt': prompt,
                'first_image': first_image,  # Store PIL Image object instead of file path
                'goal': goal,
                'category': category,
                'subcategory': subcategory,
                'row_index': idx,
                'qa_index': qa_idx
            }
            
            pairs.append(pair)
    
    print(f"   ✅ Downloaded {len(pairs)} PBench tasks")
    
    return {
        'name': 'pbench',
        'pairs': pairs,
        'source': 'huggingface',
        'hf_dataset': 'nvidia/PBench'
    }


if __name__ == "__main__":
    dataset = create_dataset()
    print(f"Dataset created with {len(dataset['pairs'])} tasks.")
