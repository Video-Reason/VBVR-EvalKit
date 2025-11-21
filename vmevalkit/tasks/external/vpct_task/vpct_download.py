#!/usr/bin/env python3
"""
VPCT Task for VMEvalKit

Downloads VPCT (Visual Physics Comprehension Test) tasks from HuggingFace.
VPCT dataset contains 100 problems with physics simulation scenarios.

Author: VMEvalKit Team
"""

import json
import shutil
from pathlib import Path
from typing import Dict, Any
from huggingface_hub import snapshot_download
from PIL import Image


def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Download VPCT dataset from HuggingFace.
    
    VPCT provides visual physics comprehension test with 100 problems.
    Each problem has sim_k_initial.png (k from 1 to 100) and associated JSON files.
    
    Args:
        num_samples: Not used for HuggingFace downloads (downloads all available)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    print(f"📥 Downloading VPCT tasks from HuggingFace...")
    
    # Download dataset to temporary directory
    temp_dir = Path("/tmp/vpct-1-download")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   Downloading from: camelCase12/vpct-1")
    snapshot_download(
        repo_id="camelCase12/vpct-1",
        repo_type="dataset",
        local_dir=str(temp_dir),
    )
    
    pairs = []
    # Process files: sim_k_initial.png (k from 1 to 100)
    for k in range(1, 101):
        sim_id = str(k)
        initial_image_path = temp_dir / f"sim_{sim_id}_initial.png"
        results_json_path = temp_dir / f"sim_{sim_id}_results.json"
        
        if not initial_image_path.exists():
            print(f"      ⚠️  Warning: sim_{sim_id}_initial.png not found, skipping")
            continue
        
        # Read image and convert to PIL Image object (so we can delete temp dir later)
        first_image = Image.open(initial_image_path)
        if first_image.mode != "RGB":
            first_image = first_image.convert("RGB")
        
        # Read goal from results.json (finalBucket)
        goal_text = None
        text_answer = None
        if results_json_path.exists():
            with open(results_json_path, 'r') as f:
                results_data = json.load(f)
                text_answer = results_data.get('finalBucket')
                if text_answer is not None:
                    goal_text = f"Check if the ball fall into the bucket number {text_answer}."

        if not goal_text:
            raise ValueError(f"No goal text found for sim_{sim_id}")
        
        task_id = f"vpct_{k:04d}"
        
        pair = {
            'id': task_id,
            'domain': 'vpct',
            'prompt': 'Generate a video that shows the ball falling, the ball does not bounce when touch the bucket, video stop when the ball hits the bucket or the boundary of the screen.',
            'first_image': first_image,  # Store PIL Image object instead of file path
            'goal': goal_text,
            'sim_id': sim_id,
            'text_answer': text_answer
        }
        
        pairs.append(pair)
    
    print(f"   ✅ Downloaded {len(pairs)} VPCT tasks")
    
    # Clean up temporary directory
    shutil.rmtree(temp_dir)
    
    return {
        'name': 'vpct',
        'pairs': pairs,
        'source': 'huggingface',
        'hf_dataset': 'camelCase12/vpct-1'
    }

