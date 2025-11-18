#!/usr/bin/env python3
"""
Perspective Taking Task for VMEvalKit

Reads pre-generated perspective taking tasks from data/perspective_taking/.
Tests video models' ability to understand spatial relationships and viewpoint transformations.

Author: VMEvalKit Team
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
from PIL import Image


def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Load perspective taking tasks from existing data directory.
    
    The perspective taking task evaluates whether video models can:
    - Understand spatial relationships between objects and agents
    - Transform viewpoints and perspectives
    - Reason about what can be seen from different angles
    - Generate scenes from a back-facing perspective
    
    Args:
        num_samples: Number of samples to include (None = all available)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    print(f"📥 Loading Perspective Taking tasks from data/perspective_taking/...")
    
    # Get the data directory path (relative to this file)
    base_dir = Path(__file__).parent.parent.parent.parent / "data" / "perspective_taking"
    
    if not base_dir.exists():
        raise FileNotFoundError(
            f"Perspective taking data directory not found at: {base_dir}\n"
            f"Please ensure data/perspective_taking/ exists with task folders."
        )
    
    pairs = []
    
    # Get all numbered task directories and sort them
    task_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                       key=lambda x: int(x.name))
    
    print(f"   Found {len(task_dirs)} perspective taking task folders")
    
    for task_dir in task_dirs:
        task_num = int(task_dir.name)
        task_id = f"perspective_taking_{task_num:04d}"
        
        # Check for required files
        first_frame = task_dir / "first_frame.png"
        final_frame = task_dir / "final_frame.png"
        prompt_file = task_dir / "prompt.txt"
        metadata_file = task_dir / "scene_metadata.json"
        
        # Skip if required files are missing
        if not all([first_frame.exists(), final_frame.exists(), prompt_file.exists()]):
            print(f"   ⚠️  Skipping {task_id}: missing required files")
            continue
        
        # Read prompt
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
        
        # Read metadata if available
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        # Read images to verify they're valid
        try:
            with Image.open(first_frame) as img:
                first_size = img.size
            with Image.open(final_frame) as img:
                final_size = img.size
        except Exception as e:
            print(f"   ⚠️  Skipping {task_id}: invalid image files - {e}")
            continue
        
        # Create task pair
        pair = {
            'id': task_id,
            'domain': 'perspective_taking',
            'task_category': 'spatial_reasoning',
            'prompt': prompt,
            'first_image_path': str(first_frame),
            'final_image_path': str(final_frame),
            'difficulty': 'medium',  # Can be adjusted based on metadata if available
            'created_at': datetime.now().isoformat(),
            'perspective_taking_data': {
                'task_number': task_num,
                'first_image_size': first_size,
                'final_image_size': final_size,
                'metadata': metadata
            }
        }
        
        pairs.append(pair)
        
        # Stop if we've reached the requested number of samples
        if num_samples is not None and len(pairs) >= num_samples:
            break
    
    print(f"   ✅ Loaded {len(pairs)} perspective taking tasks")
    
    if len(pairs) == 0:
        raise ValueError(
            "No valid perspective taking tasks found. "
            "Please check that data/perspective_taking/ contains valid task folders."
        )
    
    return {
        'name': 'perspective_taking_tasks',
        'pairs': pairs,
        'source': 'local_data',
        'data_path': str(base_dir)
    }


def validate_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the perspective taking dataset structure.
    
    Args:
        dataset: Dataset dictionary from create_dataset()
        
    Returns:
        Validation results with issues found
    """
    issues = []
    pairs = dataset.get('pairs', [])
    
    for pair in pairs:
        # Check required fields
        required_fields = ['id', 'domain', 'prompt', 'first_image_path', 'final_image_path']
        for field in required_fields:
            if field not in pair:
                issues.append(f"{pair.get('id', 'unknown')}: Missing field '{field}'")
        
        # Check image files exist
        if 'first_image_path' in pair:
            if not Path(pair['first_image_path']).exists():
                issues.append(f"{pair['id']}: First image not found: {pair['first_image_path']}")
        
        if 'final_image_path' in pair:
            if not Path(pair['final_image_path']).exists():
                issues.append(f"{pair['id']}: Final image not found: {pair['final_image_path']}")
    
    return {
        'valid': len(issues) == 0,
        'total_pairs': len(pairs),
        'issues': issues
    }


if __name__ == '__main__':
    """Test the perspective taking dataset creation."""
    print("=" * 60)
    print("Testing Perspective Taking Dataset Creation")
    print("=" * 60)
    
    # Create dataset
    dataset = create_dataset()
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Name: {dataset['name']}")
    print(f"   Source: {dataset['source']}")
    print(f"   Total pairs: {len(dataset['pairs'])}")
    
    # Show first task as example
    if dataset['pairs']:
        print(f"\n📝 Example Task (first):")
        first_pair = dataset['pairs'][0]
        print(f"   ID: {first_pair['id']}")
        print(f"   Domain: {first_pair['domain']}")
        print(f"   Prompt: {first_pair['prompt'][:100]}...")
        print(f"   First image: {first_pair['first_image_path']}")
        print(f"   Final image: {first_pair['final_image_path']}")
    
    # Validate dataset
    print(f"\n✅ Validating dataset...")
    validation = validate_dataset(dataset)
    if validation['valid']:
        print(f"   ✅ Dataset is valid!")
    else:
        print(f"   ❌ Found {len(validation['issues'])} issues:")
        for issue in validation['issues'][:5]:
            print(f"      - {issue}")
    
    print("\n" + "=" * 60)

