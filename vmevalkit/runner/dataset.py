#!/usr/bin/env python3
"""
VMEvalKit Dataset Creation Script

Directly generates the video reasoning scoring dataset into per-question folder structure
with <x> task pairs per domain, evenly distributed across all five reasoning domains:

- Chess: Strategic thinking and tactical pattern recognition
- Maze: Spatial reasoning and navigation planning  
- RAVEN: Abstract reasoning and pattern completion
- Rotation: 3D mental rotation and spatial visualization
- Sudoku: Logical reasoning and constraint satisfaction

Total: <x> task pairs (<x> per domain)

Author: VMEvalKit Team
"""

import os
import sys
import json
import random
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Callable

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Domain Registry: Scalable way to add new domains
# ============================================================
# TO ADD A NEW TASK: Simply add an entry here with:
#   - name: Display name
#   - description: Brief task description
#   - module: Path to the task module
#   - create_function: Name of dataset creation function (usually 'create_dataset')
#   - process_dataset: Lambda to extract pairs from dataset (usually: lambda dataset, num_samples: dataset['pairs'])
# ============================================================
DOMAIN_REGISTRY = {
    'chess': {
        'name': 'Chess',
        'description': 'Strategic thinking and tactical pattern recognition',
        'module': 'vmevalkit.tasks.chess_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'maze': {
        'name': 'Maze',
        'description': 'Spatial reasoning and navigation planning',
        'module': 'vmevalkit.tasks.maze_task',
        'create_function': 'create_dataset',  # Standard function like other domains
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'raven': {
        'name': 'RAVEN',
        'description': 'Abstract reasoning and pattern completion',
        'module': 'vmevalkit.tasks.raven_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'rotation': {
        'name': 'Rotation',
        'description': '3D mental rotation and spatial visualization',
        'module': 'vmevalkit.tasks.rotation_task.rotation_reasoning',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'sudoku': {
        'name': 'Sudoku',
        'description': 'Logical reasoning and constraint satisfaction',
        'module': 'vmevalkit.tasks.sudoku_task.sudoku_reasoning',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    }
}

def generate_domain_to_folders(domain_name: str, num_samples: int, 
                              output_base: Path, random_seed: int) -> List[Dict[str, Any]]:
    """
    Generate tasks for a specific domain directly into per-question folder structure.
    
    Args:
        domain_name: Name of the domain (chess, maze, raven, rotation)
        num_samples: Number of task pairs to generate
        output_base: Base output directory for questions
        random_seed: Random seed for reproducible generation
        
    Returns:
        List of task pair metadata dictionaries
    """
    
    # Check if domain exists in registry
    if domain_name not in DOMAIN_REGISTRY:
        raise ValueError(f"Unknown domain: {domain_name}. Available domains: {list(DOMAIN_REGISTRY.keys())}")
    
    # Get domain configuration
    domain_config = DOMAIN_REGISTRY[domain_name]
    
    # Create domain-specific task folder
    domain_dir = output_base / f"{domain_name}_task"
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed for this domain generation
    random.seed(random_seed + hash(domain_name))
    
    generated_pairs = []
    
    # Print generation message
    print(f"Generating {num_samples} {domain_config['name']} Tasks...")
    
    # Dynamic import and function call
    import importlib
    module = importlib.import_module(domain_config['module'])
    create_func = getattr(module, domain_config['create_function'])
    
    # Call the creation function
    dataset = create_func(num_samples=num_samples)
    
    # Process the dataset to extract pairs
    if domain_config['process_dataset']:
        pairs = domain_config['process_dataset'](dataset, num_samples)
    else:
        pairs = dataset['pairs']  # Default assumption
    
    # Now write each pair directly to its folder
    base_dir = Path(__file__).parent.parent.parent
    
    for idx, pair in enumerate(pairs):
        # Create unique ID
        pair_id = pair.get("id") or f"{domain_name}_{idx:04d}"
        pair['id'] = pair_id
        pair['domain'] = domain_name
        
        # Create question directory
        q_dir = domain_dir / pair_id
        q_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy images with standardized names
        first_rel = pair.get("first_image_path")
        final_rel = pair.get("final_image_path")
        
        if first_rel:
            src_first = base_dir / first_rel
            dst_first = q_dir / "first_frame.png"
            if src_first.exists():
                shutil.copyfile(src_first, dst_first)
                # Update path to relative from questions folder
                pair['first_image_path'] = str(Path(domain_name + "_task") / pair_id / "first_frame.png")
                
        if final_rel:
            src_final = base_dir / final_rel
            dst_final = q_dir / "final_frame.png"
            if src_final.exists():
                shutil.copyfile(src_final, dst_final)
                # Update path to relative from questions folder
                pair['final_image_path'] = str(Path(domain_name + "_task") / pair_id / "final_frame.png")
        
        # Write prompt
        prompt_text = pair.get("prompt", "")
        (q_dir / "prompt.txt").write_text(prompt_text)
        
        # Write metadata with creation timestamp
        metadata_path = q_dir / "question_metadata.json"
        pair['created_at'] = datetime.now().isoformat() + 'Z'
        with open(metadata_path, 'w') as f:
            json.dump(pair, f, indent=2, default=str)
        
        generated_pairs.append(pair)
    
    print(f"   ✅ Generated {len(generated_pairs)} {domain_name} task pairs in {domain_dir}\n")
    
    return generated_pairs

def create_vmeval_dataset_direct(pairs_per_domain: int = 50, random_seed: int = 42, 
                                 selected_tasks: List[str] = None) -> Tuple[Dict[str, Any], str]:
    """
    Create VMEvalKit Dataset directly into per-question folder structure.
    
    Args:
        pairs_per_domain: Number of task pairs to generate per domain (default: 50)
        random_seed: Random seed for reproducible generation (default: 42)
        selected_tasks: List of task names to generate. If None, generate all tasks.
        
    Returns:
        Tuple of (dataset dictionary, path to questions directory)
    """
    
    # Determine which domains to generate
    if selected_tasks is None:
        domains_to_generate = list(DOMAIN_REGISTRY.keys())
    else:
        # Validate task names
        invalid_tasks = [task for task in selected_tasks if task not in DOMAIN_REGISTRY]
        if invalid_tasks:
            raise ValueError(f"Unknown tasks: {invalid_tasks}. Available tasks: {list(DOMAIN_REGISTRY.keys())}")
        domains_to_generate = selected_tasks
    
    num_domains = len(domains_to_generate)
    total_pairs = pairs_per_domain * num_domains
    
    print("=" * 70)
    print("🚀 VMEvalKit Dataset Creation - Direct Folder Generation")
    print(f"🎯 Total target: {total_pairs} task pairs across {num_domains} domain(s)")
    print("=" * 70)
    
    # Setup output directory
    base_dir = Path(__file__).parent.parent.parent
    output_base = base_dir / "data" / "questions"
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Allocation for selected domains only
    allocation = {
        domain: pairs_per_domain 
        for domain in domains_to_generate
    }
    
    print(f"📈 Task Distribution:")
    print(f"   📌 Generating {pairs_per_domain} task pairs per reasoning domain")
    for domain, count in allocation.items():
        print(f"   {domain.title():10}: {count:3d} task pairs")
    print()
    
    # Generate each domain directly to folders
    all_pairs = []
    
    for domain_name, num_samples in allocation.items():
        pairs = generate_domain_to_folders(domain_name, num_samples, output_base, random_seed)
        all_pairs.extend(pairs)
    
    # Shuffle all pairs for diversity
    random.seed(random_seed)
    random.shuffle(all_pairs)
    
    # Create master dataset from the generated folders
    creation_timestamp = datetime.now().isoformat() + 'Z'
    dataset = {
        "name": "vmeval_dataset",
        "description": f"VMEvalKit video reasoning evaluation dataset ({len(all_pairs)} task pairs)",
        "created_at": creation_timestamp,
        "total_pairs": len(all_pairs),
        "generation_info": {
            "timestamp": creation_timestamp,
            "random_seed": random_seed,
            "pairs_per_domain": pairs_per_domain,
            "target_pairs": total_pairs,
            "actual_pairs": len(all_pairs),
            "allocation": allocation,
            "domains": {
                domain: {
                    "count": len([p for p in all_pairs if p.get('domain') == domain]),
                    "description": config['description']
                }
                for domain, config in DOMAIN_REGISTRY.items()
                if domain in domains_to_generate
            }
        },
        "pairs": all_pairs
    }
    
    # Save master JSON
    json_path = output_base / "vmeval_dataset.json"
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    
    return dataset, str(output_base)

def read_dataset_from_folders(base_dir: Path = None) -> Dict[str, Any]:
    """Read dataset from existing per-question folder structure.
    
    Args:
        base_dir: Base directory containing question folders (default: data/questions)
        
    Returns:
        Dataset dictionary constructed from folder contents
    """
    
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / "data" / "questions"
    else:
        base_dir = Path(base_dir)
    
    all_pairs = []
    domains = list(DOMAIN_REGISTRY.keys())
    
    for domain in domains:
        domain_dir = base_dir / f"{domain}_task"
        if not domain_dir.exists():
            continue
            
        # Read all question folders in this domain
        for q_dir in sorted(domain_dir.iterdir()):
            if not q_dir.is_dir():
                continue
                
            # Read metadata if exists
            metadata_path = q_dir / "question_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    pair = json.load(f)
                    # Ensure domain is tagged
                    pair['domain'] = domain
                    all_pairs.append(pair)
    
    # Create dataset structure
    # For reading existing datasets, try to preserve the creation time if metadata exists
    creation_timestamp = None
    for domain in domains:
        if creation_timestamp:
            break
        domain_dir = base_dir / f"{domain}_task"
        if domain_dir.exists():
            # Try to read creation time from first metadata file
            for q_dir in domain_dir.iterdir():
                metadata_path = q_dir / "question_metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        meta = json.load(f)
                        if 'created_at' in meta:
                            creation_timestamp = meta['created_at']
                            break
    
    # If no timestamp found, use file modification time or current time
    if not creation_timestamp:
        creation_timestamp = datetime.now().isoformat() + 'Z'
    
    dataset = {
        "name": "vmeval_dataset",
        "description": f"VMEvalKit video reasoning evaluation dataset ({len(all_pairs)} task pairs)",
        "created_at": creation_timestamp,
        "total_pairs": len(all_pairs),
        "generation_info": {
            "domains": {
                domain: {
                    "count": len([p for p in all_pairs if p.get('domain') == domain]),
                    "description": DOMAIN_REGISTRY.get(domain, {}).get('description', 'Unknown domain')
                }
                for domain in domains
                if domain in DOMAIN_REGISTRY
            }
        },
        "pairs": all_pairs
    }
    
    return dataset

def print_dataset_summary(dataset: Dict[str, Any]):
    """Print comprehensive dataset summary."""
    
    print("=" * 70)
    print("📊 VMEVAL DATASET - SUMMARY")
    print("=" * 70)
    
    gen_info = dataset.get('generation_info', {})
    domains = gen_info.get('domains', {})
    
    print(f"🎯 Dataset Statistics:")
    print(f"   Total Task Pairs: {dataset['total_pairs']}")
    if 'created_at' in dataset:
        print(f"   Created: {dataset['created_at']}")
    elif 'timestamp' in gen_info:
        print(f"   Created: {gen_info['timestamp']}")
    
    # Only show target/success rate if available (from generation)
    if 'target_pairs' in gen_info:
        print(f"   Target: {gen_info['target_pairs']} ({gen_info.get('pairs_per_domain', 'N/A')} per domain)")
        print(f"   Success Rate: {dataset['total_pairs']/gen_info['target_pairs']*100:.1f}%")
    print()
    
    print(f"🧠 Reasoning Domains:")
    for domain, info in domains.items():
        percentage = info['count'] / dataset['total_pairs'] * 100 if dataset['total_pairs'] > 0 else 0
        print(f"   {domain.title():10}: {info['count']:2d} pairs ({percentage:4.1f}%) - {info['description']}")
    print()
    
    # Difficulty distribution
    difficulties = {}
    categories = {}
    for pair in dataset['pairs']:
        diff = pair.get('difficulty', 'unknown')
        cat = pair.get('task_category', 'unknown')
        difficulties[diff] = difficulties.get(diff, 0) + 1
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"📈 Difficulty Distribution:")
    for diff, count in sorted(difficulties.items()):
        percentage = count / dataset['total_pairs'] * 100 if dataset['total_pairs'] > 0 else 0
        print(f"   {diff.title():10}: {count:3d} pairs ({percentage:4.1f}%)")
    print()
    
    print(f"🏷️  Task Categories ({len(categories)} unique):")
    # Show top 10 categories
    sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]
    for cat, count in sorted_categories:
        percentage = count / dataset['total_pairs'] * 100 if dataset['total_pairs'] > 0 else 0
        print(f"   {cat:20}: {count:3d} pairs ({percentage:4.1f}%)")
    if len(categories) > 10:
        print(f"   ... and {len(categories) - 10} more categories")
    print()

# CLI functionality has been moved to examples/create_dataset.py
# This module now provides pure functions for dataset creation