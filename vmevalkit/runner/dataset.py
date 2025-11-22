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

import sys
import json
import random
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
import importlib
from PIL import Image

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from vmevalkit.runner.TASK_CATALOG import DOMAIN_REGISTRY

# Domain Registry: Scalable way to add new domains
# ============================================================
# TO ADD A NEW TASK: Simply add an entry here with:
#   - name: Display name
#   - description: Brief task description
#   - module: Path to the task module
#   - create_function: Name of dataset creation function (usually 'create_dataset')
#   - process_dataset: Lambda to extract pairs from dataset (usually: lambda dataset, num_samples: dataset['pairs'])
# ============================================================

def download_hf_domain_to_folders(domain_name: str, output_base: Path) -> List[Dict[str, Any]]:
    """
    Download tasks for a HuggingFace-based domain into per-question folder structure.
    
    Args:
        domain_name: Name of the HuggingFace domain (arc_agi_2, eyeballing_puzzles, visual_puzzles, vpct)
        output_base: Base output directory for questions
        
    Returns:
        List of task pair metadata dictionaries
    """
    
    if domain_name not in DOMAIN_REGISTRY:
        raise ValueError(f"Unknown domain: {domain_name}. Available domains: {list(DOMAIN_REGISTRY.keys())}")
    
    domain_config = DOMAIN_REGISTRY[domain_name]
    
    if not domain_config.get('hf', False):
        raise ValueError(f"Domain {domain_name} is not a HuggingFace domain. Use generate_domain_to_folders() instead.")
    
    print("=" * 70)
    print(f"📥 Downloading {domain_name} tasks from HuggingFace...")
    print("=" * 70)
    print(f"   Dataset: {domain_config.get('hf_dataset')}")
    print(f"   Subset: {domain_config.get('hf_subset')}")
    print(f"   Split: {domain_config.get('hf_split')}")
    print(f"📁 Output directory: {output_base}")
    
    hf_domain = domain_config.get('hf_domain', domain_name)

    module = importlib.import_module(domain_config['module'])
    create_func = getattr(module, domain_config['create_function'])
    dataset = create_func()
    tasks = dataset['pairs']
    
    domain_dir = output_base / f"{hf_domain}_task"
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_tasks = []
    for task in tasks:
        task_id = task['id']
        task_dir = domain_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Handle first_image - can be Image object, numpy array, or file path
        first_image = task['first_image']
        if isinstance(first_image, str):
            # File path (for special formats like vpct)
            first_image_path = Path(first_image)
            if not first_image_path.exists():
                print(f"      ⚠️  Skipping {task_id}: Image not found at {first_image_path}")
                continue
            first_image = Image.open(first_image_path)
        elif not isinstance(first_image, Image.Image):
            # numpy array or other format
            first_image = Image.fromarray(first_image) if hasattr(first_image, 'shape') else Image.open(first_image)
        
        if first_image.mode != "RGB":
            first_image = first_image.convert("RGB")
        
        dest_first = task_dir / "first_frame.png"
        first_image.save(dest_first, format="PNG")
        
        # Handle solution_image (may not exist for special formats)
        solution_image = task.get('solution_image')
        final_image_path = None
        if solution_image is not None:
            if isinstance(solution_image, str):
                solution_image = Image.open(solution_image)
            elif not isinstance(solution_image, Image.Image):
                solution_image = Image.fromarray(solution_image) if hasattr(solution_image, 'shape') else Image.open(solution_image)
            if solution_image.mode != "RGB":
                solution_image = solution_image.convert("RGB")
            
            dest_final = task_dir / "final_frame.png"
            solution_image.save(dest_final, format="PNG")
            final_image_path = str(Path(f"{task['domain']}_task") / task_id / "final_frame.png")
        
        # Write prompt
        prompt_file = task_dir / "prompt.txt"
        prompt_file.write_text(task['prompt'])
        
        # Write goal if available (for special formats like vpct)
        if task.get('goal'):
            goal_file = task_dir / "goal.txt"
            goal_file.write_text(task['goal'])
        
        task_metadata = {
            "id": task_id,
            "domain": task['domain'],
            "prompt": task['prompt'],
            "first_image_path": str(Path(f"{task['domain']}_task") / task_id / "first_frame.png"),
            "final_image_path": final_image_path,
            "created_at": datetime.now().isoformat() + 'Z',
            "source": domain_config.get('hf_dataset'),
            "subset": domain_config.get('hf_subset')
        }
        
        # Add category if present (e.g., for MME-CoF)
        if 'category' in task:
            task_metadata['category'] = task['category']
        
        # Add goal and other metadata for special formats
        if task.get('goal'):
            task_metadata['goal'] = task['goal']
        if task.get('text_answer'):
            task_metadata['text_answer'] = task['text_answer']
        if task.get('sim_id'):
            task_metadata['sim_id'] = task['sim_id']
        
        metadata_file = task_dir / "question_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(task_metadata, f, indent=2, default=str)
        
        downloaded_tasks.append(task_metadata)
    
    print(f"✅ Downloaded {len(downloaded_tasks)} {domain_name} tasks to {domain_dir}")
    print()
    
    return downloaded_tasks


def generate_domain_to_folders(domain_name: str, num_samples: int, 
                              output_base: Path, random_seed: int | None) -> List[Dict[str, Any]]:
    """
    Generate tasks for a specific domain directly into per-question folder structure.
    
    Args:
        domain_name: Name of the domain (chess, maze, raven, rotation)
        num_samples: Number of task pairs to generate
        output_base: Base output directory for questions
        random_seed: Random seed for reproducible generation (None for non-deterministic)
        
    Returns:
        List of task pair metadata dictionaries
    """
    
    if domain_name not in DOMAIN_REGISTRY:
        raise ValueError(f"Unknown domain: {domain_name}. Available domains: {list(DOMAIN_REGISTRY.keys())}")
    
    domain_config = DOMAIN_REGISTRY[domain_name]
    
    domain_dir = output_base / f"{domain_name}_task"
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    if random_seed is not None:
        random.seed(random_seed + hash(domain_name))
    
    generated_pairs = []
    
    print(f"Generating {num_samples} {domain_config['name']} Tasks...")
    
    import importlib
    module = importlib.import_module(domain_config['module'])
    create_func = getattr(module, domain_config['create_function'])
    
    dataset = create_func(num_samples=num_samples)
    
    if domain_config['process_dataset']:
        pairs = domain_config['process_dataset'](dataset, num_samples)
    else:
        pairs = dataset['pairs']
    
    base_dir = Path(__file__).parent.parent.parent
    
    for idx, pair in enumerate(pairs):
        pair_id = pair.get("id") or f"{domain_name}_{idx:04d}"
        pair['id'] = pair_id
        pair['domain'] = domain_name
        
        q_dir = domain_dir / pair_id
        q_dir.mkdir(parents=True, exist_ok=True)
        
        first_rel = pair.get("first_image_path")
        final_rel = pair.get("final_image_path")
        
        if first_rel:
            src_first = base_dir / first_rel
            dst_first = q_dir / "first_frame.png"
            if src_first.exists():
                shutil.copyfile(src_first, dst_first)
                pair['first_image_path'] = str(Path(domain_name + "_task") / pair_id / "first_frame.png")
                
        if final_rel:
            src_final = base_dir / final_rel
            dst_final = q_dir / "final_frame.png"
            if src_final.exists():
                shutil.copyfile(src_final, dst_final)
                pair['final_image_path'] = str(Path(domain_name + "_task") / pair_id / "final_frame.png")
        
        prompt_text = pair.get("prompt", "")
        (q_dir / "prompt.txt").write_text(prompt_text)
        
        metadata_path = q_dir / "question_metadata.json"
        pair['created_at'] = datetime.now().isoformat() + 'Z'
        with open(metadata_path, 'w') as f:
            json.dump(pair, f, indent=2, default=str)
        
        generated_pairs.append(pair)
    
    print(f"   ✅ Generated {len(generated_pairs)} {domain_name} task pairs in {domain_dir}\n")
    
    return generated_pairs

def create_vmeval_dataset_direct(pairs_per_domain: int = 50, random_seed: int | None = 42, 
                                 selected_tasks: List[str] = None) -> Tuple[Dict[str, Any], str]:
    """
    Create VMEvalKit Dataset directly into per-question folder structure.
    
    Args:
        pairs_per_domain: Number of task pairs to generate per domain (default: 50)
        random_seed: Random seed for reproducible generation (default: 42, None for non-deterministic)
        selected_tasks: List of task names to generate. If None, generate all tasks.
        
    Returns:
        Tuple of (dataset dictionary, path to questions directory)
    """
    
    assert selected_tasks is not None, "selected_tasks must be provided"
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
    
    base_dir = Path(__file__).parent.parent.parent
    output_base = base_dir / "data" / "questions"
    output_base.mkdir(parents=True, exist_ok=True)
    
    allocation = {
        domain: pairs_per_domain 
        for domain in domains_to_generate
    }
    
    print(f"📈 Task Distribution:")
    print(f"   📌 Generating {pairs_per_domain} task pairs per reasoning domain")
    for domain, count in allocation.items():
        print(f"   {domain.title():10}: {count:3d} task pairs")
    print()
    
    all_pairs = []
    
    for domain_name, num_samples in allocation.items():
        pairs = generate_domain_to_folders(domain_name, num_samples, output_base, random_seed)
        all_pairs.extend(pairs)
    
    if random_seed is not None:
        random.seed(random_seed)
    random.shuffle(all_pairs)
    
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
    
    assert base_dir is not None, "base_dir must be provided"
    base_dir = Path(base_dir)
    
    all_pairs = []
    domains = list(DOMAIN_REGISTRY.keys())
    
    for domain in domains:
        domain_dir = base_dir / f"{domain}_task"
        if not domain_dir.exists():
            continue
            
        for q_dir in sorted(domain_dir.iterdir()):
            if not q_dir.is_dir():
                continue
                
            metadata_path = q_dir / "question_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    pair = json.load(f)
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