#!/usr/bin/env python3
"""
VMEvalKit Dataset Creation Script

Generates the first version of our comprehensive video reasoning evaluation dataset
with 100 task pairs randomly distributed across all four reasoning domains:

- Chess: Strategic thinking and tactical pattern recognition
- Maze: Spatial reasoning and navigation planning  
- RAVEN: Abstract reasoning and pattern completion
- Rotation: 3D mental rotation and spatial visualization

Author: VMEvalKit Team
"""

import os
import sys
import json
import random
from pathlib import Path
from typing import Dict, List, Any

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def create_vmeval_dataset_v1(pairs_per_domain: int = 100, random_seed: int = 42) -> Dict[str, Any]:
    """
    Create VMEvalKit Dataset Version 1 with equal task pairs per domain.
    
    Args:
        pairs_per_domain: Number of task pairs to generate per domain (default: 100)
        random_seed: Random seed for reproducible generation (default: 42)
        
    Returns:
        Combined dataset dictionary
    """
    
    total_pairs = pairs_per_domain * 4
    
    print("=" * 70)
    print("🚀 VMEvalKit Dataset Creation v1.0")
    print(f"📊 Generating {pairs_per_domain} task pairs per reasoning domain")
    print(f"🎯 Total target: {total_pairs} task pairs across 4 domains")
    print("=" * 70)
    
    # Set random seed for reproducible generation
    random.seed(random_seed)
    
    # Equal allocation across domains
    allocation = {
        'chess': pairs_per_domain,
        'maze': pairs_per_domain, 
        'raven': pairs_per_domain,
        'rotation': pairs_per_domain
    }
    
    print(f"📈 Task Distribution:")
    for domain, count in allocation.items():
        print(f"   {domain.title():10}: {count:3d} task pairs")
    print()
    
    # Generate each domain's dataset
    all_pairs = []
    
    def _ensure_domain_tag(pairs_list, domain_name):
        """Add a 'domain' field to each task pair dict for robust aggregation."""
        tagged = []
        for pair in pairs_list:
            item = dict(pair)
            item['domain'] = domain_name
            tagged.append(item)
        return tagged
    
    try:
        # 1. Chess Tasks - Strategic Reasoning
        print("♟️  Generating Chess Tasks...")
        from vmevalkit.tasks.chess_task import create_chess_dataset
        chess_dataset = create_chess_dataset(num_samples=allocation['chess'])
        chess_pairs = chess_dataset['pairs']
        all_pairs.extend(_ensure_domain_tag(chess_pairs, 'chess'))
        print(f"   ✅ Generated {len(chess_pairs)} chess task pairs\n")
        
    except Exception as e:
        print(f"   ❌ Chess generation failed: {e}\n")
        chess_pairs = []
    
    try:
        # 2. Maze Tasks - Spatial Navigation  
        print("🌀 Generating Maze Tasks...")
        from vmevalkit.tasks.maze_task import create_combined_dataset
        
        # Split maze allocation between KnowWhat and Irregular (roughly 40/60)
        maze_total = allocation['maze']
        knowwhat_count = max(1, maze_total * 2 // 5)  # ~40%
        irregular_count = maze_total - knowwhat_count   # ~60%
        
        maze_dataset = create_combined_dataset(
            knowwhat_samples=knowwhat_count,
            irregular_samples=irregular_count
        )
        maze_pairs = maze_dataset.pairs
        # Convert dataclass instances to dicts and tag domain
        maze_dicts = [dict(pair.__dict__) for pair in maze_pairs]
        all_pairs.extend(_ensure_domain_tag(maze_dicts, 'maze'))
        print(f"   ✅ Generated {len(maze_pairs)} maze task pairs\n")
        
    except Exception as e:
        print(f"   ❌ Maze generation failed: {e}\n")
        maze_pairs = []
    
    try:
        # 3. RAVEN Tasks - Abstract Reasoning
        print("🧩 Generating RAVEN Tasks...")
        from vmevalkit.tasks.raven_task.raven_reasoning import create_dataset as create_raven_dataset
        raven_dataset = create_raven_dataset(num_samples=allocation['raven'])
        raven_pairs = raven_dataset['pairs']
        all_pairs.extend(_ensure_domain_tag(raven_pairs, 'raven'))
        print(f"   ✅ Generated {len(raven_pairs)} RAVEN task pairs\n")
        
    except Exception as e:
        print(f"   ❌ RAVEN generation failed: {e}\n")
        raven_pairs = []
    
    try:
        # 4. Rotation Tasks - 3D Mental Rotation
        print("🔄 Generating Rotation Tasks...")
        from vmevalkit.tasks.rotation_task.rotation_reasoning import create_dataset as create_rotation_dataset
        rotation_dataset = create_rotation_dataset(num_samples=allocation['rotation'])
        rotation_pairs = rotation_dataset['pairs']
        all_pairs.extend(_ensure_domain_tag(rotation_pairs, 'rotation'))
        print(f"   ✅ Generated {len(rotation_pairs)} rotation task pairs\n")
        
    except Exception as e:
        print(f"   ❌ Rotation generation failed: {e}\n")
        rotation_pairs = []
    
    # Shuffle all pairs for random ordering
    random.shuffle(all_pairs)
    
    # Create master dataset
    dataset = {
        "name": "vmeval_dataset_v1",
        "description": f"VMEvalKit comprehensive video reasoning evaluation dataset v1.0 ({len(all_pairs)} task pairs)",
        "version": "1.0.0",
        "total_pairs": len(all_pairs),
        "generation_info": {
            "random_seed": random_seed,
            "pairs_per_domain": pairs_per_domain,
            "target_pairs": total_pairs,
            "actual_pairs": len(all_pairs),
            "allocation": allocation,
            "domains": {
                # Use explicit domain tagging for robust counts
                "chess": {"count": len([p for p in all_pairs if p.get('domain') == 'chess']), 
                         "description": "Strategic thinking and tactical pattern recognition"},
                "maze": {"count": len([p for p in all_pairs if p.get('domain') == 'maze']), 
                        "description": "Spatial reasoning and navigation planning"},
                "raven": {"count": len([p for p in all_pairs if p.get('domain') == 'raven']), 
                         "description": "Abstract reasoning and pattern completion"},
                "rotation": {"count": len([p for p in all_pairs if p.get('domain') == 'rotation']), 
                           "description": "3D mental rotation and spatial visualization"}
            }
        },
        "pairs": all_pairs
    }
    
    return dataset

def save_master_dataset(dataset: Dict[str, Any], output_path: str = None) -> str:
    """Save the master dataset to JSON file."""
    
    if output_path is None:
        questions_dir = Path(__file__).parent.parent.parent / "data" / "questions"
        questions_dir.mkdir(parents=True, exist_ok=True)
        output_path = questions_dir / "vmeval_dataset_v1.json"
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2, default=str)
    
    return str(output_path)

def print_dataset_summary(dataset: Dict[str, Any]):
    """Print comprehensive dataset summary."""
    
    print("=" * 70)
    print("📊 VMEVAL DATASET V1.0 - GENERATION COMPLETE")
    print("=" * 70)
    
    gen_info = dataset['generation_info']
    domains = gen_info['domains']
    
    print(f"🎯 Dataset Statistics:")
    print(f"   Total Task Pairs: {dataset['total_pairs']}")
    print(f"   Target: {gen_info['target_pairs']} ({gen_info['pairs_per_domain']} per domain)")
    print(f"   Success Rate: {dataset['total_pairs']/gen_info['target_pairs']*100:.1f}%")
    print()
    
    print(f"🧠 Reasoning Domains:")
    for domain, info in domains.items():
        percentage = info['count'] / dataset['total_pairs'] * 100
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
        percentage = count / dataset['total_pairs'] * 100
        print(f"   {diff.title():10}: {count:2d} pairs ({percentage:4.1f}%)")
    print()
    
    print(f"🏷️  Task Categories:")
    for cat, count in sorted(categories.items()):
        percentage = count / dataset['total_pairs'] * 100
        print(f"   {cat:15}: {count:2d} pairs ({percentage:4.1f}%)")
    print()

def main():
    """Generate VMEvalKit Dataset Version 1."""
    
    # Generate dataset with 150 task pairs per domain
    dataset = create_vmeval_dataset_v1(pairs_per_domain=150, random_seed=42)
    
    # Save to file
    output_path = save_master_dataset(dataset)
    
    # Print comprehensive summary
    print_dataset_summary(dataset)
    
    print(f"💾 Dataset saved: {output_path}")
    print(f"🔗 Images location: data/questions/generated_*/")
    print()
    print("🎉 VMEvalKit Dataset v1.0 ready for video reasoning evaluation!")
    print("🚀 Use `vmevalkit/runner/inference.py` to evaluate models on this dataset")
    print("=" * 70)

if __name__ == "__main__":
    main()
