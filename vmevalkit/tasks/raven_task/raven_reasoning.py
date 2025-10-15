#!/usr/bin/env python3
"""
RAVEN Reasoning Task for VMEvalKit

This module generates Progressive Matrix (RPM) reasoning tasks for video model evaluation.


The task evaluates video models' ability to:
1. Recognize visual patterns across multiple panels
2. Apply abstract logical rules (progression, arithmetic, etc.)
3. Complete missing patterns through reasoning
4. Generate coherent reasoning sequences in video form

Author: VMEvalKit Team
"""

import os
import sys
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

# Import local RAVEN core without try-catch to see actual errors
from .local_raven import (
    IMAGE_SIZE,
    build_center_single,
    build_distribute_four,
    build_distribute_nine,
    generate_matrix,
    render_panel,
    sample_rules,
)
from .local_raven.rules import Rule_Wrapper


class RavenGenerator:
    """Self-contained RAVEN Progressive Matrix task generator."""
    
    # Configuration mapping
    CONFIGURATIONS = {
        "Center": "center_single",          # Most reliable
        "2x2Grid": "distribute_four",       # 2x2 grid
        "3x3Grid": "distribute_nine"        # 3x3 grid
    }
    
    def __init__(self):
        """Initialize RAVEN generator with configurations."""
        self.generated_tasks = []
        self.setup_configurations()
        
    def setup_configurations(self):
        """Setup RAVEN configuration trees."""
        # Build configuration trees
        self.config_trees = {
            "center_single": build_center_single(),
            "distribute_four": build_distribute_four(),
            "distribute_nine": build_distribute_nine()
        }
        
    def generate_single_task(self, config_name: str, difficulty: str = None) -> Dict[str, Any]:
        """Generate a single RAVEN task."""
        
        # Get configuration tree
        if config_name not in self.config_trees:
            raise ValueError(f"Unknown configuration: {config_name}")
            
        root = self.config_trees[config_name]
        
        # Sample rules for this configuration
        max_attempts = 10
        for attempt in range(max_attempts):
            # Get rules (no try-catch to see actual errors)
            rule_groups = sample_rules(configuration=config_name)
            new_root = root.prune(rule_groups)
            
            if new_root is not None:
                panels = self.generate_panels(new_root, rule_groups)
                return {
                    "config_name": config_name,
                    "config_display": [k for k, v in self.CONFIGURATIONS.items() if v == config_name][0],
                    "matrix": panels,
                    "attempts": attempt + 1,
                }
            print(f"Attempt {attempt + 1}: Pruning failed for {config_name}, trying again...")
                
        raise RuntimeError(f"Failed to generate valid task for {config_name} after {max_attempts} attempts")
    
    def generate_panels(self, root, rule_groups) -> List[np.ndarray]:
        """Generate the 9 panels of a Progressive Matrix following original RAVEN logic."""
        import copy

        start_node = root.sample()

        # Row 1: Apply rules to generate row_1_1, row_1_2, row_1_3
        row_1_1 = copy.deepcopy(start_node)
        for l in range(len(rule_groups)):
            rule_group = rule_groups[l]
            rule_num_pos = rule_group[0]
            row_1_2 = rule_num_pos.apply_rule(row_1_1)
            row_1_3 = rule_num_pos.apply_rule(row_1_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_1_2 = rule.apply_rule(row_1_1, row_1_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_1_3 = rule.apply_rule(row_1_2, row_1_3)
            if l == 0:
                to_merge = [row_1_1, row_1_2, row_1_3]
            else:
                self.merge_component(to_merge[1], row_1_2, l)
                self.merge_component(to_merge[2], row_1_3, l)
        row_1_1, row_1_2, row_1_3 = to_merge

        # Row 2: Create new base and apply same rules
        row_2_1 = copy.deepcopy(start_node)
        row_2_1.resample(True)
        for l in range(len(rule_groups)):
            rule_group = rule_groups[l]
            rule_num_pos = rule_group[0]
            row_2_2 = rule_num_pos.apply_rule(row_2_1)
            row_2_3 = rule_num_pos.apply_rule(row_2_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_2_2 = rule.apply_rule(row_2_1, row_2_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_2_3 = rule.apply_rule(row_2_2, row_2_3)
            if l == 0:
                to_merge = [row_2_1, row_2_2, row_2_3]
            else:
                self.merge_component(to_merge[1], row_2_2, l)
                self.merge_component(to_merge[2], row_2_3, l)
        row_2_1, row_2_2, row_2_3 = to_merge

        # Row 3: Create new base and apply same rules
        row_3_1 = copy.deepcopy(start_node)
        row_3_1.resample(True)
        for l in range(len(rule_groups)):
            rule_group = rule_groups[l]
            rule_num_pos = rule_group[0]
            row_3_2 = rule_num_pos.apply_rule(row_3_1)
            row_3_3 = rule_num_pos.apply_rule(row_3_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_3_2 = rule.apply_rule(row_3_1, row_3_2)
            for i in range(1, len(rule_group)):
                rule = rule_group[i]
                row_3_3 = rule.apply_rule(row_3_2, row_3_3)
            if l == 0:
                to_merge = [row_3_1, row_3_2, row_3_3]
            else:
                self.merge_component(to_merge[1], row_3_2, l)
                self.merge_component(to_merge[2], row_3_3, l)
        row_3_1, row_3_2, row_3_3 = to_merge

        # Return rendered panels following original RAVEN order
        imgs = [render_panel(row_1_1),
                render_panel(row_1_2), 
                render_panel(row_1_3),
                render_panel(row_2_1),
                render_panel(row_2_2),
                render_panel(row_2_3),
                render_panel(row_3_1),
                render_panel(row_3_2),
                render_panel(row_3_3)]  # Include 9th panel for complete matrix
        
        return imgs
    
    def merge_component(self, dst_aot, src_aot, component_idx):
        """Merge component from src to dst (from RAVEN main.py)."""
        src_component = src_aot.children[0].children[component_idx]
        dst_aot.children[0].children[component_idx] = src_component
    
    def generate_tasks(self, num_tasks: int = 50) -> List[Dict[str, Any]]:
        """Generate tasks across different configurations."""
        print(f"🎯 Generating {num_tasks} RAVEN tasks across {len(self.CONFIGURATIONS)} configurations...")
        tasks: List[Dict[str, Any]] = []
        configs = list(self.CONFIGURATIONS.values())
        for i in range(num_tasks):
            cfg = configs[i % len(configs)]
            task = self.generate_single_task(cfg)
            tasks.append(task)
            print(f"✅ {i+1}/{num_tasks}: {task['config_display']}")
        self.generated_tasks = tasks
        return tasks


def generate_task_images(task_data: Dict[str, Any], output_dir: str, task_id: str) -> Tuple[str, str]:
    """
    Generate first and final frame images for a RAVEN task.
    
    Args:
        task_data: Generated task data containing matrix
        output_dir: Base output directory
        task_id: Task identifier for naming
    
    Returns:
        (first_image_path, final_image_path)
    """
    matrix = task_data["matrix"]
    
    # Create temporary files that will be moved to per-question folders
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    # Image paths
    first_image_path = os.path.join(temp_dir, f"{task_id}_first.png")
    final_image_path = os.path.join(temp_dir, f"{task_id}_final.png")

    # Always compose into a 3x3 RAVEN matrix for consistency
    generate_rpm_image(matrix, first_image_path, incomplete=True)
    generate_rpm_image(matrix, final_image_path, incomplete=False)
    
    # Return temp paths that will be moved by create_dataset.py
    return first_image_path, final_image_path


def generate_rpm_image(matrix_panels: List[np.ndarray], output_path: str, incomplete: bool = False):
    """Render a 3x3 RAVEN matrix image using the local renderer."""
    import numpy as np
    from PIL import Image

    # Prepare exactly 9 panels; fill missing with white
    panels: List[np.ndarray] = []
    take = min(len(matrix_panels), 9)
    if incomplete and take >= 8:
        panels.extend(matrix_panels[:8])
        panels.append(np.ones((IMAGE_SIZE, IMAGE_SIZE), np.uint8) * 255)
    else:
        panels.extend(matrix_panels[:take])
        while len(panels) < 9:
            panels.append(np.ones((IMAGE_SIZE, IMAGE_SIZE), np.uint8) * 255)

    grid = generate_matrix(panels)
    Image.fromarray(grid).save(output_path)


def generate_prompt(task_data: Dict[str, Any]) -> str:
    """Generate concise prompt with config-aware base."""
    config_display = task_data["config_display"]
    base = {
        "Center": "Complete this center-focused pattern matrix",
        "2x2Grid": "Complete this 2x2 grid pattern matrix",
        "3x3Grid": "Complete this 3x3 grid pattern matrix",
    }.get(config_display, "Complete this pattern matrix")
    panel = "4th panel" if config_display == "2x2Grid" else "9th panel" if config_display == "3x3Grid" else "missing panel"
    return f"{base}. Show what goes in the missing {panel}."


def create_task_pair(task_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Create a RAVEN task pair in VMEvalKit format."""
    
    # Generate images
    base_dir = Path(__file__).parent.parent.parent.parent
    first_image_path, final_image_path = generate_task_images(task_data, str(base_dir), task_id)
    
    # Generate prompt  
    prompt = generate_prompt(task_data)
    
    # Create task pair following VMEvalKit structure
    task_pair = {
        "id": task_id,
        "prompt": prompt,
        "first_image_path": first_image_path,
        "final_image_path": final_image_path,
        "task_category": task_data["config_display"],
        "raven_data": {
            "generation_method": "RAVEN Progressive Matrix Generator",
            "configuration": task_data["config_name"],
            "matrix_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            "pattern_type": "Progressive Matrix"
        },
        "configuration_type": task_data["config_display"],
        "created_at": datetime.now().isoformat()
    }
    
    return task_pair


def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """Create Progressive Matrix tasks across multiple configurations for better generation success."""
    
    print(f"🎯 Creating RAVEN Progressive Matrix dataset with {num_samples} samples across 3 configurations...")
    
    # Generate tasks
    generator = RavenGenerator()
    tasks = generator.generate_tasks(num_samples)
    
    if len(tasks) == 0:
        raise RuntimeError("Failed to generate any valid RAVEN tasks")
    
    # Create task pairs
    pairs = []
    for i, task_data in enumerate(tasks):
        task_id = f"raven_{i:04d}"
        
        pair = create_task_pair(task_data, task_id)
        pairs.append(pair)
        print(f"✅ Created task {task_id}: {pair['task_category']}")
    
    if len(pairs) == 0:
        raise RuntimeError("Failed to create any valid task pairs")
    
    # Create dataset
    dataset = {
        "name": "raven_tasks",
        "description": f"RAVEN Progressive Matrix tasks across 3 configurations (2x2, 3x3, center) for video reasoning evaluation ({len(pairs)} pairs)",
        "pairs": pairs
    }
    
    # Don't save to intermediate folder anymore - will be handled by create_dataset.py
    print(f"📊 Dataset stats:")
    
    # Print statistics
    categories = {}
    for pair in pairs:
        cat = pair['task_category']
        categories[cat] = categories.get(cat, 0) + 1

    print(f"   Categories: {categories}")
    
    return dataset


# Dataset creation should only be done via vmevalkit/runner/create_dataset.py
# This module only provides the create_dataset() function as an API