#!/usr/bin/env python3
"""
RAVEN Reasoning Task for VMEvalKit

This module generates Progressive Matrix (RPM) reasoning tasks for video model evaluation.
Based on the CVPR 2019 RAVEN dataset for Relational and Analogical Visual rEasoNing.

The task evaluates video models' ability to:
1. Recognize visual patterns across multiple panels
2. Apply abstract logical rules (progression, arithmetic, etc.)
3. Complete missing patterns through reasoning
4. Generate coherent reasoning sequences in video form

Author: VMEvalKit Team
"""

import os
import sys
import json
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

# Add RAVEN submodule to path and fix scipy compatibility
RAVEN_PATH = Path(__file__).parent.parent.parent.parent / "submodules" / "RAVEN"
sys.path.append(str(RAVEN_PATH / "src" / "dataset"))

# Fix scipy compatibility issue (comb moved from misc to special in newer versions)
try:
    import scipy.misc
    import scipy.special
    if not hasattr(scipy.misc, 'comb'):
        scipy.misc.comb = scipy.special.comb
except ImportError:
    pass

# Import RAVEN components (avoiding Python 2.7 main.py)
try:
    from build_tree import (
        build_center_single, build_distribute_four, build_distribute_nine,
        build_in_center_single_out_center_single,
        build_in_distribute_four_out_center_single,
        build_left_center_single_right_center_single,
        build_up_center_single_down_center_single
    )
    from const import IMAGE_SIZE, RULE_ATTR
    from rendering import generate_matrix, render_panel
    from sampling import sample_rules
    from Rule import Rule_Wrapper
    RAVEN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import RAVEN components: {e}")
    print("Make sure RAVEN submodule is properly initialized")
    RAVEN_AVAILABLE = False
    # Define fallback values
    IMAGE_SIZE = 160
    RULE_ATTR = []


class RavenGenerator:
    """Self-contained RAVEN Progressive Matrix task generator."""
    
    # Configuration mapping from paper naming to internal naming
    CONFIGURATIONS = {
        "Center": "center_single",
        "2x2Grid": "distribute_four", 
        "3x3Grid": "distribute_nine",
        "Left-Right": "left_center_single_right_center_single",
        "Up-Down": "up_center_single_down_center_single",
        "Out-InCenter": "in_center_single_out_center_single",
        "Out-InGrid": "in_distribute_four_out_center_single"
    }
    
    # Rule types and their difficulty classification
    RULE_DIFFICULTY = {
        "Constant": "easy",
        "Progression": "medium", 
        "Arithmetic": "hard",
        "Distribute_Three": "medium"
    }
    
    def __init__(self):
        """Initialize RAVEN generator with configurations."""
        self.generated_tasks = []
        self.setup_configurations()
        
    def setup_configurations(self):
        """Setup RAVEN configuration trees."""
        if not RAVEN_AVAILABLE:
            print("Warning: RAVEN components not available, using mock configurations")
            self.config_trees = {}
            return
            
        self.config_trees = {
            "center_single": build_center_single(),
            "distribute_four": build_distribute_four(),
            "distribute_nine": build_distribute_nine(),
            "left_center_single_right_center_single": build_left_center_single_right_center_single(),
            "up_center_single_down_center_single": build_up_center_single_down_center_single(),
            "in_center_single_out_center_single": build_in_center_single_out_center_single(),
            "in_distribute_four_out_center_single": build_in_distribute_four_out_center_single()
        }
        
    def generate_single_task(self, config_name: str, difficulty: str = None) -> Dict[str, Any]:
        """Generate a single RAVEN task."""
        
        if not RAVEN_AVAILABLE:
            return self.generate_mock_task(config_name, difficulty)
            
        # Get configuration tree
        if config_name not in self.config_trees:
            raise ValueError(f"Unknown configuration: {config_name}")
            
        root = self.config_trees[config_name]
        
        # Sample rules for this configuration
        max_attempts = 10
        for attempt in range(max_attempts):
            try:
                rule_groups = sample_rules()
                new_root = root.prune(rule_groups)
                
                if new_root is not None:
                    # Generate the 9 panels following RAVEN logic
                    panels = self.generate_panels(new_root, rule_groups)
                    
                    # Extract rule information for metadata
                    rules_info = self.extract_rule_info(rule_groups)
                    
                    # Determine difficulty if not specified
                    if difficulty is None:
                        difficulty = self.determine_difficulty(rules_info)
                    
                    # Create task data
                    task_data = {
                        "config_name": config_name,
                        "config_display": [k for k, v in self.CONFIGURATIONS.items() if v == config_name][0],
                        "matrix": panels,
                        "rules": rules_info,
                        "difficulty": difficulty,
                        "attempts": attempt + 1
                    }
                    
                    return task_data
                    
            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {config_name}: {e}")
                continue
                
        raise RuntimeError(f"Failed to generate valid task for {config_name} after {max_attempts} attempts")
    
    def generate_panels(self, root, rule_groups) -> List[np.ndarray]:
        """Generate the 9 panels of a Progressive Matrix following RAVEN logic."""
        import copy
        
        # Sample starting node  
        start_node = root.sample()
        
        # Generate Row 1 (panels 1-3)
        row_1_1 = copy.deepcopy(start_node)
        to_merge_row1 = None
        
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
                to_merge_row1 = [row_1_1, row_1_2, row_1_3]
            else:
                self.merge_component(to_merge_row1[1], row_1_2, l)
                self.merge_component(to_merge_row1[2], row_1_3, l)
        
        row_1_1, row_1_2, row_1_3 = to_merge_row1
        
        # Generate Row 2 (panels 4-6)
        row_2_1 = copy.deepcopy(start_node)
        row_2_1.resample(True)
        to_merge_row2 = None
        
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
                to_merge_row2 = [row_2_1, row_2_2, row_2_3]
            else:
                self.merge_component(to_merge_row2[1], row_2_2, l)
                self.merge_component(to_merge_row2[2], row_2_3, l)
                
        row_2_1, row_2_2, row_2_3 = to_merge_row2
        
        # Generate Row 3 (panels 7-9)  
        row_3_1 = copy.deepcopy(start_node)
        row_3_1.resample(True)
        to_merge_row3 = None
        
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
                to_merge_row3 = [row_3_1, row_3_2, row_3_3]
            else:
                self.merge_component(to_merge_row3[1], row_3_2, l)
                self.merge_component(to_merge_row3[2], row_3_3, l)
                
        row_3_1, row_3_2, row_3_3 = to_merge_row3
        
        # Render all 9 panels
        all_nodes = [row_1_1, row_1_2, row_1_3, row_2_1, row_2_2, row_2_3, row_3_1, row_3_2, row_3_3]
        rendered_panels = []
        
        for node in all_nodes:
            panel_image = render_panel(node)
            rendered_panels.append(panel_image)
            
        return rendered_panels
    
    def merge_component(self, dst_aot, src_aot, component_idx):
        """Merge component from src to dst (from RAVEN main.py)."""
        src_component = src_aot.children[0].children[component_idx]
        dst_aot.children[0].children[component_idx] = src_component
    
    def generate_mock_task(self, config_name: str, difficulty: str = None) -> Dict[str, Any]:
        """Generate a mock task when RAVEN components are not available."""
        print(f"Generating mock task for {config_name}")
        
        # Create mock matrix (9 panels of random patterns)
        mock_matrix = []
        for i in range(9):
            # Create a simple pattern for demonstration
            panel = np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8) * 255  # White background
            # Add a simple shape based on panel index
            center = IMAGE_SIZE // 2
            size = 20 + i * 5
            if i < 8:  # First 8 panels
                panel[center-size:center+size, center-size:center+size] = 128  # Gray square
            else:  # 9th panel (solution)
                panel[center-size:center+size, center-size:center+size] = 64   # Darker square
            mock_matrix.append(panel)
        
        # Mock rules info
        mock_rules = {
            "rule_groups": {"component_0": [{"name": "Progression", "attr": "Size", "value": 1}]},
            "primary_rules": ["Progression"],
            "rule_count": 1
        }
        
        config_display = [k for k, v in self.CONFIGURATIONS.items() if v == config_name][0]
        
        return {
            "config_name": config_name,
            "config_display": config_display,
            "matrix": mock_matrix,
            "rules": mock_rules,
            "difficulty": difficulty or "easy",
            "attempts": 1,
            "mock": True
        }
    
    def extract_rule_info(self, rule_groups: List[List]) -> Dict[str, Any]:
        """Extract rule information from rule groups."""
        rules = {}
        primary_rules = []
        
        for i, group in enumerate(rule_groups):
            attr_rules = []
            for rule in group:
                # Convert numpy types to Python types for JSON serialization
                value = rule.value if hasattr(rule, 'value') else None
                if hasattr(value, 'item'):  # numpy scalar
                    value = value.item()
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                
                rule_info = {
                    "name": str(rule.name),
                    "attr": str(rule.attr), 
                    "value": value
                }
                attr_rules.append(rule_info)
                primary_rules.append(str(rule.name))
                
            rules[f"component_{i}"] = attr_rules
            
        return {
            "rule_groups": rules,
            "primary_rules": list(set(primary_rules)),
            "rule_count": int(len(primary_rules))
        }
    
    def determine_difficulty(self, rules_info: Dict[str, Any]) -> str:
        """Determine task difficulty based on rules."""
        primary_rules = rules_info.get("primary_rules", [])
        
        # Complex rules or multiple rules increase difficulty
        if "Arithmetic" in primary_rules:
            return "hard"
        elif len(primary_rules) > 2 or "Distribute_Three" in primary_rules:
            return "medium"
        else:
            return "easy"
    
    def generate_tasks(self, num_tasks: int = 50) -> List[Dict[str, Any]]:
        """Generate RAVEN tasks across all configurations."""
        print(f"🎯 Generating {num_tasks} RAVEN Progressive Matrix tasks...")
        
        tasks = []
        config_names = list(self.CONFIGURATIONS.values())
        
        for i in range(num_tasks):
            # Distribute tasks across configurations
            config_name = config_names[i % len(config_names)]
            
            try:
                task_data = self.generate_single_task(config_name)
                tasks.append(task_data)
                print(f"✅ Generated task {i+1}/{num_tasks}: {task_data['config_display']} ({task_data['difficulty']})")
                
            except Exception as e:
                print(f"❌ Failed to generate task {i+1}: {e}")
                # Try with a different configuration
                for alt_config in config_names:
                    try:
                        task_data = self.generate_single_task(alt_config)
                        tasks.append(task_data)
                        print(f"✅ Generated task {i+1}/{num_tasks} (fallback): {task_data['config_display']}")
                        break
                    except:
                        continue
                        
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
    config_name = task_data["config_name"]
    
    # Create output directory
    image_dir = os.path.join(output_dir, "data", "questions", "generated_raven")
    os.makedirs(image_dir, exist_ok=True)
    
    # Image paths
    first_image_path = os.path.join(image_dir, f"{task_id}_first.png")
    final_image_path = os.path.join(image_dir, f"{task_id}_final.png")
    
    # Generate incomplete matrix (first 8 panels + empty 9th)
    generate_rpm_image(matrix[:8], first_image_path, incomplete=True)
    
    # Generate complete matrix (all 9 panels)
    generate_rpm_image(matrix, final_image_path, incomplete=False)
    
    return f"data/questions/generated_raven/{task_id}_first.png", f"data/questions/generated_raven/{task_id}_final.png"


def generate_rpm_image(matrix_panels: List[np.ndarray], output_path: str, incomplete: bool = False):
    """Generate RPM visualization as PNG image."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    
    # Create figure with 3x3 grid
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.patch.set_facecolor('white')
    
    # Fill known panels
    panel_idx = 0
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            ax.set_xlim(0, IMAGE_SIZE)
            ax.set_ylim(0, IMAGE_SIZE)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Add border
            border = Rectangle((0, 0), IMAGE_SIZE, IMAGE_SIZE, 
                             linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(border)
            
            if panel_idx < len(matrix_panels):
                # Render the panel content
                panel = matrix_panels[panel_idx]
                ax.imshow(panel, cmap='gray', vmin=0, vmax=255)
                panel_idx += 1
            elif incomplete and i == 2 and j == 2:
                # Empty panel with question mark for incomplete matrices
                ax.text(IMAGE_SIZE//2, IMAGE_SIZE//2, '?', 
                       fontsize=60, ha='center', va='center', 
                       color='gray', weight='bold')
                ax.set_facecolor('#f0f0f0')
            elif not incomplete and panel_idx < len(matrix_panels):
                # Fill remaining panels for complete matrix
                panel = matrix_panels[panel_idx]
                ax.imshow(panel, cmap='gray', vmin=0, vmax=255)
                panel_idx += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()


def generate_prompt(task_data: Dict[str, Any]) -> str:
    """Generate task prompt based on task data."""
    
    config_display = task_data["config_display"]
    difficulty = task_data["difficulty"]
    rules = task_data["rules"]["primary_rules"]
    
    # Base prompts by configuration
    config_prompts = {
        "Center": "Complete this center-focused pattern matrix",
        "2x2Grid": "Complete this 2x2 grid pattern matrix", 
        "3x3Grid": "Complete this 3x3 grid pattern matrix",
        "Left-Right": "Complete this left-right pattern matrix",
        "Up-Down": "Complete this up-down pattern matrix",
        "Out-InCenter": "Complete this outside-inside center pattern matrix",
        "Out-InGrid": "Complete this outside-inside grid pattern matrix"
    }
    
    base_prompt = config_prompts.get(config_display, "Complete this pattern matrix")
    
    # Add rule hints for harder tasks
    rule_hints = {
        "Progression": "Look for progressive changes",
        "Arithmetic": "Consider mathematical relationships", 
        "Distribute_Three": "Notice the three-way distribution",
        "Constant": "Some elements remain constant"
    }
    
    hints = [rule_hints[rule] for rule in rules if rule in rule_hints]
    
    if hints and difficulty in ["medium", "hard"]:
        hint_text = ". " + ". ".join(hints)
    else:
        hint_text = ""
    
    full_prompt = f"{base_prompt}. Show the logical reasoning process to determine what goes in the missing 9th panel{hint_text}."
    
    return full_prompt


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
            "rule_groups": task_data["rules"]["rule_groups"],
            "primary_rules": task_data["rules"]["primary_rules"],
            "matrix_size": f"{IMAGE_SIZE}x{IMAGE_SIZE}",
            "pattern_type": "Progressive Matrix"
        },
        "difficulty": task_data["difficulty"],
        "rule_types": task_data["rules"]["primary_rules"],
        "configuration_type": task_data["config_display"],
        "created_at": datetime.now().isoformat()
    }
    
    return task_pair


def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """Create complete RAVEN dataset."""
    
    print(f"🎯 Creating RAVEN Progressive Matrix dataset with {num_samples} samples...")
    
    # Generate tasks
    generator = RavenGenerator()
    tasks = generator.generate_tasks(num_samples)
    
    if len(tasks) == 0:
        raise RuntimeError("Failed to generate any valid RAVEN tasks")
    
    # Create task pairs
    pairs = []
    for i, task_data in enumerate(tasks):
        task_id = f"raven_{i:04d}"
        
        try:
            pair = create_task_pair(task_data, task_id)
            pairs.append(pair)
            print(f"✅ Created task {task_id}: {pair['task_category']} ({pair['difficulty']})")
        except Exception as e:
            print(f"❌ Failed to create task pair {task_id}: {e}")
            continue
    
    if len(pairs) == 0:
        raise RuntimeError("Failed to create any valid task pairs")
    
    # Create dataset
    dataset = {
        "name": "raven_tasks",
        "description": f"RAVEN Progressive Matrix reasoning tasks for video model evaluation ({len(pairs)} pairs)",
        "pairs": pairs
    }
    
    # Save dataset
    base_dir = Path(__file__).parent.parent.parent.parent
    dataset_dir = base_dir / "data" / "questions" / "raven_tasks"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    output_path = dataset_dir / "raven_tasks.json"
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Saved dataset: {output_path}")
    print(f"📊 Dataset stats:")
    
    # Print statistics
    difficulties = {}
    categories = {}
    for pair in pairs:
        diff = pair['difficulty']
        cat = pair['task_category']
        difficulties[diff] = difficulties.get(diff, 0) + 1
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"   Difficulties: {difficulties}")
    print(f"   Categories: {categories}")
    
    return dataset


def main():
    """Generate RAVEN Progressive Matrix dataset."""
    try:
        dataset = create_dataset(num_samples=50)
        print(f"🚀 RAVEN Progressive Matrix reasoning dataset ready!")
        print(f"📁 Generated {len(dataset['pairs'])} task pairs")
        
    except Exception as e:
        print(f"❌ Error generating RAVEN dataset: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
