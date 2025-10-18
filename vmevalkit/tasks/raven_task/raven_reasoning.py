"""
Raven Progressive Matrices reasoning task for VMEvalKit.

This module generates RPM-style visual reasoning puzzles where video models
must demonstrate abstract pattern recognition by showing the reasoning process
from an incomplete matrix to the correct solution.
"""

import os
import json
from typing import Dict, Any, List
from datetime import datetime
from .rpm_generator import RPMPuzzleGenerator
from PIL import Image, ImageDraw, ImageFont

# Import prompts from centralized location
from .PROMPTS import PROMPTS


def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """
    Generate Raven Progressive Matrices dataset for VMEvalKit.
    
    Creates task pairs where:
    - First frame: Incomplete 3x3 matrix with missing bottom-right cell
    - Final frame: Complete matrix with correct pattern filled in
    - Prompt: Instructions to complete the pattern
    
    Args:
        num_samples: Number of puzzle pairs to generate
        
    Returns:
        Dataset dictionary with all task pairs
    """
    
    print(f"🧩 Generating {num_samples} Raven Progressive Matrices puzzles...")
    
    pairs = []
    generator = RPMPuzzleGenerator(tile_size=150)  # Smaller tiles for 450x450 total
    
    for i in range(num_samples):
        task_id = f"raven_{i:04d}"
        
        # Use temporary directory like other tasks
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Generate puzzle with unique seed
        seed = 2025 + i
        generator.rng.seed(seed)
        
        # Generate the pattern matrix and metadata
        matrix, rule = generator.generate_pattern_matrix()
        
        # Create and save first frame (incomplete matrix) in temp directory
        first_frame = generator.render_matrix(matrix, hide_last=True)
        first_frame_path = os.path.join(temp_dir, f"{task_id}_first.png")
        first_frame.save(first_frame_path)
        
        # Create and save final frame (complete matrix) in temp directory
        final_frame = generator.render_matrix(matrix, hide_last=False)
        final_frame_path = os.path.join(temp_dir, f"{task_id}_final.png")
        final_frame.save(final_frame_path)
        
        # Generate prompt (using PROMPTS[0] as default)
        prompt = PROMPTS[0]
        
        # Determine difficulty based on rule type
        difficulty_map = {
            "shape_progression": "easy",
            "number_progression": "medium", 
            "rotation": "medium",
            "color_pattern": "easy",
            "combination": "hard"
        }
        
        rule_type = rule.split()[0].lower()
        difficulty = difficulty_map.get(rule_type, "medium")
        
        # Create task pair metadata (return temp paths that will be moved by create_dataset.py)
        pair = {
            "id": task_id,
            "prompt": prompt,
            "first_image_path": first_frame_path,  # temp path
            "final_image_path": final_frame_path,  # temp path
            "domain": "raven",
            "task_category": "AbstractReasoning",
            "difficulty": difficulty,
            "raven_data": {
                "rule": rule,
                "rule_type": rule_type,
                "matrix_size": "3x3",
                "seed": seed
            },
            "created_at": datetime.now().isoformat()
        }
        
        pairs.append(pair)
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_samples} puzzles...")
    
    print(f"✅ Successfully generated {len(pairs)} Raven Progressive Matrices puzzles")
    
    # Return dataset dictionary
    return {
        "name": "raven_tasks",
        "description": f"Raven Progressive Matrices visual reasoning tasks ({len(pairs)} pairs)",
        "created_at": datetime.now().isoformat(),
        "total_pairs": len(pairs),
        "pairs": pairs,
        "generation_info": {
            "generator": "RPMPuzzleGenerator",
            "tile_size": 150,
            "matrix_size": "3x3",
            "difficulty_distribution": {
                "easy": sum(1 for p in pairs if p["difficulty"] == "easy"),
                "medium": sum(1 for p in pairs if p["difficulty"] == "medium"),
                "hard": sum(1 for p in pairs if p["difficulty"] == "hard")
            }
        }
    }


def visualize_solution_process(task_id: str, output_dir: str = "output/raven_solutions"):
    """
    Create a visualization showing the solution process for a Raven puzzle.
    
    This creates an image showing:
    1. The incomplete matrix (first frame)
    2. The complete solution (final frame)
    
    Args:
        task_id: The task ID (e.g., "raven_0001")
        output_dir: Directory to save the visualization
    """
    
    base_dir = f"data/questions/raven_task/{task_id}"
    
    # Load images
    first_frame = Image.open(os.path.join(base_dir, "first_frame.png"))
    final_frame = Image.open(os.path.join(base_dir, "final_frame.png"))
    
    # Load metadata
    with open(os.path.join(base_dir, "question_metadata.json"), 'r') as f:
        metadata = json.load(f)
    
    # Create combined visualization
    width = max(first_frame.width, final_frame.width)
    height = first_frame.height + final_frame.height + 80
    
    viz = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(viz)
    
    # Add title
    y_offset = 10
    draw.text((10, y_offset), f"Raven Matrix Puzzle: {task_id}", fill="black")
    y_offset += 30
    
    # Add incomplete matrix with label
    draw.text((10, y_offset), "Initial State (find the missing pattern):", fill="black")
    y_offset += 20
    viz.paste(first_frame, (0, y_offset))
    y_offset += first_frame.height + 10
    
    # Add solution with label
    draw.text((10, y_offset), "Solution (completed pattern):", fill="green")
    y_offset += 20
    viz.paste(final_frame, (0, y_offset))
    
    # Save visualization
    os.makedirs(output_dir, exist_ok=True)
    viz_path = os.path.join(output_dir, f"{task_id}_solution.png")
    viz.save(viz_path)
    
    print(f"Saved solution visualization to {viz_path}")
    return viz_path


# For backward compatibility and testing
def generate_raven_tasks(num_tasks: int = 10, output_dir: str = "output/raven_matrices"):
    """
    Legacy function for generating Raven tasks.
    Redirects to create_dataset for VMEvalKit compatibility.
    """
    dataset = create_dataset(num_samples=num_tasks)
    print(f"Generated {len(dataset['pairs'])} Raven Progressive Matrices tasks")
    return dataset
