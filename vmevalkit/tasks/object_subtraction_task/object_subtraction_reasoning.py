"""
Object Subtraction Reasoning Task for VMEvalKit

Multi-level cognitive reasoning benchmark where video generation models must
remove specific objects while keeping others stationary.

Level 1: Explicit Specificity - Remove objects by explicit visual attributes (color, shape)

Author: VMEvalKit Team
"""

import json
import random
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, RegularPolygon, FancyBboxPatch

# Import prompts from centralized location
from .PROMPTS import PROMPTS_L1, DEFAULT_PROMPT_INDEX_L1


@dataclass
class ObjectSubtractionTaskPair:
    """
    Data structure for object subtraction video model evaluation.
    
    Contains:
    - prompt: Instructions for the video model
    - first_image: The initial scene with all objects
    - final_image: The scene with specified objects removed
    """
    id: str
    prompt: str                      # What to ask the video model to do
    first_image_path: str           # The initial scene image (all objects)
    final_image_path: str           # The final scene image (objects removed)
    task_category: str = "ObjectSubtraction"
    level: str = "L1"               # "L1", "L2", "L3", "L4"
    object_subtraction_data: Dict[str, Any] = None  # Metadata (rules, objects, etc.)
    difficulty: str = "easy"        # "easy", "medium", "hard"
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class ObjectGenerator:
    """Generate objects with colors, shapes, and positions."""
    
    COLORS = ["red", "green", "blue", "yellow", "orange", "purple"]
    SHAPES = ["cube", "sphere", "pyramid", "cone"]
    
    def __init__(self, canvas_size: Tuple[int, int] = (256, 256), 
                 min_size: int = 20, max_size: int = 40,
                 min_spacing: int = 10):
        """
        Initialize object generator.
        
        Args:
            canvas_size: (width, height) of the canvas
            min_size: Minimum object size in pixels
            max_size: Maximum object size in pixels
            min_spacing: Minimum spacing between objects
        """
        self.canvas_size = canvas_size
        self.min_size = min_size
        self.max_size = max_size
        self.min_spacing = min_spacing
        self.rng = random.Random()
    
    def generate_objects(self, num_objects: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate objects with random colors, shapes, and positions.
        Ensures objects don't overlap.
        
        Args:
            num_objects: Number of objects to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of object dictionaries with id, color, shape, x, y, size, area
        """
        if seed is not None:
            self.rng.seed(seed)
        
        objects = []
        canvas_w, canvas_h = self.canvas_size
        
        for i in range(num_objects):
            # Try to place object (with collision detection)
            max_attempts = 100
            placed = False
            
            for attempt in range(max_attempts):
                # Random size
                size = self.rng.randint(self.min_size, self.max_size)
                
                # Random position (with margin for object size)
                margin = size // 2 + self.min_spacing
                x = self.rng.randint(margin, canvas_w - margin)
                y = self.rng.randint(margin, canvas_h - margin)
                
                # Check collision with existing objects
                collision = False
                for obj in objects:
                    dx = x - obj["x"]
                    dy = y - obj["y"]
                    distance = np.sqrt(dx*dx + dy*dy)
                    min_distance = (size + obj["size"]) // 2 + self.min_spacing
                    
                    if distance < min_distance:
                        collision = True
                        break
                
                if not collision:
                    # Random color and shape
                    color = self.rng.choice(self.COLORS)
                    shape = self.rng.choice(self.SHAPES)
                    
                    # Calculate area (approximate)
                    if shape == "cube":
                        area = size * size
                    elif shape == "sphere":
                        area = int(np.pi * (size // 2) ** 2)
                    elif shape == "pyramid":
                        area = int(size * size * 0.433)  # Equilateral triangle
                    else:  # cone
                        area = int(np.pi * (size // 2) ** 2)
                    
                    obj = {
                        "id": i,
                        "color": color,
                        "shape": shape,
                        "x": x,
                        "y": y,
                        "size": size,
                        "area": area
                    }
                    objects.append(obj)
                    placed = True
                    break
            
            if not placed:
                # If we can't place, use a grid-based fallback
                grid_size = int(np.ceil(np.sqrt(num_objects)))
                cell_w = canvas_w // (grid_size + 1)
                cell_h = canvas_h // (grid_size + 1)
                row = i // grid_size
                col = i % grid_size
                x = cell_w * (col + 1)
                y = cell_h * (row + 1)
                
                size = self.rng.randint(self.min_size, self.max_size)
                color = self.rng.choice(self.COLORS)
                shape = self.rng.choice(self.SHAPES)
                
                if shape == "cube":
                    area = size * size
                elif shape == "sphere":
                    area = int(np.pi * (size // 2) ** 2)
                elif shape == "pyramid":
                    area = int(size * size * 0.433)
                else:
                    area = int(np.pi * (size // 2) ** 2)
                
                obj = {
                    "id": i,
                    "color": color,
                    "shape": shape,
                    "x": x,
                    "y": y,
                    "size": size,
                    "area": area
                }
                objects.append(obj)
        
        return objects


class SceneRenderer:
    """Render scene images with objects."""
    
    COLOR_MAP = {
        "red": "#FF0000",
        "green": "#00FF00",
        "blue": "#0000FF",
        "yellow": "#FFFF00",
        "orange": "#FFA500",
        "purple": "#800080"
    }
    
    def __init__(self, canvas_size: Tuple[int, int] = (256, 256), dpi: int = 100):
        self.canvas_size = canvas_size
        self.dpi = dpi
    
    def render_scene(self, objects: List[Dict[str, Any]], 
                    output_path: Union[str, Path],
                    show_object_ids: Optional[List[int]] = None) -> None:
        """
        Render a scene with objects.
        
        Args:
            objects: List of object dictionaries
            show_object_ids: If provided, only show objects with these IDs
            output_path: Path to save the image
        """
        fig, ax = plt.subplots(1, 1, figsize=(self.canvas_size[0]/self.dpi, 
                                               self.canvas_size[1]/self.dpi), 
                              dpi=self.dpi)
        
        # Filter objects if needed
        if show_object_ids is not None:
            objects_to_render = [obj for obj in objects if obj["id"] in show_object_ids]
        else:
            objects_to_render = objects
        
        # Draw each object
        for obj in objects_to_render:
            color = self.COLOR_MAP.get(obj["color"], "#000000")
            x = obj["x"]
            y = obj["y"]
            size = obj["size"]
            shape = obj["shape"]
            
            if shape == "cube":
                # Draw square
                rect = Rectangle((x - size//2, y - size//2), size, size,
                               facecolor=color, edgecolor="black", linewidth=2)
                ax.add_patch(rect)
            elif shape == "sphere":
                # Draw circle
                circle = Circle((x, y), size//2, facecolor=color, 
                              edgecolor="black", linewidth=2)
                ax.add_patch(circle)
            elif shape == "pyramid":
                # Draw triangle (equilateral)
                triangle = RegularPolygon((x, y), 3, radius=size//2,
                                        orientation=np.pi/6,
                                        facecolor=color, edgecolor="black", linewidth=2)
                ax.add_patch(triangle)
            elif shape == "cone":
                # Draw trapezoid (cone-like)
                half_size = size // 2
                points = [
                    (x - half_size, y - half_size),
                    (x + half_size, y - half_size),
                    (x + half_size//2, y + half_size),
                    (x - half_size//2, y + half_size)
                ]
                polygon = patches.Polygon(points, facecolor=color, 
                                        edgecolor="black", linewidth=2)
                ax.add_patch(polygon)
        
        # Set up axes
        ax.set_xlim(0, self.canvas_size[0])
        ax.set_ylim(0, self.canvas_size[1])
        ax.set_aspect('equal')
        ax.axis('off')
        # Note: Matplotlib uses bottom-left origin, but we want top-left origin for image coordinates
        # We'll invert y-axis to match image coordinate system
        ax.invert_yaxis()
        
        # Save figure
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)


class RuleGenerator:
    """Generate rules for different cognitive levels."""
    
    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random()
    
    def generate_l1_rule(self, objects: List[Dict[str, Any]], 
                        prompt_index: int = 0) -> Tuple[Dict[str, Any], str]:
        """
        Generate Level 1 rule: Remove objects by explicit visual attributes.
        
        Args:
            objects: List of objects
            prompt_index: Index into PROMPTS_L1 list
            
        Returns:
            (rule_dict, prompt_text)
        """
        # Choose a removal criterion (color or shape)
        criterion_type = self.rng.choice(["color", "shape"])
        
        if criterion_type == "color":
            # Find all unique colors
            colors = list(set(obj["color"] for obj in objects))
            if len(colors) < 2:
                # Fallback to shape if not enough colors
                criterion_type = "shape"
        
        if criterion_type == "color":
            # Select a color to remove
            colors = list(set(obj["color"] for obj in objects))
            remove_color = self.rng.choice(colors)
            
            # Find all objects with this color
            target_object_ids = [obj["id"] for obj in objects if obj["color"] == remove_color]
            
            rule = {
                "level": "L1",
                "rule_type": "color",
                "remove_color": remove_color,
                "target_object_ids": target_object_ids
            }
            
            # Generate prompt
            prompt = PROMPTS_L1[prompt_index % len(PROMPTS_L1)]
            # Replace color in prompt if needed
            if "red" in prompt.lower():
                prompt = prompt.replace("red", remove_color)
            elif "blue" in prompt.lower():
                prompt = prompt.replace("blue", remove_color)
            elif "green" in prompt.lower():
                prompt = prompt.replace("green", remove_color)
            elif "yellow" in prompt.lower():
                prompt = prompt.replace("yellow", remove_color)
            else:
                # Use generic prompt
                prompt = f"Remove all {remove_color} objects from the scene. Keep all other objects in their exact positions."
        
        else:  # shape
            # Find all unique shapes
            shapes = list(set(obj["shape"] for obj in objects))
            remove_shape = self.rng.choice(shapes)
            
            # Find all objects with this shape
            target_object_ids = [obj["id"] for obj in objects if obj["shape"] == remove_shape]
            
            rule = {
                "level": "L1",
                "rule_type": "shape",
                "remove_shape": remove_shape,
                "target_object_ids": target_object_ids
            }
            
            # Generate prompt
            shape_name = remove_shape.capitalize()
            prompt = f"Remove all {shape_name.lower()} objects from the scene. Keep all other objects in their exact positions."
        
        return rule, prompt


class ObjectSubtractionGenerator:
    """Main generator for object subtraction tasks."""
    
    def __init__(self, canvas_size: Tuple[int, int] = (256, 256),
                 num_objects_range: Tuple[int, int] = (5, 8)):
        """
        Initialize generator.
        
        Args:
            canvas_size: (width, height) of canvas
            num_objects_range: (min, max) number of objects per scene
        """
        self.canvas_size = canvas_size
        self.num_objects_range = num_objects_range
        self.object_gen = ObjectGenerator(canvas_size=canvas_size)
        self.renderer = SceneRenderer(canvas_size=canvas_size)
        self.rule_gen = RuleGenerator()
    
    def generate_single_task(self, task_id: str, level: str = "L1", 
                            seed: Optional[int] = None) -> ObjectSubtractionTaskPair:
        """
        Generate a single object subtraction task.
        
        Args:
            task_id: Unique task identifier
            level: Cognitive level ("L1", "L2", "L3", "L4")
            seed: Random seed for reproducibility
            
        Returns:
            ObjectSubtractionTaskPair
        """
        # Use temporary directory like other tasks
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            self.rule_gen.rng.seed(seed)
            self.object_gen.rng.seed(seed)
        
        # Generate objects
        num_objects = random.randint(self.num_objects_range[0], self.num_objects_range[1])
        objects = self.object_gen.generate_objects(num_objects, seed=seed)
        
        # Generate rule and prompt (Level 1 only for now)
        if level == "L1":
            rule, prompt = self.rule_gen.generate_l1_rule(objects, 
                                                          prompt_index=DEFAULT_PROMPT_INDEX_L1)
        else:
            raise ValueError(f"Level {level} not yet implemented")
        
        # Get target object IDs to remove
        remove_ids = rule["target_object_ids"]
        keep_ids = [obj["id"] for obj in objects if obj["id"] not in remove_ids]
        
        # Render first frame (all objects)
        first_path = Path(temp_dir) / f"{task_id}_first.png"
        self.renderer.render_scene(objects, output_path=first_path, show_object_ids=None)
        
        # Render final frame (only kept objects)
        final_path = Path(temp_dir) / f"{task_id}_final.png"
        self.renderer.render_scene(objects, output_path=final_path, show_object_ids=keep_ids)
        
        # Create task pair
        task_pair = ObjectSubtractionTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(first_path),
            final_image_path=str(final_path),
            task_category="ObjectSubtraction",
            level=level,
            object_subtraction_data={
                "objects": objects,
                "rule": rule,
                "remove_object_ids": remove_ids,
                "keep_object_ids": keep_ids,
                "num_objects": num_objects,
                "num_removed": len(remove_ids),
                "num_kept": len(keep_ids)
            },
            difficulty="easy" if level == "L1" else "medium",
            created_at=datetime.now().isoformat()
        )
        
        return task_pair


def create_dataset(num_samples: int = 50, levels: List[str] = ["L1"]) -> Dict[str, Any]:
    """
    Create object subtraction dataset - main entry point matching other tasks.
    
    Args:
        num_samples: Number of tasks to generate
        levels: List of cognitive levels to generate ("L1", "L2", "L3", "L4")
        
    Returns:
        Dataset dictionary in standard format
    """
    print(f"🎯 Creating Object Subtraction Dataset")
    print(f"   Total samples: {num_samples}")
    print(f"   Levels: {', '.join(levels)}")
    
    start_time = datetime.now()
    
    generator = ObjectSubtractionGenerator()
    pairs = []
    
    # Distribute samples across levels
    samples_per_level = num_samples // len(levels)
    remainder = num_samples % len(levels)
    
    for level_idx, level in enumerate(levels):
        level_samples = samples_per_level + (1 if level_idx < remainder else 0)
        
        print(f"\n📊 Generating {level_samples} tasks for Level {level}...")
        
        for i in range(level_samples):
            task_id = f"object_subtraction_{level.lower()}_{i:04d}"
            seed = 2025 + level_idx * 10000 + i  # Deterministic seed
            
            try:
                task_pair = generator.generate_single_task(task_id, level=level, seed=seed)
                pairs.append(task_pair)
                
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{level_samples} tasks...")
            except Exception as e:
                print(f"❌ Error generating task {task_id}: {e}")
                continue
    
    # Convert to dictionary format for consistency with other tasks
    pairs_dict = []
    for pair in pairs:
        pair_dict = {
            "id": pair.id,
            "prompt": pair.prompt,
            "first_image_path": pair.first_image_path,
            "final_image_path": pair.final_image_path,
            "task_category": pair.task_category,
            "level": pair.level,
            "object_subtraction_data": pair.object_subtraction_data,
            "difficulty": pair.difficulty,
            "created_at": pair.created_at
        }
        pairs_dict.append(pair_dict)
    
    # Create dataset dictionary
    dataset = {
        "name": "object_subtraction_tasks",
        "description": f"Object subtraction reasoning tasks for video model evaluation ({len(pairs)} pairs, levels: {', '.join(levels)})",
        "pairs": pairs_dict,
        "metadata": {
            "total_tasks": len(pairs),
            "levels": levels,
            "canvas_size": generator.canvas_size,
            "num_objects_range": generator.num_objects_range,
            "generation_date": datetime.now().isoformat()
        },
        "created_at": datetime.now().isoformat()
    }
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n✅ Dataset creation complete!")
    print(f"   Total tasks: {len(pairs)}")
    print(f"   Time elapsed: {elapsed:.1f}s")
    
    return dataset

