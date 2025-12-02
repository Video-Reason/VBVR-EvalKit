"""
Object Subtraction Reasoning Task for VMEvalKit

Multi-level cognitive reasoning benchmark where video generation models must
remove specific objects while keeping others stationary.

Type 1: Explicit Specificity - Remove objects by explicit visual attributes (color, shape)

Author: VMEvalKit Team
"""

import json
import random
import hashlib
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, RegularPolygon, FancyBboxPatch

# Import prompts from centralized location
from .PROMPTS import PROMPTS_TYPE1, PROMPTS_TYPE2, PROMPTS_TYPE3, PROMPTS_TYPE4, DEFAULT_PROMPT_INDEX_TYPE1, DEFAULT_PROMPT_INDEX_TYPE2, DEFAULT_PROMPT_INDEX_TYPE3, DEFAULT_PROMPT_INDEX_TYPE4


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
    level: str = "type1"               # "type1", "type2", "type3", "type4"
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
    
    def generate_type1_rule(self, objects: List[Dict[str, Any]], 
                        prompt_index: int = 0,
                        task_type_hint: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
        """
        Generate Type 1 rule: Remove objects by explicit visual attributes.
        Supports: color, shape only (size-based rules removed)
        
        Args:
            objects: List of objects
            prompt_index: Index into PROMPTS_TYPE1 list
            task_type_hint: Optional hint for task type ("color" or "shape") for balanced distribution
            
        Returns:
            (rule_dict, prompt_text)
        """
        # Choose a removal criterion (color or shape only)
        # Use hint if provided, otherwise random choice
        if task_type_hint in ["color", "shape"]:
            criterion_type = task_type_hint
        else:
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
                "level": "type1",
                "rule_type": "color",
                "remove_color": remove_color,
                "target_object_ids": target_object_ids
            }
            
            # Generate prompt - always use generic format
            prompt = f"Remove all {remove_color} objects from the scene. Do not do anything to other objects."
        
        elif criterion_type == "shape":
            # Find all unique shapes
            shapes = list(set(obj["shape"] for obj in objects))
            remove_shape = self.rng.choice(shapes)
            
            # Find all objects with this shape
            target_object_ids = [obj["id"] for obj in objects if obj["shape"] == remove_shape]
            
            rule = {
                "level": "type1",
                "rule_type": "shape",
                "remove_shape": remove_shape,
                "target_object_ids": target_object_ids
            }
            
            # Generate prompt
            shape_name = remove_shape.capitalize()
            prompt = f"Remove all {shape_name.lower()} objects from the scene. Do not do anything to other objects."
        
        return rule, prompt
    
    def _calculate_spatial_relations(self, objects: List[Dict[str, Any]], 
                                   canvas_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        Calculate spatial relations for all objects.
        Uses stricter boundaries to ensure spatial relations are visually clear.
        
        Args:
            objects: List of object dictionaries
            canvas_size: (width, height) of canvas
            
        Returns:
            Dictionary with spatial relation information
        """
        canvas_w, canvas_h = canvas_size
        center_x, center_y = canvas_w // 2, canvas_h // 2
        
        # Very strict boundaries to ensure maximum visual clarity and accuracy
        # For 256x256 canvas: center = 128
        # Use much tighter thresholds to ensure objects are clearly in their regions
        # Left: x < 64 (left 25%), Right: x > 192 (right 25%)
        # Top: y < 64 (top 25%), Bottom: y > 192 (bottom 25%)
        # This ensures objects are clearly positioned, not ambiguous
        margin_ratio = 0.25  # Use 25% margin from center for very clear distinction
        left_threshold = int(center_x * (1 - margin_ratio))  # 96 for 256 canvas
        right_threshold = int(center_x * (1 + margin_ratio))  # 160 for 256 canvas
        top_threshold = int(center_y * (1 - margin_ratio))  # 96 for 256 canvas
        bottom_threshold = int(center_y * (1 + margin_ratio))  # 160 for 256 canvas
        quadrant_margin = int(center_x * 0.3)  # 30% margin = 77px for 256 canvas, very strict
        
        relations = {
            "center": (center_x, center_y),
            "left_objects": [],  # x < center_x (loose, for general classification)
            "right_objects": [],  # x >= center_x (loose)
            "top_objects": [],  # y < center_y (loose)
            "bottom_objects": [],  # y >= center_y (loose)
            "clear_left_objects": [],  # x < left_threshold (strict, for selection)
            "clear_right_objects": [],  # x > right_threshold (strict)
            "clear_top_objects": [],  # y < top_threshold (strict)
            "clear_bottom_objects": [],  # y > bottom_threshold (strict)
            "distances_from_center": {},
            "quadrants": {},
            "clear_quadrants": {},  # Objects clearly in quadrants (with margin)
            "x_sorted_objects": [],  # Objects sorted by x coordinate (ascending)
            "y_sorted_objects": [],  # Objects sorted by y coordinate (ascending)
            "corner_distances": {}  # Distance to nearest corner for each object
        }
        
        for obj in objects:
            x, y = obj["x"], obj["y"]
            
            # Loose Left/right classification (for general reference)
            if x < center_x:
                relations["left_objects"].append(obj["id"])
            else:
                relations["right_objects"].append(obj["id"])
            
            # Strict Left/right classification (for clear visual distinction)
            if x < left_threshold:
                relations["clear_left_objects"].append(obj["id"])
            elif x > right_threshold:
                relations["clear_right_objects"].append(obj["id"])
            
            # Loose Top/bottom classification
            if y < center_y:
                relations["top_objects"].append(obj["id"])
            else:
                relations["bottom_objects"].append(obj["id"])
            
            # Strict Top/bottom classification
            if y < top_threshold:
                relations["clear_top_objects"].append(obj["id"])
            elif y > bottom_threshold:
                relations["clear_bottom_objects"].append(obj["id"])
            
            # Distance from center
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            relations["distances_from_center"][obj["id"]] = distance
            
            # Loose Quadrant classification
            if x < center_x and y < center_y:
                relations["quadrants"][obj["id"]] = "top_left"
            elif x >= center_x and y < center_y:
                relations["quadrants"][obj["id"]] = "top_right"
            elif x < center_x and y >= center_y:
                relations["quadrants"][obj["id"]] = "bottom_left"
            else:
                relations["quadrants"][obj["id"]] = "bottom_right"
            
            # Strict Quadrant classification (with margin from center lines)
            if x < center_x - quadrant_margin and y < center_y - quadrant_margin:
                relations["clear_quadrants"][obj["id"]] = "top_left"
            elif x > center_x + quadrant_margin and y < center_y - quadrant_margin:
                relations["clear_quadrants"][obj["id"]] = "top_right"
            elif x < center_x - quadrant_margin and y > center_y + quadrant_margin:
                relations["clear_quadrants"][obj["id"]] = "bottom_left"
            elif x > center_x + quadrant_margin and y > center_y + quadrant_margin:
                relations["clear_quadrants"][obj["id"]] = "bottom_right"
            
            # Calculate distance to nearest corner
            corners = [
                (0, 0),  # top-left
                (canvas_w, 0),  # top-right
                (0, canvas_h),  # bottom-left
                (canvas_w, canvas_h)  # bottom-right
            ]
            min_corner_dist = min([np.sqrt((x - cx)**2 + (y - cy)**2) for cx, cy in corners])
            relations["corner_distances"][obj["id"]] = min_corner_dist
        
        # Sort objects by x and y coordinates for leftmost/rightmost/topmost/bottommost selection
        relations["x_sorted_objects"] = sorted(objects, key=lambda o: o["x"])
        relations["y_sorted_objects"] = sorted(objects, key=lambda o: o["y"])
        
        return relations
    
    def generate_type3_rule(self, objects: List[Dict[str, Any]], 
                         canvas_size: Tuple[int, int],
                         prompt_index: int = 0,
                         relation_type_hint: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
        """
        Generate Type 3 rule: Remove objects using spatial relations.
        
        Args:
            objects: List of objects
            canvas_size: (width, height) of canvas
            prompt_index: Index into PROMPTS_TYPE3 list
            relation_type_hint: Optional hint for relation type ("leftmost", "rightmost", "topmost", "bottommost") for balanced distribution
            
        Returns:
            (rule_dict, prompt_text)
        """
        # Ensure we have enough objects (at least 3 for Type 3)
        if len(objects) < 3:
            return self.generate_type1_rule(objects, prompt_index)
        
        # Calculate spatial relations
        relations = self._calculate_spatial_relations(objects, canvas_size)
        
        # Define available spatial relation types (using strict boundaries + sorting)
        relation_types = []
        
        # Check which relation types are available
        # Only keep edge-based relations (leftmost, rightmost, topmost, bottommost)
        if len(relations["clear_left_objects"]) > 0:
            relation_types.append("leftmost")
        if len(relations["clear_right_objects"]) > 0:
            relation_types.append("rightmost")
        if len(relations["clear_top_objects"]) > 0:
            relation_types.append("topmost")
        if len(relations["clear_bottom_objects"]) > 0:
            relation_types.append("bottommost")
        
        if not relation_types:
            return self.generate_type1_rule(objects, prompt_index)
        
        # Randomly select a relation type
        # Use hint if provided and available, otherwise random choice
        if relation_type_hint in relation_types:
            relation_type = relation_type_hint
        else:
            relation_type = self.rng.choice(relation_types)
        
        # Select target objects based on relation type (using strict boundaries + sorting)
        target_object_ids = []
        num_to_remove = 0
        
        if relation_type == "leftmost":
            # Use sorting: select the N leftmost objects that are clearly on the left
            candidates = [obj for obj in relations["x_sorted_objects"] 
                         if obj["id"] in relations["clear_left_objects"]]
            if not candidates:
                return self.generate_type1_rule(objects, prompt_index)
            num_to_remove = min(self.rng.randint(1, min(3, len(candidates))), len(objects) - 1)
            # Select the leftmost N objects (already sorted by x)
            target_object_ids = [obj["id"] for obj in candidates[:num_to_remove]]
        
        elif relation_type == "rightmost":
            # Use sorting: select the N rightmost objects that are clearly on the right
            candidates = [obj for obj in relations["x_sorted_objects"] 
                         if obj["id"] in relations["clear_right_objects"]]
            if not candidates:
                return self.generate_type1_rule(objects, prompt_index)
            num_to_remove = min(self.rng.randint(1, min(3, len(candidates))), len(objects) - 1)
            # Select the rightmost N objects (reverse sorted by x)
            candidates.reverse()
            target_object_ids = [obj["id"] for obj in candidates[:num_to_remove]]
        
        elif relation_type == "topmost":
            # Use sorting: select the N topmost objects that are clearly on the top
            candidates = [obj for obj in relations["y_sorted_objects"] 
                         if obj["id"] in relations["clear_top_objects"]]
            if not candidates:
                return self.generate_type1_rule(objects, prompt_index)
            num_to_remove = min(self.rng.randint(1, min(3, len(candidates))), len(objects) - 1)
            # Select the topmost N objects (already sorted by y)
            target_object_ids = [obj["id"] for obj in candidates[:num_to_remove]]
        
        elif relation_type == "bottommost":
            # Use sorting: select the N bottommost objects that are clearly on the bottom
            candidates = [obj for obj in relations["y_sorted_objects"] 
                         if obj["id"] in relations["clear_bottom_objects"]]
            if not candidates:
                return self.generate_type1_rule(objects, prompt_index)
            num_to_remove = min(self.rng.randint(1, min(3, len(candidates))), len(objects) - 1)
            # Select the bottommost N objects (reverse sorted by y)
            candidates.reverse()
            target_object_ids = [obj["id"] for obj in candidates[:num_to_remove]]
        
        # Generate prompt based on relation type
        prompt = self._generate_type3_prompt(relation_type, num_to_remove)
        
        # Create rule
        rule = {
            "level": "type3",
            "relation_type": relation_type,
            "target_object_ids": target_object_ids,
            "num_objects": num_to_remove
        }
        
        return rule, prompt
    
    def _calculate_conceptual_attributes(self, objects: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate conceptual/semantic attributes for Type 4 reasoning.
        
        Args:
            objects: List of objects
            
        Returns:
            Dictionary with conceptual attributes:
            - size_categories: {"large": [ids], "small": [ids]}
            - shape_groups: {shape: [ids]}
            - color_groups: {color: [ids]}
            - outlier_objects: [ids] (objects that differ from majority)
        """
        attributes = {
            "size_categories": {"large": [], "small": []},
            "shape_groups": {},
            "color_groups": {},
            "outlier_objects": [],
            "size_threshold": None
        }
        
        if len(objects) < 2:
            return attributes
        
        # Calculate size categories (large vs small)
        # Use extreme percentiles (10th and 90th) to ensure VERY clear size distinction
        # This makes large vs small visually obvious and reduces ambiguity
        sizes = [obj["size"] for obj in objects]
        areas = [obj["area"] for obj in objects]
        
        sorted_sizes = sorted(sizes)
        num_objects = len(sorted_sizes)
        
        # Use 10th and 90th percentiles for extreme size distinction
        # Small: bottom 10%, Large: top 10%
        # This ensures maximum visual difference between large and small objects
        p10_index = max(0, num_objects // 10 - 1)  # 10th percentile
        p90_index = min(num_objects - 1, (9 * num_objects) // 10)  # 90th percentile
        
        if p10_index < len(sorted_sizes) and p90_index < len(sorted_sizes) and p10_index < p90_index:
            size_threshold_small = sorted_sizes[p10_index]
            size_threshold_large = sorted_sizes[p90_index]
            
            # Additional check: ensure minimum size gap between large and small
            # If gap is too small (< 15 pixels), use even more extreme thresholds
            min_gap = 15
            if size_threshold_large - size_threshold_small < min_gap:
                # Use 5th and 95th percentiles for even more extreme distinction
                p5_index = max(0, num_objects // 20 - 1)
                p95_index = min(num_objects - 1, (19 * num_objects) // 20)
                if p5_index < len(sorted_sizes) and p95_index < len(sorted_sizes):
                    size_threshold_small = sorted_sizes[p5_index]
                    size_threshold_large = sorted_sizes[p95_index]
        else:
            # Fallback: use median with very large margin
            size_median = np.median(sizes)
            size_threshold_small = size_median - 15  # Very large margin
            size_threshold_large = size_median + 15
        
        attributes["size_threshold"] = (size_threshold_small, size_threshold_large)
        
        for obj in objects:
            # Size classification using extreme percentiles
            # Small: size <= 10th percentile (or 5th if gap too small)
            # Large: size >= 90th percentile (or 95th if gap too small)
            if obj["size"] <= size_threshold_small:
                attributes["size_categories"]["small"].append(obj["id"])
            elif obj["size"] >= size_threshold_large:
                attributes["size_categories"]["large"].append(obj["id"])
            
            # Shape grouping
            shape = obj["shape"]
            if shape not in attributes["shape_groups"]:
                attributes["shape_groups"][shape] = []
            attributes["shape_groups"][shape].append(obj["id"])
            
            # Color grouping
            color = obj["color"]
            if color not in attributes["color_groups"]:
                attributes["color_groups"][color] = []
            attributes["color_groups"][color].append(obj["id"])
        
        # Find outlier objects (different from majority)
        # Outlier: objects that are clearly different from the majority
        # Strategy: Find the majority group (color+shape combination that appears most)
        # Then mark all objects NOT in this majority group as outliers
        # This ensures outliers are visually distinct and minority
        
        # Count all color+shape combinations
        combo_counts = {}
        for obj in objects:
            combo = (obj["color"], obj["shape"])
            if combo not in combo_counts:
                combo_counts[combo] = []
            combo_counts[combo].append(obj["id"])
        
        # Find the majority combination (must be > 33% to be considered majority)
        # This ensures clear distinction between majority and outliers
        majority_threshold = len(objects) * 0.33
        majority_combo = None
        max_count = 0
        
        for combo, ids in combo_counts.items():
            if len(ids) > max_count:
                max_count = len(ids)
                majority_combo = combo
        
        # For Type 4, we want to find THE ONE outlier (singular)
        # So we only mark outliers if there's a VERY clear majority (>=60%) and exactly ONE outlier
        # This will be checked again in generate_type4_rule, but we prepare the list here
        if majority_combo and max_count >= len(objects) * 0.6:
            # All objects NOT in the majority group are potential outliers
            for obj in objects:
                combo = (obj["color"], obj["shape"])
                if combo != majority_combo:
                    attributes["outlier_objects"].append(obj["id"])
        else:
            # If no clear majority, mark unique combinations as potential outliers
            for obj in objects:
                combo = (obj["color"], obj["shape"])
                same_combo_count = combo_counts[combo]
                # Only mark as outlier if truly unique (appears only once)
                if same_combo_count == 1:
                    attributes["outlier_objects"].append(obj["id"])
        
        # Remove duplicates from outlier_objects
        attributes["outlier_objects"] = list(dict.fromkeys(attributes["outlier_objects"]))
        
        return attributes
    
    def generate_type4_rule(self, objects: List[Dict[str, Any]], 
                         prompt_index: int = 0,
                         relation_type_hint: Optional[str] = None) -> Tuple[Dict[str, Any], str]:
        """
        Generate Type 4 rule: Remove objects based on conceptual/semantic properties.
        
        Args:
            objects: List of objects
            prompt_index: Index into PROMPTS_TYPE4 list
            relation_type_hint: Optional hint for relation type ("remove_outlier", "remove_shape_outlier", "remove_color_outlier") for balanced distribution
            
        Returns:
            (rule_dict, prompt_text)
        """
        # Ensure we have enough objects (at least 3 for Type 4)
        if len(objects) < 3:
            return self.generate_type1_rule(objects, prompt_index)
        
        # Calculate conceptual attributes
        attributes = self._calculate_conceptual_attributes(objects)
        
        # Define available conceptual relation types
        # Type 4 supports three types of outlier detection:
        # 1. remove_outlier: Combination outlier (color+shape combination)
        # 2. remove_shape_outlier: Shape consistency outlier (same shape, different colors)
        # 3. remove_color_outlier: Color consistency outlier (same color, different shapes)
        relation_types = []
        
        # Type 4.1: Combination outlier detection
        # Check for clear majority group (color+shape combination)
        combo_counts = {}
        for obj in objects:
            combo = (obj["color"], obj["shape"])
            combo_counts[combo] = combo_counts.get(combo, 0) + 1
        
        majority_combo = max(combo_counts.items(), key=lambda x: x[1], default=None)
        
        # Only add outlier if there's a clear majority (>=50%) and exactly ONE outlier
        if majority_combo and majority_combo[1] >= len(objects) * 0.5:
            # Count how many objects are NOT in the majority
            outlier_count = len(objects) - majority_combo[1]
            if outlier_count == 1:  # Exactly ONE outlier
                relation_types.append("remove_outlier")  # Remove THE one that looks different
        
        # Type 4.2.1: Shape consistency outlier detection
        # Check if majority have same shape but different colors
        shape_counts = {}
        for obj in objects:
            shape = obj["shape"]
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
        
        majority_shape = max(shape_counts.items(), key=lambda x: x[1], default=None)
        if majority_shape and majority_shape[1] >= len(objects) * 0.5:
            # Count how many objects have different shapes
            different_shape_count = len(objects) - majority_shape[1]
            if different_shape_count == 1:  # Exactly ONE object with different shape
                # Verify that majority objects have different colors (not all same color)
                majority_shape_objs = [obj for obj in objects if obj["shape"] == majority_shape[0]]
                majority_colors = set(obj["color"] for obj in majority_shape_objs)
                # Only add if majority has at least 2 different colors (ensuring color diversity)
                if len(majority_colors) >= 2:
                    relation_types.append("remove_shape_outlier")
        
        # Type 4.2.2: Color consistency outlier detection
        # Check if majority have same color but different shapes
        color_counts = {}
        for obj in objects:
            color = obj["color"]
            color_counts[color] = color_counts.get(color, 0) + 1
        
        majority_color = max(color_counts.items(), key=lambda x: x[1], default=None)
        if majority_color and majority_color[1] >= len(objects) * 0.5:
            # Count how many objects have different colors
            different_color_count = len(objects) - majority_color[1]
            if different_color_count == 1:  # Exactly ONE object with different color
                # Verify that majority objects have different shapes (not all same shape)
                majority_color_objs = [obj for obj in objects if obj["color"] == majority_color[0]]
                majority_shapes = set(obj["shape"] for obj in majority_color_objs)
                # Only add if majority has at least 2 different shapes (ensuring shape diversity)
                if len(majority_shapes) >= 2:
                    relation_types.append("remove_color_outlier")
        
        # Type 4 supports three types of outlier detection:
        # 1. remove_outlier: Combination outlier (color+shape combination)
        # 2. remove_shape_outlier: Shape consistency outlier
        # 3. remove_color_outlier: Color consistency outlier
        
        if not relation_types:
            # Fallback to type1 if no suitable type4 rule can be generated
            return self.generate_type1_rule(objects, prompt_index)
        
        # Randomly select from available relation types
        # Use hint if provided and available, otherwise random choice
        if relation_type_hint in relation_types:
            relation_type = relation_type_hint
        else:
            relation_type = self.rng.choice(relation_types)
        
        # Select target objects based on relation type
        target_object_ids = []
        num_to_remove = 0
        
        if relation_type == "remove_outlier":
            # Remove THE one object that looks different (singular, exactly one)
            # This should be exactly one due to our check above
            target_object_ids = attributes["outlier_objects"]
            # Ensure it's exactly one (should be guaranteed by our check)
            if len(target_object_ids) != 1:
                # Fallback: if somehow multiple, pick the first one
                target_object_ids = [target_object_ids[0]] if target_object_ids else []
            num_to_remove = len(target_object_ids)
        
        elif relation_type == "remove_shape_outlier":
            # Remove THE one object with different shape
            # Find the majority shape
            shape_counts = {}
            for obj in objects:
                shape = obj["shape"]
                shape_counts[shape] = shape_counts.get(shape, 0) + 1
            majority_shape = max(shape_counts.items(), key=lambda x: x[1], default=None)
            
            if majority_shape:
                # Find objects with different shape
                different_shape_objs = [obj for obj in objects if obj["shape"] != majority_shape[0]]
                if len(different_shape_objs) == 1:
                    target_object_ids = [different_shape_objs[0]["id"]]
                else:
                    # Fallback: pick first one
                    target_object_ids = [different_shape_objs[0]["id"]] if different_shape_objs else []
            num_to_remove = len(target_object_ids)
        
        elif relation_type == "remove_color_outlier":
            # Remove THE one object with different color
            # Find the majority color
            color_counts = {}
            for obj in objects:
                color = obj["color"]
                color_counts[color] = color_counts.get(color, 0) + 1
            majority_color = max(color_counts.items(), key=lambda x: x[1], default=None)
            
            if majority_color:
                # Find objects with different color
                different_color_objs = [obj for obj in objects if obj["color"] != majority_color[0]]
                if len(different_color_objs) == 1:
                    target_object_ids = [different_color_objs[0]["id"]]
                else:
                    # Fallback: pick first one
                    target_object_ids = [different_color_objs[0]["id"]] if different_color_objs else []
            num_to_remove = len(target_object_ids)
        
        
        # Generate prompt based on relation type
        prompt = self._generate_type4_prompt(relation_type, num_to_remove, attributes, objects)
        
        # Create rule
        rule = {
            "level": "type4",
            "relation_type": relation_type,
            "target_object_ids": target_object_ids,
            "num_objects": num_to_remove,
            "attributes": {
                "size_categories": attributes["size_categories"],
                "outlier_objects": attributes["outlier_objects"]
            }
        }
        
        return rule, prompt
    
    def _generate_type4_prompt(self, relation_type: str, num_objects: int,
                           attributes: Dict[str, Any], objects: List[Dict[str, Any]]) -> str:
        """
        Generate prompt text for Type 4 based on conceptual relation type.
        Type 4 now only supports remove_outlier (largest/smallest moved to type1).
        
        Args:
            relation_type: Type of conceptual relation (should be "remove_outlier")
            num_objects: Number of objects to remove
            attributes: Conceptual attributes dictionary
            objects: List of all objects
            
        Returns:
            Prompt text
        """
        # All type4 types use the same unified prompt
        # This makes the task more abstract - the model needs to figure out what makes the object different
        # (whether it's shape, color, or combination)
        prompt = "Remove the object that looks different from the others. Do not do anything to other objects."
        
        return prompt
    
    def _generate_type3_prompt(self, relation_type: str, num_objects: int) -> str:
        """
        Generate prompt text for Type 3 based on relation type.
        
        Args:
            relation_type: Type of spatial relation
            num_objects: Number of objects to remove
            
        Returns:
            Prompt text
        """
        if relation_type == "leftmost":
            if num_objects == 1:
                prompt = "Remove the leftmost object. Do not do anything to other objects."
            else:
                prompt = f"Remove the {num_objects} leftmost objects. Do not do anything to other objects."
        
        elif relation_type == "rightmost":
            if num_objects == 1:
                prompt = "Remove the rightmost object. Do not do anything to other objects."
            else:
                prompt = f"Remove the {num_objects} rightmost objects. Do not do anything to other objects."
        
        elif relation_type == "topmost":
            if num_objects == 1:
                prompt = "Remove the topmost object. Do not do anything to other objects."
            else:
                prompt = f"Remove the {num_objects} topmost objects. Do not do anything to other objects."
        
        elif relation_type == "bottommost":
            if num_objects == 1:
                prompt = "Remove the bottommost object. Do not do anything to other objects."
            else:
                prompt = f"Remove the {num_objects} bottommost objects. Do not do anything to other objects."
        
        else:
            # Fallback prompt
            prompt = f"Remove {num_objects} objects based on spatial relations. Do not do anything to other objects."
        
        return prompt
    
    def generate_type2_rule(self, objects: List[Dict[str, Any]], 
                        prompt_index: int = 0,
                        num_to_remove_hint: Optional[int] = None) -> Tuple[Dict[str, Any], str]:
        """
        Generate Type 2 rule: Remove multiple explicitly listed objects by color and shape.
        
        Args:
            objects: List of objects
            prompt_index: Index into PROMPTS_TYPE2 list
            num_to_remove_hint: Optional hint for number of objects to remove (1, 2, or 3) for balanced distribution
            
        Returns:
            (rule_dict, prompt_text)
        """
        # Ensure we have enough objects (at least 2 for Type 2)
        if len(objects) < 2:
            # Fallback to type1 if not enough objects
            return self.generate_type1_rule(objects, prompt_index)
        
        # Randomly select 1-3 objects to remove (ensure at least 1 object remains)
        # Use hint if provided and valid, otherwise random choice
        if num_to_remove_hint in [1, 2, 3] and num_to_remove_hint <= len(objects) - 1:
            num_to_remove = num_to_remove_hint
        else:
            num_to_remove = self.rng.randint(1, min(3, len(objects) - 1))
        selected_objects = self.rng.sample(objects, num_to_remove)
        
        # Build targets list with color and shape combinations
        targets = [
            {"color": obj["color"], "shape": obj["shape"]}
            for obj in selected_objects
        ]
        
        # Get target object IDs - include ALL objects matching each target
        # This ensures "Remove the yellow sphere" removes ALL yellow spheres, not just one
        target_object_ids = []
        for target in targets:
            matching_objs = [obj for obj in objects 
                           if obj["color"] == target["color"] and obj["shape"] == target["shape"]]
            target_object_ids.extend([obj["id"] for obj in matching_objs])
        
        # Remove duplicates while preserving order
        target_object_ids = list(dict.fromkeys(target_object_ids))
        
        # Generate prompt dynamically
        # Format: "Remove the red cube, all green spheres, and the blue pyramid..."
        # Use "all" + plural if multiple objects match the target
        target_descriptions = []
        for target in targets:
            matching_count = len([obj for obj in objects 
                                if obj["color"] == target["color"] and obj["shape"] == target["shape"]])
            
            if matching_count > 1:
                # Multiple objects: use "all" + plural form
                shape_plural = target['shape'] + 's' if not target['shape'].endswith('s') else target['shape']
                target_descriptions.append(f"all {target['color']} {shape_plural}")
            else:
                # Single object: use "the" + singular form
                target_descriptions.append(f"the {target['color']} {target['shape']}")
        
        if len(target_descriptions) == 1:
            prompt = f"Remove {target_descriptions[0]} from the scene. Do not do anything to other objects."
        elif len(target_descriptions) == 2:
            prompt = f"Remove {target_descriptions[0]} and {target_descriptions[1]} from the scene. Do not do anything to other objects."
        else:  # 3 objects
            prompt = f"Remove {target_descriptions[0]}, {target_descriptions[1]}, and {target_descriptions[2]} from the scene. Do not do anything to other objects."
        
        rule = {
            "level": "type2",
            "targets": targets,
            "target_object_ids": target_object_ids,
            "num_to_remove": num_to_remove
        }
        
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
        
        # Task type counters for balanced distribution
        # Type 1: 2 types (color, shape) - each 50%
        self.type1_counters = {"color": 0, "shape": 0}
        # Type 2: 3 types (remove 1, 2, 3 objects) - each 33.33%
        self.type2_counters = {1: 0, 2: 0, 3: 0}
        # Type 3: 4 types (leftmost, rightmost, topmost, bottommost) - each 25%
        self.type3_counters = {"leftmost": 0, "rightmost": 0, "topmost": 0, "bottommost": 0}
        # Type 4: 3 types (remove_outlier, remove_shape_outlier, remove_color_outlier) - each 33.33%
        self.type4_counters = {"remove_outlier": 0, "remove_shape_outlier": 0, "remove_color_outlier": 0}
    
    def _select_task_type_by_balance(self, level: str) -> Optional[str]:
        """
        Select task type based on balanced distribution.
        Returns the task type that needs to be generated to maintain balance.
        
        Args:
            level: "type1", "type2", "type3", or "type4"
            
        Returns:
            Task type identifier, or None if random selection should be used
        """
        if level == "type1":
            # Select the type with minimum count
            min_count = min(self.type1_counters.values())
            candidates = [t for t, count in self.type1_counters.items() if count == min_count]
            return self.rule_gen.rng.choice(candidates) if candidates else None
        
        elif level == "type2":
            # Select the num_to_remove with minimum count
            min_count = min(self.type2_counters.values())
            candidates = [t for t, count in self.type2_counters.items() if count == min_count]
            return self.rule_gen.rng.choice(candidates) if candidates else None
        
        elif level == "type3":
            # Select the relation type with minimum count
            min_count = min(self.type3_counters.values())
            candidates = [t for t, count in self.type3_counters.items() if count == min_count]
            return self.rule_gen.rng.choice(candidates) if candidates else None
        
        elif level == "type4":
            # Select the outlier type with minimum count
            min_count = min(self.type4_counters.values())
            candidates = [t for t, count in self.type4_counters.items() if count == min_count]
            return self.rule_gen.rng.choice(candidates) if candidates else None
        
        return None
    
    def _update_task_type_counter(self, level: str, task_type: str):
        """
        Update counter for a specific task type.
        
        Args:
            level: "type1", "type2", "type3", or "type4"
            task_type: Task type identifier
        """
        if level == "type1" and task_type in self.type1_counters:
            self.type1_counters[task_type] += 1
        elif level == "type2" and isinstance(task_type, int) and task_type in self.type2_counters:
            self.type2_counters[task_type] += 1
        elif level == "type3" and task_type in self.type3_counters:
            self.type3_counters[task_type] += 1
        elif level == "type4" and task_type in self.type4_counters:
            self.type4_counters[task_type] += 1
    
    def _get_difficulty(self, level: str) -> str:
        """Get difficulty level based on cognitive level."""
        difficulty_map = {
            "type1": "easy",
            "type2": "medium",
            "type3": "hard",
            "type4": "hard"
        }
        return difficulty_map.get(level, "medium")
    
    def generate_single_task(self, task_id: str, level: str = "type1", 
                            seed: Optional[int] = None) -> ObjectSubtractionTaskPair:
        """
        Generate a single object subtraction task.
        
        Args:
            task_id: Unique task identifier
            level: Task type ("type1", "type2", "type3", "type4")
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
        
        # Select task type based on balanced distribution (before object generation for type4)
        task_type_hint = self._select_task_type_by_balance(level)
        
        # For type4, map task type hint to type4_type for object generation
        l4_type_hint = None
        if level == "type4" and task_type_hint:
            type_map = {
                "remove_outlier": 1,
                "remove_shape_outlier": 2,
                "remove_color_outlier": 3
            }
            l4_type_hint = type_map.get(task_type_hint)
        
        # Generate objects
        # For type1, type2, type3, type4: use uniform size (no size differences needed)
        original_min = None
        original_max = None
        if level in ["type1", "type2", "type3", "type4"]:
            # Store original size range
            original_min = self.object_gen.min_size
            original_max = self.object_gen.max_size
            # Use a fixed size for all objects in all levels
            # Choose a middle size (e.g., 30 pixels) for uniform appearance
            uniform_size = 30
            self.object_gen.min_size = uniform_size
            self.object_gen.max_size = uniform_size
        
        # For type4, generate more objects to increase chance of having clear majorities
        # This makes it easier to satisfy type4 conditions (>=50% majority, etc.)
        if level == "type4":
            # Generate 6-8 objects (instead of 5-8) to increase chance of clear majorities
            num_objects = random.randint(6, 8)
        else:
            num_objects = random.randint(self.num_objects_range[0], self.num_objects_range[1])
        
        objects = self.object_gen.generate_objects(num_objects, seed=seed)
        
        # For type1, type2, type3, type4: ensure all objects have exactly the same size
        if level in ["type1", "type2", "type3", "type4"]:
            # Get the first object's size (all should be same after generation with min=max)
            if objects:
                uniform_size = objects[0]["size"]
                # Ensure all objects have exactly this size
                for obj in objects:
                    obj["size"] = uniform_size
                    # Update area accordingly
                    if obj["shape"] == "cube":
                        obj["area"] = uniform_size * uniform_size
                    elif obj["shape"] == "sphere":
                        obj["area"] = int(np.pi * (uniform_size // 2) ** 2)
                    elif obj["shape"] == "pyramid":
                        obj["area"] = int(uniform_size * uniform_size * 0.433)
                    else:  # cone
                        obj["area"] = int(np.pi * (uniform_size // 2) ** 2)
        
        # For type4, actively create scenarios for different outlier detection types
        # This ensures we can generate proper type4 tasks
        if level == "type4":
            num_objects = len(objects)
            majority_count = num_objects - 1  # All but one
            
            # Choose which type of type4 task to create based on hint or random
            # 1: remove_outlier (combination), 2: remove_shape_outlier, 3: remove_color_outlier
            if l4_type_hint in [1, 2, 3]:
                l4_type = l4_type_hint
            else:
                l4_type = self.rule_gen.rng.choice([1, 2, 3])
            
            if l4_type == 1:
                # Type 4.1: Combination outlier (color+shape combination)
                # Create a clear majority: most objects with same color+shape, one different
                majority_color = self.rule_gen.rng.choice(ObjectGenerator.COLORS)
                majority_shape = self.rule_gen.rng.choice(ObjectGenerator.SHAPES)
                
                # Choose an outlier color and shape (different from majority)
                outlier_color = self.rule_gen.rng.choice([c for c in ObjectGenerator.COLORS if c != majority_color])
                outlier_shape = self.rule_gen.rng.choice([s for s in ObjectGenerator.SHAPES if s != majority_shape])
                
                # Assign majority color+shape to most objects
                for i, obj in enumerate(objects):
                    if i < majority_count:
                        obj["color"] = majority_color
                        obj["shape"] = majority_shape
                    else:
                        # The last one is the outlier
                        obj["color"] = outlier_color
                        obj["shape"] = outlier_shape
            
            elif l4_type == 2:
                # Type 4.2.1: Shape consistency outlier
                # Majority: same shape but different colors, outlier: different shape
                majority_shape = self.rule_gen.rng.choice(ObjectGenerator.SHAPES)
                outlier_shape = self.rule_gen.rng.choice([s for s in ObjectGenerator.SHAPES if s != majority_shape])
                
                # Assign majority shape to most objects with different colors
                available_colors = ObjectGenerator.COLORS.copy()
                for i, obj in enumerate(objects):
                    if i < majority_count:
                        # Assign majority shape with different colors
                        color = self.rule_gen.rng.choice(available_colors)
                        obj["color"] = color
                        obj["shape"] = majority_shape
                        # Ensure color diversity: remove used color if we have enough colors
                        if len(available_colors) > 1:
                            available_colors.remove(color)
                            if not available_colors:
                                available_colors = ObjectGenerator.COLORS.copy()
                    else:
                        # The last one is the outlier (different shape)
                        obj["shape"] = outlier_shape
                        # Outlier can have any color
                        obj["color"] = self.rule_gen.rng.choice(ObjectGenerator.COLORS)
            
            elif l4_type == 3:
                # Type 4.2.2: Color consistency outlier
                # Majority: same color but different shapes, outlier: different color
                majority_color = self.rule_gen.rng.choice(ObjectGenerator.COLORS)
                outlier_color = self.rule_gen.rng.choice([c for c in ObjectGenerator.COLORS if c != majority_color])
                
                # Assign majority color to most objects with different shapes
                available_shapes = ObjectGenerator.SHAPES.copy()
                for i, obj in enumerate(objects):
                    if i < majority_count:
                        # Assign majority color with different shapes
                        shape = self.rule_gen.rng.choice(available_shapes)
                        obj["color"] = majority_color
                        obj["shape"] = shape
                        # Ensure shape diversity: remove used shape if we have enough shapes
                        if len(available_shapes) > 1:
                            available_shapes.remove(shape)
                            if not available_shapes:
                                available_shapes = ObjectGenerator.SHAPES.copy()
                    else:
                        # The last one is the outlier (different color)
                        obj["color"] = outlier_color
                        # Outlier can have any shape
                        obj["shape"] = self.rule_gen.rng.choice(ObjectGenerator.SHAPES)
        
        # Restore original size range after generation (for type1, type2, type3, type4)
        if level in ["type1", "type2", "type3", "type4"] and original_min is not None and original_max is not None:
            self.object_gen.min_size = original_min
            self.object_gen.max_size = original_max
        
        # Generate rule and prompt based on level
        # For type4, retry if fallback to type1 to ensure we get proper type4 tasks
        # Use more retries to ensure we get proper type4 tasks
        max_retries = 10 if level == "type4" else 1
        
        for attempt in range(max_retries):
            if level == "type1":
                rule, prompt = self.rule_gen.generate_type1_rule(objects, 
                                                              prompt_index=DEFAULT_PROMPT_INDEX_TYPE1,
                                                              task_type_hint=task_type_hint)
            elif level == "type2":
                rule, prompt = self.rule_gen.generate_type2_rule(objects, 
                                                              prompt_index=DEFAULT_PROMPT_INDEX_TYPE2,
                                                              num_to_remove_hint=task_type_hint)
            elif level == "type3":
                rule, prompt = self.rule_gen.generate_type3_rule(objects, 
                                                              canvas_size=self.canvas_size,
                                                              prompt_index=DEFAULT_PROMPT_INDEX_TYPE3,
                                                              relation_type_hint=task_type_hint)
            elif level == "type4":
                rule, prompt = self.rule_gen.generate_type4_rule(objects, 
                                                              prompt_index=DEFAULT_PROMPT_INDEX_TYPE4,
                                                              relation_type_hint=task_type_hint)
                # Check if we got a proper type4 rule (not fallback to type1)
                if rule.get("level") == "type4":
                    break  # Success! Got a proper type4 rule
                else:
                    # Fallback to type1 - retry with new objects
                    if attempt < max_retries - 1:
                        # Regenerate objects with different seed
                        retry_seed = seed + 1000 + attempt if seed is not None else None
                        if retry_seed is not None:
                            random.seed(retry_seed)
                            np.random.seed(retry_seed)
                            self.rule_gen.rng.seed(retry_seed)
                            self.object_gen.rng.seed(retry_seed)
                        
                        # Regenerate objects
                        # For type1, type2, type3, type4: use uniform size
                        retry_original_min = None
                        retry_original_max = None
                        if level in ["type1", "type2", "type3", "type4"]:
                            retry_original_min = self.object_gen.min_size
                            retry_original_max = self.object_gen.max_size
                            uniform_size = 30
                            self.object_gen.min_size = uniform_size
                            self.object_gen.max_size = uniform_size
                        
                        # For type4 retry, use 6-8 objects to increase chance of clear majorities
                        num_objects = random.randint(6, 8) if level == "type4" else random.randint(self.num_objects_range[0], self.num_objects_range[1])
                        objects = self.object_gen.generate_objects(num_objects, seed=retry_seed)
                        
                        # For type1, type2, type3, type4: ensure all objects have exactly the same size
                        if level in ["type1", "type2", "type3", "type4"]:
                            if objects:
                                uniform_size = objects[0]["size"]
                                for obj in objects:
                                    obj["size"] = uniform_size
                                    if obj["shape"] == "cube":
                                        obj["area"] = uniform_size * uniform_size
                                    elif obj["shape"] == "sphere":
                                        obj["area"] = int(np.pi * (uniform_size // 2) ** 2)
                                    elif obj["shape"] == "pyramid":
                                        obj["area"] = int(uniform_size * uniform_size * 0.433)
                                    else:  # cone
                                        obj["area"] = int(np.pi * (uniform_size // 2) ** 2)
                            
                            # Restore size range
                            if retry_original_min is not None and retry_original_max is not None:
                                self.object_gen.min_size = retry_original_min
                                self.object_gen.max_size = retry_original_max
                        
                        # For type4, actively create scenarios for different outlier detection types
                        if level == "type4" and attempt < max_retries - 1:
                            num_objects = len(objects)
                            majority_count = num_objects - 1  # All but one
                            
                            # Randomly choose which type of type4 task to create
                            l4_type = self.rule_gen.rng.choice([1, 2, 3])
                            
                            if l4_type == 1:
                                # Type 4.1: Combination outlier
                                majority_color = self.rule_gen.rng.choice(ObjectGenerator.COLORS)
                                majority_shape = self.rule_gen.rng.choice(ObjectGenerator.SHAPES)
                                outlier_color = self.rule_gen.rng.choice([c for c in ObjectGenerator.COLORS if c != majority_color])
                                outlier_shape = self.rule_gen.rng.choice([s for s in ObjectGenerator.SHAPES if s != majority_shape])
                                
                                for i, obj in enumerate(objects):
                                    if i < majority_count:
                                        obj["color"] = majority_color
                                        obj["shape"] = majority_shape
                                    else:
                                        obj["color"] = outlier_color
                                        obj["shape"] = outlier_shape
                            
                            elif l4_type == 2:
                                # Type 4.2.1: Shape consistency outlier
                                majority_shape = self.rule_gen.rng.choice(ObjectGenerator.SHAPES)
                                outlier_shape = self.rule_gen.rng.choice([s for s in ObjectGenerator.SHAPES if s != majority_shape])
                                available_colors = ObjectGenerator.COLORS.copy()
                                
                                for i, obj in enumerate(objects):
                                    if i < majority_count:
                                        color = self.rule_gen.rng.choice(available_colors)
                                        obj["color"] = color
                                        obj["shape"] = majority_shape
                                        if len(available_colors) > 1:
                                            available_colors.remove(color)
                                            if not available_colors:
                                                available_colors = ObjectGenerator.COLORS.copy()
                                    else:
                                        obj["shape"] = outlier_shape
                                        obj["color"] = self.rule_gen.rng.choice(ObjectGenerator.COLORS)
                            
                            elif l4_type == 3:
                                # Type 4.2.2: Color consistency outlier
                                majority_color = self.rule_gen.rng.choice(ObjectGenerator.COLORS)
                                outlier_color = self.rule_gen.rng.choice([c for c in ObjectGenerator.COLORS if c != majority_color])
                                available_shapes = ObjectGenerator.SHAPES.copy()
                                
                                for i, obj in enumerate(objects):
                                    if i < majority_count:
                                        shape = self.rule_gen.rng.choice(available_shapes)
                                        obj["color"] = majority_color
                                        obj["shape"] = shape
                                        if len(available_shapes) > 1:
                                            available_shapes.remove(shape)
                                            if not available_shapes:
                                                available_shapes = ObjectGenerator.SHAPES.copy()
                                    else:
                                        obj["color"] = outlier_color
                                        obj["shape"] = self.rule_gen.rng.choice(ObjectGenerator.SHAPES)
                            
                            # Size range already restored above for type1/type2/type3/type4
                            if level not in ["type1", "type2", "type3", "type4"]:
                                if retry_original_min is not None and retry_original_max is not None:
                                    self.object_gen.min_size = retry_original_min
                                    self.object_gen.max_size = retry_original_max
                        continue  # Retry with new objects
            else:
                # For type1, type2, type3, we should have generated rule and prompt successfully
                # Break out of the loop to use the generated rule
                break
        
        # For type4, if we still got type1 rule after retries, that's a problem
        # But we'll accept it as a last resort (should be rare with retries)
        if level == "type4" and rule.get("level") != "type4":
            # This should be very rare after retries, but log it
            pass
        
        # Update task type counter based on generated rule
        if level == "type1":
            rule_type = rule.get("rule_type")
            if rule_type in ["color", "shape"]:
                self._update_task_type_counter(level, rule_type)
        elif level == "type2":
            # Get num_to_remove from rule (stored during generation)
            num_to_remove = rule.get("num_to_remove")
            if num_to_remove is None:
                # Fallback: count unique target combinations
                targets = rule.get("targets", [])
                if targets:
                    num_to_remove = len(targets)
            if num_to_remove in [1, 2, 3]:
                self._update_task_type_counter(level, num_to_remove)
        elif level == "type3":
            relation_type = rule.get("relation_type")
            if relation_type in ["leftmost", "rightmost", "topmost", "bottommost"]:
                self._update_task_type_counter(level, relation_type)
        elif level == "type4":
            relation_type = rule.get("relation_type")
            if relation_type in ["remove_outlier", "remove_shape_outlier", "remove_color_outlier"]:
                self._update_task_type_counter(level, relation_type)
        
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
            difficulty=self._get_difficulty(level),
            created_at=datetime.now().isoformat()
        )
        
        return task_pair


def create_dataset(num_samples: int = 50, levels: List[str] = ["type1", "type2", "type3", "type4"]) -> Dict[str, Any]:
    """
    Create object subtraction dataset - main entry point matching other tasks.
    
    Args:
        num_samples: Number of tasks to generate
        levels: List of task types to generate ("type1", "type2", "type3", "type4")
        
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
            # Map level to type number: type1->type1, type2->type2, type3->type3, type4->type4
            type_name = level.lower().replace('l', 'type')
            task_id = f"object_subtraction_{type_name}_{i:04d}"
            # Generate deterministic seed based on task ID hash for reproducibility
            # This ensures deterministic generation while not requiring a random_seed parameter
            task_hash = int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16)
            seed = task_hash + level_idx * 10000 + i
            
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

