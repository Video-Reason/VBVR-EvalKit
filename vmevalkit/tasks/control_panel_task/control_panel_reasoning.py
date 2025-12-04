"""
Control Panel Animation Task for VMEvalKit

This task tests whether video generation models can:
- Understand the relationship between lever position and light color
- Generate smooth animations of lever movements
- Synchronize lever movements with light color changes
- Reason about control panel state transitions

Design:
- First Frame: Control panel with levers in initial positions, lights showing initial colors
- Final Frame: Levers moved to target positions, lights showing target colors
- Animation: Smooth transition showing lever movements and simultaneous color changes
"""

import json
import random
import numpy as np
import tempfile
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch

# Import prompts from centralized location
from .PROMPTS import (
    PROMPTS,
    DEFAULT_PROMPT_INDEX
)


@dataclass
class ControlPanelTaskPair:
    """
    Data structure for control panel animation video model evaluation.
    
    Contains:
    - prompt: Instructions for the video model
    - first_image: Control panel with levers in initial positions
    - final_image: Control panel with levers in target positions
    """
    id: str
    prompt: str                      # What to ask the video model to do
    first_image_path: str           # Initial state
    final_image_path: str           # Target state
    task_category: str = "ControlPanel"
    control_panel_data: Dict[str, Any] = None  # Metadata
    difficulty: str = "easy"        # "easy", "medium", "hard"
    num_lights: int = 1            # Number of lights (1, 2, or 3)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class ControlPanelGenerator:
    """Generate control panel configurations with levers and lights."""
    
    # Color mapping: lever position -> light color
    POSITION_COLORS = {
        "left": "red",
        "middle": "green",
        "right": "blue"
    }
    
    POSITIONS = ["left", "middle", "right"]
    
    def __init__(self, canvas_size: Tuple[int, int] = (256, 256), seed: Optional[int] = None):
        """
        Initialize control panel generator.
        
        Args:
            canvas_size: (width, height) of the canvas
            seed: Random seed for reproducibility
        """
        self.canvas_size = canvas_size
        self.rng = random.Random(seed) if seed is not None else random.Random()
    
    def generate_panel_config(self, num_lights: int = 3, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a control panel configuration.
        
        Requirements:
        - All cases: at least 2 different colors in initial state
        - 2 lights: 2 or 3 colors (at least 2)
        - 3, 4, 6 lights: 2 or 3 colors (at least 2), can have duplicate colors
        
        Args:
            num_lights: Number of lights (2, 3, 4, or 6)
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with panel configuration including:
            - lights: List of light configurations
            - initial_positions: List of initial lever positions
            - target_positions: List of target lever positions
        """
        if seed is not None:
            self.rng.seed(seed)
        
        lights = []
        initial_positions = []
        target_positions = []
        
        # Determine how many colors to use (2 or 3)
        num_colors = self.rng.choice([2, 3])
        
        # Select which colors to use
        available_colors = list(self.POSITION_COLORS.values())  # ['red', 'green', 'blue']
        selected_colors = self.rng.sample(available_colors, num_colors)
        
        # Generate initial positions ensuring at least 2 different colors
        # Strategy: assign colors to lights, ensuring first two are different
        initial_colors = []
        for i in range(num_lights):
            if i < 2:
                # First two lights: use first two selected colors (ensures at least 2 colors)
                color = selected_colors[i]
            else:
                # Remaining lights: randomly choose from selected colors (allows duplicates)
                color = self.rng.choice(selected_colors)
            initial_colors.append(color)
        
        # Convert colors to positions
        for color in initial_colors:
            # Find the position that corresponds to this color
            for pos, pos_color in self.POSITION_COLORS.items():
                if pos_color == color:
                    initial_positions.append(pos)
                    break
        
        # Generate target positions (will be overridden by generate_single_task based on target_color)
        for i in range(num_lights):
            initial_pos = initial_positions[i]
            # Target position: ensure at least one lever needs to move
            if initial_pos == "right":
                target_pos = self.rng.choice(["left", "middle"])
            else:
                other_positions = [p for p in self.POSITIONS if p != initial_pos]
                target_pos = self.rng.choice(other_positions)
            target_positions.append(target_pos)
            
            # Get colors based on positions
            initial_color = self.POSITION_COLORS[initial_pos]
            target_color = self.POSITION_COLORS[target_pos]
            
            lights.append({
                "id": i,
                "initial_position": initial_pos,
                "target_position": target_pos,
                "initial_color": initial_color,
                "target_color": target_color
            })
        
        return {
            "lights": lights,
            "initial_positions": initial_positions,
            "target_positions": target_positions,
            "num_lights": num_lights
        }


class SceneRenderer:
    """Render control panel scenes with lights and levers."""
    
    COLOR_MAP = {
        "red": "#FF0000",
        "green": "#00FF00",
        "blue": "#0000FF",
        "off": "#404040",  # Dark gray for off state
        "black": "#000000"
    }
    
    # Position to color mapping (same as ControlPanelGenerator)
    POSITION_COLORS = {
        "left": "red",
        "middle": "green",
        "right": "blue"
    }
    
    def __init__(self, canvas_size: Tuple[int, int] = (256, 256), dpi: int = 100):
        self.canvas_size = canvas_size
        self.dpi = dpi
    
    def render_panel(self, panel_config: Dict[str, Any], 
                    lever_positions: List[str],
                    output_path: Union[str, Path]) -> None:
        """
        Render control panel with specified lever positions.
        
        Args:
            panel_config: Panel configuration from generator
            lever_positions: List of lever positions for each light
            output_path: Path to save the image
        """
        fig, ax = plt.subplots(1, 1, figsize=(self.canvas_size[0]/self.dpi, 
                                               self.canvas_size[1]/self.dpi), 
                              dpi=self.dpi)
        
        # Draw background
        ax.set_xlim(0, self.canvas_size[0])
        ax.set_ylim(0, self.canvas_size[1])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.invert_yaxis()
        
        num_lights = panel_config["num_lights"]
        lights = panel_config["lights"]
        
        # Calculate layout with support for multi-row arrangement
        if num_lights == 2:
            # Single row layout
            num_rows = 1
            lights_per_row = 2
        elif num_lights == 3:
            # Single row layout
            num_rows = 1
            lights_per_row = 3
        elif num_lights == 4:
            # 2 rows x 2 columns
            num_rows = 2
            lights_per_row = 2
        elif num_lights == 6:
            # 2 rows x 3 columns
            num_rows = 2
            lights_per_row = 3
        elif num_lights == 8:
            # 2 rows x 4 columns
            num_rows = 2
            lights_per_row = 4
        elif num_lights == 9:
            # 3 rows x 3 columns
            num_rows = 3
            lights_per_row = 3
        else:
            # Default: try to fit in 2 rows
            num_rows = 2
            lights_per_row = (num_lights + 1) // 2
        
        panel_width = self.canvas_size[0] * 0.9
        panel_height = self.canvas_size[1] * 0.8
        panel_x = (self.canvas_size[0] - panel_width) / 2
        panel_y = (self.canvas_size[1] - panel_height) / 2
        
        # Calculate unit dimensions based on layout
        unit_width = panel_width / lights_per_row
        
        # For 4 or 6 lights (2 rows), increase row spacing between groups
        if (num_lights == 4 or num_lights == 6) and num_rows == 2:
            # Increase vertical spacing between rows
            row_spacing = panel_height * 0.25  # Extra space between rows (increased from 0.15)
            available_height = panel_height - row_spacing
            unit_height = available_height / num_rows
            unit_spacing = min(unit_width * 0.1, unit_height * 0.05)  # Less spacing within unit
        elif num_lights == 3:
            # For 3 lights, increase spacing between columns
            unit_height = panel_height / num_rows
            unit_spacing = min(unit_width * 0.4, unit_height * 0.1)  # Further increased spacing between columns (from 0.3 to 0.4)
        else:
            unit_height = panel_height / num_rows
            unit_spacing = min(unit_width * 0.1, unit_height * 0.1)
        
        # Draw each control unit
        for i, light in enumerate(lights):
            # Calculate row and column position
            row = i // lights_per_row
            col = i % lights_per_row
            
            # Calculate unit position based on row/column
            unit_x = panel_x + col * unit_width + unit_spacing / 2
            unit_width_actual = unit_width - unit_spacing
            
            # For 4 or 6 lights (2 rows), add extra spacing between rows
            if (num_lights == 4 or num_lights == 6) and num_rows == 2:
                row_spacing = panel_height * 0.25  # Increased spacing between rows (from 0.15)
                unit_y = panel_y + row * (unit_height + row_spacing)
                unit_height_actual = unit_height - unit_spacing
            else:
                unit_y = panel_y + row * unit_height
                unit_height_actual = unit_height - unit_spacing
            
            # Light position (top) - lights are at the top
            # Since invert_yaxis is used, smaller y values are at the top
            # Adjust position relative to unit, not global panel
            # For 4 or 6 lights (2 rows), reduce distance between light and button
            if num_lights == 4 or num_lights == 6:
                light_y = unit_y + unit_height_actual * 0.25  # Light at top, closer to button
            else:
                light_y = unit_y + unit_height_actual * 0.2  # Light at top (in inverted y-axis, small values are at top)
            light_size = min(unit_width_actual * 0.4, unit_height_actual * 0.15)
            light_x = unit_x + unit_width_actual / 2
            
            # Get current color based on lever position
            lever_pos = lever_positions[i]
            current_color = SceneRenderer.POSITION_COLORS[lever_pos]
            
            # Draw indicator light (circle) - using same style as button
            light_circle = Circle((light_x, light_y), light_size / 2,
                                facecolor=self.COLOR_MAP[current_color],
                                edgecolor="black", linewidth=2)
            ax.add_patch(light_circle)
            
            # Draw light border/ring
            light_ring = Circle((light_x, light_y), light_size / 2 + 2,
                              facecolor="none", edgecolor="black", linewidth=1)
            ax.add_patch(light_ring)
            
            # Control slot position (below light) - button is at the bottom
            # Since invert_yaxis is used, larger y values are at the bottom
            # Adjust position relative to unit, not global panel
            # For 4 or 6 lights (2 rows), reduce distance between light and button
            if num_lights == 4 or num_lights == 6:
                slot_y = unit_y + unit_height_actual * 0.65  # Button at bottom, closer to light
            else:
                slot_y = unit_y + unit_height_actual * 0.7  # Button at bottom (in inverted y-axis, large values are at bottom)
            
            # Control slot and button dimensions: maintain 2-lights proportions, but scale down when more lights
            # Calculation for 2 lights:
            #   unit_width = panel_width / 2 (two lights, each takes half)
            #   unit_width_actual = unit_width - unit_spacing = (panel_width / 2) * 0.9
            #   slot_width = unit_width_actual * 0.7 = (panel_width / 2) * 0.9 * 0.7
            #   slot_height = unit_height_actual * 0.2 = panel_height * 0.8 * 0.2
            
            # Calculate reference dimensions and proportions for 2 lights
            reference_unit_width = (panel_width * 0.9) / 2  # Width of each unit for 2 lights
            reference_unit_spacing = reference_unit_width * 0.1
            reference_unit_width_actual = reference_unit_width - reference_unit_spacing
            reference_slot_width = reference_unit_width_actual * 0.7  # Slot width for 2 lights
            reference_slot_height = (panel_height * 0.8) * 0.2  # Slot height for 2 lights
            
            # Determine scale factor based on number of lights
            # 2 lights: 1.0 (baseline)
            # 3 lights: 0.9 (slightly smaller)
            # 4 lights: 0.85 (smaller)
            # 6 lights: 0.75 (smallest)
            if num_lights == 2:
                scale_factor = 1.0
            elif num_lights == 3:
                scale_factor = 0.9
            elif num_lights == 4:
                scale_factor = 0.85
            elif num_lights == 6:
                scale_factor = 0.75
            else:
                scale_factor = 1.0
            
            # Use scaled dimensions, but maintain the same proportions as 2 lights
            slot_width = reference_slot_width * scale_factor
            slot_height = reference_slot_height * scale_factor
            slot_x = unit_x + (unit_width_actual - slot_width) / 2
            
            # Draw black control slot
            slot = Rectangle((slot_x, slot_y), slot_width, slot_height,
                           facecolor=self.COLOR_MAP["black"],
                           edgecolor="gray", linewidth=2)
            ax.add_patch(slot)
            
            # Draw position markers (left, middle, right)
            # Use 3-lights small dot style (unified for all light counts)
            marker_size = 2  # Small dots: smaller size
            left_marker_x = slot_x + slot_width * 0.15
            middle_marker_x = slot_x + slot_width * 0.5
            right_marker_x = slot_x + slot_width * 0.85
            marker_y = slot_y + slot_height / 2
            
            # Draw markers as small circles - small dot markers
            # Only show dots at non-button positions
            marker_positions = []
            if lever_pos != "left":
                marker_positions.append(left_marker_x)
            if lever_pos != "middle":
                marker_positions.append(middle_marker_x)
            if lever_pos != "right":
                marker_positions.append(right_marker_x)
            
            for mx in marker_positions:
                marker = Circle((mx, marker_y), marker_size,
                              facecolor="white", edgecolor="none", alpha=0.6)
                ax.add_patch(marker)
            
            # Draw lever based on position
            # Use 2-lights button dimensions and proportions (unified for all light counts)
            # Button proportions for 2 lights: width is 25% of slot, height is 50% of slot
            lever_width = slot_width * 0.25  # Same proportion as 2 lights
            lever_height = slot_height * 0.5  # Same proportion as 2 lights
            lever_y = slot_y + (slot_height - lever_height) / 2
            
            if lever_pos == "left":
                lever_x = slot_x + slot_width * 0.05  # Left position
            elif lever_pos == "middle":
                lever_x = slot_x + slot_width * 0.5 - lever_width / 2  # Middle position
            else:  # right
                lever_x = slot_x + slot_width * 0.95 - lever_width  # Right position
            
            # Draw lever (rectangle) - long gray block
            lever = Rectangle((lever_x, lever_y), lever_width, lever_height,
                            facecolor="#808080",  # Pure gray
                            edgecolor="black", linewidth=1)  # Thin border
            ax.add_patch(lever)
        
        # Save figure
        plt.tight_layout(pad=0)
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)


class ControlPanelTaskGenerator:
    """Generator for control panel animation tasks."""
    
    def __init__(self, canvas_size: Tuple[int, int] = (256, 256), 
                 temp_dir: Optional[str] = None):
        """
        Initialize the generator.
        
        Args:
            canvas_size: Size of the canvas in pixels
            temp_dir: Temporary directory for image generation (if None, uses system temp)
        """
        self.canvas_size = canvas_size
        if temp_dir:
            self.temp_dir = temp_dir
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
            self._cleanup_temp = False
        else:
            self.temp_dir = tempfile.mkdtemp(prefix="control_panel_")
            self._cleanup_temp = True
        
        self.panel_generator = ControlPanelGenerator(canvas_size=canvas_size)
        self.renderer = SceneRenderer(canvas_size=canvas_size)
    
    def generate_single_task(self, task_id: str, difficulty: str = "easy", 
                            seed: Optional[int] = None,
                            target_color: str = "red") -> ControlPanelTaskPair:
        """
        Generate a single control panel task.
        
        Args:
            task_id: Unique identifier for the task
            difficulty: "2_lights", "3_lights", "4_lights", "6_lights", etc.
                     or legacy: "easy", "medium", "hard", "very_hard"
            seed: Random seed for reproducibility
            target_color: Target color for all lights ("red", "green", or "blue")
            
        Returns:
            ControlPanelTaskPair instance
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Determine number of lights based on difficulty
        # Difficulty can be "2_lights", "3_lights", "4_lights", "6_lights", etc.
        # Or legacy names: "easy" (2), "medium" (3), "hard" (4), "very_hard" (6)
        if difficulty == "easy" or difficulty == "2_lights":
            num_lights = 2
        elif difficulty == "medium" or difficulty == "3_lights":
            num_lights = 3
        elif difficulty == "hard" or difficulty == "4_lights":
            num_lights = 4
        elif difficulty == "very_hard" or difficulty == "6_lights":
            num_lights = 6
        elif difficulty.endswith("_lights"):
            # Extract number from "N_lights" format
            try:
                num_lights = int(difficulty.split("_")[0])
            except ValueError:
                num_lights = 2
        else:
            num_lights = 2  # Default to 2
        
        # Generate initial panel configuration (random initial state)
        panel_config = self.panel_generator.generate_panel_config(
            num_lights=num_lights, 
            seed=seed
        )
        
        # Determine target position based on target color
        # Reverse mapping: color -> position
        color_to_position = {v: k for k, v in self.panel_generator.POSITION_COLORS.items()}
        target_position = color_to_position[target_color]
        
        # Update target positions: all lights should be at target_position
        target_positions = [target_position] * num_lights
        
        # Update panel_config with target positions
        panel_config["target_positions"] = target_positions
        for i, light in enumerate(panel_config["lights"]):
            light["target_position"] = target_position
            light["target_color"] = target_color
        
        # Generate prompt
        prompt = self._format_prompt(panel_config, target_color)
        
        # Create image paths
        first_image_path = Path(self.temp_dir) / f"{task_id}_first.png"
        final_image_path = Path(self.temp_dir) / f"{task_id}_final.png"
        
        # Use number of lights as difficulty name
        difficulty_name = f"{num_lights}_lights"
        
        # Render first frame: initial lever positions
        self.renderer.render_panel(
            panel_config, 
            panel_config["initial_positions"], 
            first_image_path
        )
        
        # Render final frame: all levers at target position (all lights showing target color)
        self.renderer.render_panel(
            panel_config, 
            target_positions, 
            final_image_path
        )
        
        # Create metadata
        control_panel_data = {
            "panel_config": panel_config,
            "canvas_size": self.canvas_size,
            "target_color": target_color,
        }
        
        # Create task pair
        task_pair = ControlPanelTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(first_image_path),
            final_image_path=str(final_image_path),
            task_category="ControlPanel",
            control_panel_data=control_panel_data,
            difficulty=difficulty_name,  # Use number of lights as difficulty
            num_lights=num_lights,
            created_at=datetime.now().isoformat()
        )
        
        return task_pair
    
    def _format_prompt(self, panel_config: Dict[str, Any], target_color: str) -> str:
        """Format prompt template with panel configuration."""
        template = PROMPTS[DEFAULT_PROMPT_INDEX]
        lights = panel_config["lights"]
        num_lights = len(lights)
        
        # Number to word conversion
        num_words = {
            1: "one",
            2: "two",
            3: "three",
            4: "four",
            5: "five",
            6: "six"
        }
        num_lights_desc = num_words.get(num_lights, str(num_lights))
        
        # Check if there are duplicate colors in initial state
        initial_colors = [light["initial_color"] for light in lights]
        has_duplicates = len(initial_colors) != len(set(initial_colors))
        
        # Build optional notes
        num_lights_note = ""
        if num_lights == 2:
            num_lights_note = " Note that each control panel has exactly three positions, corresponding to the three possible colors."
        
        return template.format(
            num_lights=num_lights_desc,
            target_color=target_color,
            num_lights_note=num_lights_note
        )
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory if we created it."""
        if self._cleanup_temp and Path(self.temp_dir).exists():
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
    
    def __del__(self):
        """Clean up temporary directory if we created it."""
        # Don't auto-cleanup to ensure files are available for dataset.py to copy
        pass


def create_dataset(num_samples: int = 50, 
                  difficulty_distribution: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Create control panel animation dataset.
    
    Args:
        num_samples: Total number of task pairs to generate
        difficulty_distribution: Optional dict like {
            "2_lights": 0.25,
            "3_lights": 0.25,
            "4_lights": 0.25,
            "6_lights": 0.25
        }
        
    Returns:
        Dictionary with 'pairs' key containing list of ControlPanelTaskPair
    """
    if difficulty_distribution is None:
        # Default distribution - equal distribution
        difficulty_distribution = {
            "2_lights": 0.25,
            "3_lights": 0.25,
            "4_lights": 0.25,
            "6_lights": 0.25
        }
    
    print(f"🎯 Creating Control Panel Animation Dataset")
    print(f"   Total samples: {num_samples}")
    
    # Use system temporary directory
    generator = ControlPanelTaskGenerator(temp_dir=None)
    pairs = []
    
    # Calculate number of samples per difficulty
    lights_2_count = int(num_samples * difficulty_distribution.get("2_lights", 0.25))
    lights_3_count = int(num_samples * difficulty_distribution.get("3_lights", 0.25))
    lights_4_count = int(num_samples * difficulty_distribution.get("4_lights", 0.25))
    lights_6_count = num_samples - lights_2_count - lights_3_count - lights_4_count
    
    print(f"   2 lights: {lights_2_count}")
    print(f"   3 lights: {lights_3_count}")
    print(f"   4 lights: {lights_4_count}")
    print(f"   6 lights: {lights_6_count}")
    
    # Generate tasks
    # For each scene, generate 3 independent tasks (one for each target color: red, green, blue)
    task_idx = 0
    target_colors = ["red", "green", "blue"]
    
    for difficulty, count in [("2_lights", lights_2_count), ("3_lights", lights_3_count), 
                              ("4_lights", lights_4_count), ("6_lights", lights_6_count)]:
        for i in range(count):
            # Use number of lights in task ID (extract from difficulty name)
            if difficulty.endswith("_lights"):
                num_lights_str = difficulty.split("_")[0]
            else:
                # Legacy support: map old names to numbers
                num_lights_str = "2" if difficulty == "easy" else \
                                "3" if difficulty == "medium" else \
                                "4" if difficulty == "hard" else \
                                "6" if difficulty == "very_hard" else "2"
            
            # Generate 3 tasks for this scene (one per target color)
            # Use the same seed for all 3 tasks to ensure same initial state
            scene_seed = i * 1000 + hash(difficulty) % 1000
            
            for target_color in target_colors:
                task_id = f"control_panel_{num_lights_str}lights_{task_idx:04d}"
                
                try:
                    task_pair = generator.generate_single_task(
                        task_id=task_id,
                        difficulty=difficulty,
                        seed=scene_seed,  # Same seed for same scene
                        target_color=target_color
                    )
                    pairs.append(task_pair)
                    task_idx += 1
                except Exception as e:
                    # Raise exception to ensure user knows about the problem
                    # This prevents silent failures that could be missed in large outputs
                    raise RuntimeError(
                        f"Failed to generate task {task_id}. This indicates a problem with the generation logic. "
                        f"Original error: {e}"
                    ) from e
            
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{count} {difficulty} scenes ({(i + 1) * 3} tasks)...")
    
    # Convert to dictionary format
    pairs_dict = []
    for pair in pairs:
        pair_dict = {
            "id": pair.id,
            "prompt": pair.prompt,
            "first_image_path": pair.first_image_path,
            "final_image_path": pair.final_image_path,
            "task_category": pair.task_category,
            "control_panel_data": pair.control_panel_data,
            "difficulty": pair.difficulty,
            "num_lights": pair.num_lights,
            "created_at": pair.created_at
        }
        pairs_dict.append(pair_dict)
    
    # Create dataset dictionary
    dataset = {
        "name": "control_panel_tasks",
        "description": f"Control panel animation tasks for video model evaluation ({len(pairs)} pairs)",
        "pairs": pairs_dict,
        "created_at": datetime.now().isoformat()
    }
    
    print(f"\n✅ Dataset creation complete!")
    print(f"   Total tasks: {len(pairs)}")
    
    return dataset

