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
        
        Args:
            num_lights: Number of lights (1, 2, or 3)
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
        
        for i in range(num_lights):
            # Random initial position
            initial_pos = self.rng.choice(self.POSITIONS)
            initial_positions.append(initial_pos)
            
            # Target position: ensure at least one lever needs to move
            # For the first light, if initial is not right, target is right
            # Otherwise, choose a different position
            if i == 0 and initial_pos != "right":
                target_pos = "right"
            elif initial_pos == "right":
                # If already at right, move to middle or left
                target_pos = self.rng.choice(["left", "middle"])
            else:
                # Move to a different position
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
        
        # Calculate layout
        panel_width = self.canvas_size[0] * 0.9
        panel_height = self.canvas_size[1] * 0.8
        panel_x = (self.canvas_size[0] - panel_width) / 2
        panel_y = (self.canvas_size[1] - panel_height) / 2
        
        # Spacing between units
        unit_width = panel_width / num_lights
        unit_spacing = unit_width * 0.1
        
        # Draw each control unit
        for i, light in enumerate(lights):
            unit_x = panel_x + i * unit_width + unit_spacing / 2
            unit_width_actual = unit_width - unit_spacing
            
            # Light position (top)
            light_y = panel_y + panel_height * 0.6
            light_size = min(unit_width_actual * 0.4, panel_height * 0.15)
            light_x = unit_x + unit_width_actual / 2
            
            # Get current color based on lever position
            lever_pos = lever_positions[i]
            current_color = SceneRenderer.POSITION_COLORS[lever_pos]
            
            # Draw indicator light (circle)
            light_circle = Circle((light_x, light_y), light_size / 2,
                                facecolor=self.COLOR_MAP[current_color],
                                edgecolor="black", linewidth=2)
            ax.add_patch(light_circle)
            
            # Draw light border/ring
            light_ring = Circle((light_x, light_y), light_size / 2 + 2,
                              facecolor="none", edgecolor="black", linewidth=1)
            ax.add_patch(light_ring)
            
            # Control slot position (below light)
            slot_y = panel_y + panel_height * 0.25
            slot_height = panel_height * 0.25
            slot_width = unit_width_actual * 0.7
            slot_x = unit_x + (unit_width_actual - slot_width) / 2
            
            # Draw black control slot
            slot = Rectangle((slot_x, slot_y), slot_width, slot_height,
                           facecolor=self.COLOR_MAP["black"],
                           edgecolor="gray", linewidth=2)
            ax.add_patch(slot)
            
            # Draw position markers (left, middle, right)
            marker_size = 3
            left_marker_x = slot_x + slot_width * 0.2
            middle_marker_x = slot_x + slot_width * 0.5
            right_marker_x = slot_x + slot_width * 0.8
            marker_y = slot_y + slot_height / 2
            
            # Draw markers as small circles
            for mx in [left_marker_x, middle_marker_x, right_marker_x]:
                marker = Circle((mx, marker_y), marker_size,
                              facecolor="white", edgecolor="none", alpha=0.5)
                ax.add_patch(marker)
            
            # Draw lever based on position
            lever_width = slot_width * 0.15
            lever_height = slot_height * 0.6
            lever_y = slot_y + (slot_height - lever_height) / 2
            
            if lever_pos == "left":
                lever_x = slot_x + slot_width * 0.1
            elif lever_pos == "middle":
                lever_x = slot_x + slot_width * 0.5 - lever_width / 2
            else:  # right
                lever_x = slot_x + slot_width * 0.9 - lever_width
            
            # Draw lever (rectangle)
            lever = Rectangle((lever_x, lever_y), lever_width, lever_height,
                            facecolor="#808080",  # Gray lever
                            edgecolor="black", linewidth=2)
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
                            seed: Optional[int] = None) -> ControlPanelTaskPair:
        """
        Generate a single control panel task.
        
        Args:
            task_id: Unique identifier for the task
            difficulty: "easy" (1 light), "medium" (2 lights), "hard" (3 lights)
            seed: Random seed for reproducibility
            
        Returns:
            ControlPanelTaskPair instance
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # Determine number of lights based on difficulty
        if difficulty == "easy":
            num_lights = 1
        elif difficulty == "medium":
            num_lights = 2
        elif difficulty == "hard":
            num_lights = 3
        else:
            num_lights = 1
        
        # Generate panel configuration
        panel_config = self.panel_generator.generate_panel_config(
            num_lights=num_lights, 
            seed=seed
        )
        
        # Generate prompt
        prompt = self._format_prompt(panel_config)
        
        # Create image paths
        first_image_path = Path(self.temp_dir) / f"{task_id}_first.png"
        final_image_path = Path(self.temp_dir) / f"{task_id}_final.png"
        
        # Render first frame: initial lever positions
        self.renderer.render_panel(
            panel_config, 
            panel_config["initial_positions"], 
            first_image_path
        )
        
        # Render final frame: target lever positions
        self.renderer.render_panel(
            panel_config, 
            panel_config["target_positions"], 
            final_image_path
        )
        
        # Create metadata
        control_panel_data = {
            "panel_config": panel_config,
            "canvas_size": self.canvas_size,
        }
        
        # Create task pair
        task_pair = ControlPanelTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(first_image_path),
            final_image_path=str(final_image_path),
            task_category="ControlPanel",
            control_panel_data=control_panel_data,
            difficulty=difficulty,
            num_lights=num_lights,
            created_at=datetime.now().isoformat()
        )
        
        return task_pair
    
    def _format_prompt(self, panel_config: Dict[str, Any]) -> str:
        """Format prompt template with panel configuration."""
        template = PROMPTS[DEFAULT_PROMPT_INDEX]
        lights = panel_config["lights"]
        num_lights = len(lights)
        
        # Number to word conversion with proper singular/plural
        num_words = ["one", "two", "three", "four", "five"]
        if num_lights == 1:
            num_lights_description = "one indicator light"
        else:
            num_lights_description = f"{num_words[num_lights - 1] if num_lights <= 5 else str(num_lights)} indicator lights"
        
        # Build initial state description
        initial_parts = []
        for i, light in enumerate(lights):
            pos = light["initial_position"]
            color = light["initial_color"]
            initial_parts.append(
                f"Light {i+1}: lever at {pos} position, showing {color} light"
            )
        initial_state = "\n".join([f"({i+1}) {part}" for i, part in enumerate(initial_parts)])
        
        # Build target state description
        target_parts = []
        lever_actions = []
        for i, light in enumerate(lights):
            initial_pos = light["initial_position"]
            target_pos = light["target_position"]
            target_color = light["target_color"]
            
            if initial_pos != target_pos:
                target_parts.append(
                    f"Light {i+1}: lever should be at {target_pos} position, showing {target_color} light"
                )
                lever_actions.append(
                    f"Move the lever for Light {i+1} from {initial_pos} to {target_pos} position"
                )
            else:
                target_parts.append(
                    f"Light {i+1}: lever remains at {target_pos} position, showing {target_color} light"
                )
        
        target_state = "\n".join([f"({i+1}) {part}" for i, part in enumerate(target_parts)])
        
        # Build lever actions description
        if lever_actions:
            actions_text = "Perform the following lever movements:\n" + "\n".join([f"- {action}" for action in lever_actions])
        else:
            actions_text = "No lever movements are needed (all levers are already in target positions)."
        
        return template.format(
            num_lights_description=num_lights_description,
            initial_state_description=initial_state,
            target_state_description=target_state,
            lever_actions=actions_text
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
            "easy": 0.33,   # 1 light
            "medium": 0.33,  # 2 lights
            "hard": 0.34    # 3 lights
        }
        
    Returns:
        Dictionary with 'pairs' key containing list of ControlPanelTaskPair
    """
    if difficulty_distribution is None:
        # Default distribution - equal distribution
        difficulty_distribution = {
            "easy": 1/3,   # 1 light
            "medium": 1/3, # 2 lights
            "hard": 1/3   # 3 lights
        }
    
    print(f"🎯 Creating Control Panel Animation Dataset")
    print(f"   Total samples: {num_samples}")
    
    # Use system temporary directory
    generator = ControlPanelTaskGenerator(temp_dir=None)
    pairs = []
    
    # Calculate number of samples per difficulty
    easy_count = int(num_samples * difficulty_distribution.get("easy", 1/3))
    medium_count = int(num_samples * difficulty_distribution.get("medium", 1/3))
    hard_count = num_samples - easy_count - medium_count
    
    print(f"   Easy (1 light): {easy_count}")
    print(f"   Medium (2 lights): {medium_count}")
    print(f"   Hard (3 lights): {hard_count}")
    
    # Generate tasks
    task_idx = 0
    for difficulty, count in [("easy", easy_count), ("medium", medium_count), ("hard", hard_count)]:
        for i in range(count):
            task_id = f"control_panel_{difficulty}_{task_idx:04d}"
            # Generate deterministic seed
            seed = task_idx * 1000 + hash(difficulty) % 1000
            
            try:
                task_pair = generator.generate_single_task(
                    task_id=task_id,
                    difficulty=difficulty,
                    seed=seed
                )
                pairs.append(task_pair)
                task_idx += 1
                
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{count} {difficulty} tasks...")
            except Exception as e:
                # Raise exception to ensure user knows about the problem
                # This prevents silent failures that could be missed in large outputs
                raise RuntimeError(
                    f"Failed to generate task {task_id}. This indicates a problem with the generation logic. "
                    f"Original error: {e}"
                ) from e
    
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

