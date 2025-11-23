"""
Clock Reasoning Task for VMEvalKit

Clock time reasoning system for video model evaluation.
The task shows a clock with only the hour hand at a random time,
and asks the model to show the clock after k hours (k from 1 to 24).

Follows the same data format as other tasks with first/final frames and prompts.

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

# Import prompts from centralized location
from .PROMPTS import PROMPTS


@dataclass
class ClockTaskPair:
    """
    Data structure for clock time reasoning video model evaluation.
    
    Contains:
    - prompt: Instructions for the video model
    - first_image: The clock with hour hand at initial time
    - final_image: The clock with hour hand after k hours
    """
    id: str
    prompt: str                      # What to ask the video model to do
    first_image_path: str           # The initial clock image
    final_image_path: str           # The clock image after k hours
    task_category: str              # "Clock"
    clock_data: Dict[str, Any] = None  # Metadata
    initial_hour: int = 0           # Initial hour (0-11, where 0=12 o'clock)
    hours_to_add: int = 0           # Number of hours to add (1-24)
    final_hour: int = 0             # Final hour after adding k hours (0-11)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class ClockDataset:
    """Collection of ClockTaskPair instances."""
    name: str
    description: str
    pairs: List[ClockTaskPair]
    metadata: Dict[str, Any]
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> ClockTaskPair:
        return self.pairs[idx]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'ClockDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert dictionaries back to ClockTaskPair objects
        pairs = []
        for pair_data in data['pairs']:
            pairs.append(ClockTaskPair(**pair_data))
        
        data['pairs'] = pairs
        return cls(**data)


class ClockGenerator:
    """Clock image generator for time reasoning tasks."""
    
    def create_clock_image(self, hour: int, filepath: str, figsize: Tuple[int, int] = (6, 6)):
        """
        Create a clock image with only the hour hand.
        
        Args:
            hour: Hour value (0-11, where 0 represents 12 o'clock)
            filepath: Path to save the image
            figsize: Figure size for matplotlib
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Draw clock circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=3)
        ax.add_patch(circle)
        
        # Draw hour markers (12 positions)
        for i in range(12):
            angle = np.pi / 2 - i * 2 * np.pi / 12  # Start from top (12 o'clock)
            # Outer point for marker
            outer_x = 0.9 * np.cos(angle)
            outer_y = 0.9 * np.sin(angle)
            # Inner point for marker
            inner_x = 0.85 * np.cos(angle)
            inner_y = 0.85 * np.sin(angle)
            
            ax.plot([inner_x, outer_x], [inner_y, outer_y], 'k-', linewidth=2)
        
        # Draw hour numbers (1-12)
        for i in range(12):
            hour_num = 12 if i == 0 else i
            angle = np.pi / 2 - i * 2 * np.pi / 12
            # Position number slightly inside the circle
            text_x = 0.7 * np.cos(angle)
            text_y = 0.7 * np.sin(angle)
            ax.text(text_x, text_y, str(hour_num), 
                   fontsize=16, ha='center', va='center', 
                   fontweight='bold')
        
        # Draw hour hand only
        # Hour hand angle: 0 = 12 o'clock, 1 = 1 o'clock, etc.
        hour_angle = np.pi / 2 - hour * 2 * np.pi / 12
        hour_hand_length = 0.6
        
        hour_hand_x = hour_hand_length * np.cos(hour_angle)
        hour_hand_y = hour_hand_length * np.sin(hour_angle)
        
        ax.plot([0, hour_hand_x], [0, hour_hand_y], 'k-', linewidth=4)
        
        # Draw center dot
        center_dot = plt.Circle((0, 0), 0.05, color='black', fill=True)
        ax.add_patch(center_dot)
        
        # Set equal aspect and limits
        ax.set_aspect('equal')
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()


class ClockTaskGenerator:
    """Main class for generating clock time reasoning tasks."""
    
    def __init__(self):
        self.clock_gen = ClockGenerator()
        
    def generate_single_task(self, task_id: str) -> ClockTaskPair:
        """Generate a single clock task pair."""
        
        # Use temporary directory like other tasks
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Generate random initial hour (0-11, where 0 = 12 o'clock)
        initial_hour = random.randint(0, 11)
        
        # Generate random hours to add (1-24)
        hours_to_add = random.randint(1, 24)
        
        # Calculate final hour (0-11)
        final_hour = (initial_hour + hours_to_add) % 12
        
        # Save images in temp directory
        first_path = Path(temp_dir) / f"{task_id}_first.png"
        final_path = Path(temp_dir) / f"{task_id}_final.png"
        
        self.clock_gen.create_clock_image(initial_hour, str(first_path))
        self.clock_gen.create_clock_image(final_hour, str(final_path))
        
        # Use standardized prompt template from PROMPTS list
        prompt = PROMPTS[0].format(k=hours_to_add)
        
        # Create task pair (return temp paths that will be moved by create_dataset.py)
        task_pair = ClockTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(first_path),
            final_image_path=str(final_path),
            task_category="Clock",
            initial_hour=initial_hour,
            hours_to_add=hours_to_add,
            final_hour=final_hour,
            clock_data={
                "initial_hour": initial_hour,
                "hours_to_add": hours_to_add,
                "final_hour": final_hour,
                "initial_time_display": f"{12 if initial_hour == 0 else initial_hour}:00",
                "final_time_display": f"{12 if final_hour == 0 else final_hour}:00"
            }
        )
        
        return task_pair
    
    def generate_dataset(self, num_samples: int = 10) -> Dict[str, Any]:
        """Generate a dataset of clock time reasoning tasks."""
        
        tasks = []
        
        print(f"🕐 Generating {num_samples} Clock time reasoning tasks...")
        
        for i in range(num_samples):
            task_id = f"clock_{i:04d}"
            
            task = self.generate_single_task(task_id)
            tasks.append(task)
            initial_display = f"{12 if task.initial_hour == 0 else task.initial_hour}:00"
            final_display = f"{12 if task.final_hour == 0 else task.final_hour}:00"
            print(f"✅ Generated task {i+1}/{num_samples}: {task_id} - {initial_display} + {task.hours_to_add}h = {final_display}")
        
        # Convert task pairs to dictionaries for consistency with other tasks
        task_dicts = []
        for task in tasks:
            task_dict = {
                'id': task.id,
                'prompt': task.prompt,
                'first_image_path': task.first_image_path,
                'final_image_path': task.final_image_path,
                'task_category': task.task_category,
                'initial_hour': task.initial_hour,
                'hours_to_add': task.hours_to_add,
                'final_hour': task.final_hour,
                'clock_data': task.clock_data,
                'created_at': task.created_at
            }
            task_dicts.append(task_dict)
        
        # Create dataset dictionary for consistency with other tasks
        dataset_dict = {
            'name': "Clock Time Reasoning Dataset",
            'description': "Clock time reasoning tasks for video model evaluation - showing clock after k hours",
            'pairs': task_dicts,
            'metadata': {
                "total_tasks": len(tasks),
                "hours_range": [1, 24],
                "generation_date": datetime.now().isoformat(),
                "task_categories": ["Clock"]
            }
        }
        
        # Note: We don't save the JSON file here - that's handled by create_dataset.py
        # This matches the pattern used by other tasks (maze, rotation, chess, raven, sudoku)
        
        # Return the dictionary format for consistency with other tasks
        return dataset_dict


def generate_clock_image(hour: int, output_path: str) -> str:
    """Utility function to generate a clock image from hour value."""
    generator = ClockGenerator()
    generator.create_clock_image(hour, output_path)
    return output_path


def create_dataset(num_samples: int = 10) -> Dict[str, Any]:
    """Main function to create clock time reasoning dataset - matches API of other tasks."""
    
    generator = ClockTaskGenerator()
    return generator.generate_dataset(num_samples)


# Dataset creation should only be done via vmevalkit/runner/create_dataset.py
# This module only provides the create_dataset() function as an API

