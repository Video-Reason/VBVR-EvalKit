"""
Light Sequence Reasoning Task for VMEvalKit

Light sequence reasoning system for video model evaluation.
The task shows a row of lights in random on/off states and asks the model
to manipulate them according to spatial and mathematical instructions.

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
from .PROMPTS import TYPE_PROMPTS, DEFAULT_PROMPT_INDEX


@dataclass
class LightSequenceTaskPair:
    """
    Data structure for light sequence reasoning video model evaluation.
    
    Contains:
    - prompt: Instructions for the video model
    - first_image: The lights in initial state
    - final_image: The lights in target state
    """
    id: str
    prompt: str                      # What to ask the video model to do
    first_image_path: str           # The initial light sequence image
    final_image_path: str           # The target light sequence image
    task_category: str              # "Light Sequence"
    light_sequence_data: Dict[str, Any] = None  # Metadata
    num_lights: int = 0             # Number of lights (4, 6, 8, or 10)
    initial_state: str = ""         # Binary string representing initial state
    final_state: str = ""           # Binary string representing final state
    task_type: int = 0              # Type 1-6
    question_index: int = 0         # 0 or 1 (first or second question of the type)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class LightSequenceDataset:
    """Collection of LightSequenceTaskPair instances."""
    name: str
    description: str
    pairs: List[LightSequenceTaskPair]
    metadata: Dict[str, Any]
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> LightSequenceTaskPair:
        return self.pairs[idx]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'LightSequenceDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert dictionaries back to LightSequenceTaskPair objects
        pairs = []
        for pair_data in data['pairs']:
            pairs.append(LightSequenceTaskPair(**pair_data))
        
        data['pairs'] = pairs
        return cls(**data)


class LightSequenceGenerator:
    """Light sequence image generator for spatial reasoning tasks."""
    
    def render_light_sequence(self, num_lights: int, state: Union[str, List[int]], 
                             output_path: str, figsize: Tuple[int, int] = (8, 2)):
        """
        Create a light sequence image.
        
        Args:
            num_lights: Number of lights (4, 6, 8, or 10)
            state: Binary string (e.g., "1010") or list of 0/1 representing light states
            output_path: Path to save the image
            figsize: Figure size for matplotlib
        """
        # Convert state to list of integers
        if isinstance(state, str):
            state_list = [int(c) for c in state]
        else:
            state_list = list(state)
        
        # Ensure state matches num_lights
        if len(state_list) != num_lights:
            raise ValueError(f"State length {len(state_list)} doesn't match num_lights {num_lights}")
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
        # Calculate spacing
        total_width = num_lights * 1.5  # Each light takes 1.5 units
        start_x = -total_width / 2 + 0.75
        
        # Draw lights
        for i in range(num_lights):
            x = start_x + i * 1.5
            is_on = state_list[i] == 1
            
            # Light color: yellow/white when on, gray when off
            color = '#FFD700' if is_on else '#808080'  # Gold when on, gray when off
            edge_color = '#FFA500' if is_on else '#404040'  # Orange edge when on, dark gray when off
            
            # Draw light circle
            circle = plt.Circle((x, 0), 0.4, facecolor=color, edgecolor=edge_color, 
                               linewidth=3, zorder=2)
            ax.add_patch(circle)
            
            # Add glow effect for on lights
            if is_on:
                glow = plt.Circle((x, 0), 0.5, color=color, alpha=0.3, zorder=1)
                ax.add_patch(glow)
        
        # Set equal aspect and limits
        ax.set_aspect('equal')
        ax.set_xlim(-total_width / 2 - 0.5, total_width / 2 + 0.5)
        ax.set_ylim(-1, 1)
        ax.axis('off')
        ax.set_facecolor('white')
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150, facecolor='white')
        plt.close()


class LightSequenceTaskGenerator:
    """Main class for generating light sequence reasoning tasks."""
    
    # Fixed initial states for each light count (14 states each, excluding all-on/all-off)
    INITIAL_STATES = {
        4: [
            # Removed: '0000' (all off), '1111' (all on)
            '1000', '0001', '0110', '1001', '0101', '1010',
            '1100', '0011', '1110', '0111', '1101', '1011', '0100', '0010'
        ],
        6: [
            # Removed: '011100' (index 0), '101000' (index 3)
            '100101', '010011', '000110', '110010', '001101', '111001',
            '010100', '101101', '011010', '100011', '110100', '001011', '101110', '010001'
        ],
        8: [
            # Removed: '11010101' (index 3), '01010011' (index 8)
            '01101001', '10110010', '01001100', '00111011', '10010110', '01100101', '10101100',
            '11100010', '00011101', '11001011', '01110100', '10100111', '01011010', '11001110'
        ],
        10: [
            # Removed: '0101101010' (index 2), '1010101101' (index 7)
            '0110100110', '1011010011', '1100110101', '0011011100', '1001010110', '0110011011',
            '0100110011', '1110001010', '0001110101', '1101000111', '0111100010', '1010011010', '0101011011', '1100101100'
        ]
    }
    
    # Type definitions for each light count
    TYPE_DEFINITIONS = {
        4: {
            1: [
                {'positions': [1], 'desc': '1st'},
                {'positions': [4], 'desc': '4th'}
            ],
            2: [
                {'positions': [1, 3], 'desc': '1st and 3rd'},
                {'positions': [2, 4], 'desc': '2nd and 4th'}
            ],
            3: [
                {'pattern': 'odd', 'desc': 'odd'},
                {'pattern': 'even', 'desc': 'even'}
            ],
            4: [
                {'range': 'left', 'desc': 'left half'},
                {'range': 'right', 'desc': 'right half'}
            ],
            5: [
                {'start': 1, 'end': 3, 'desc_start': '1st', 'desc_end': '3rd'},
                {'start': 2, 'end': 4, 'desc_start': '2nd', 'desc_end': '4th'}
            ],
            6: [
                {'count': 2, 'side': 'left', 'desc': 'leftmost 2'},
                {'count': 2, 'side': 'right', 'desc': 'rightmost 2'}
            ]
        },
        6: {
            1: [
                {'positions': [1], 'desc': '1st'},
                {'positions': [6], 'desc': '6th'}
            ],
            2: [
                {'positions': [1, 3, 5], 'desc': '1st, 3rd, and 5th'},
                {'positions': [2, 4, 6], 'desc': '2nd, 4th, and 6th'}
            ],
            3: [
                {'pattern': 'odd', 'desc': 'odd'},
                {'pattern': 'even', 'desc': 'even'}
            ],
            4: [
                {'range': 'left', 'desc': 'left half'},
                {'range': 'right', 'desc': 'right half'}
            ],
            5: [
                {'start': 1, 'end': 4, 'desc_start': '1st', 'desc_end': '4th'},
                {'start': 3, 'end': 6, 'desc_start': '3rd', 'desc_end': '6th'}
            ],
            6: [
                {'count': 2, 'side': 'left', 'desc': 'leftmost 2'},
                {'count': 2, 'side': 'right', 'desc': 'rightmost 2'}
            ]
        },
        8: {
            1: [
                {'positions': [1], 'desc': '1st'},
                {'positions': [8], 'desc': '8th'}
            ],
            2: [
                {'positions': [2, 5, 7], 'desc': '2nd, 5th, and 7th'},
                {'positions': [1, 4, 8], 'desc': '1st, 4th, and 8th'}
            ],
            3: [
                {'pattern': 'odd', 'desc': 'odd'},
                {'pattern': 'even', 'desc': 'even'}
            ],
            4: [
                {'range': 'left', 'desc': 'left half'},
                {'range': 'right', 'desc': 'right half'}
            ],
            5: [
                {'start': 2, 'end': 5, 'desc_start': '2nd', 'desc_end': '5th'},
                {'start': 4, 'end': 7, 'desc_start': '4th', 'desc_end': '7th'}
            ],
            6: [
                {'count': 2, 'side': 'left', 'desc': 'leftmost 2'},
                {'count': 2, 'side': 'right', 'desc': 'rightmost 2'}
            ]
        },
        10: {
            1: [
                {'positions': [1], 'desc': '1st'},
                {'positions': [10], 'desc': '10th'}
            ],
            2: [
                {'positions': [1, 4, 7, 9], 'desc': '1st, 4th, 7th, and 9th'},
                {'positions': [2, 5, 8, 10], 'desc': '2nd, 5th, 8th, and 10th'}
            ],
            3: [
                {'pattern': 'odd', 'desc': 'odd'},
                {'pattern': 'even', 'desc': 'even'}
            ],
            4: [
                {'range': 'left', 'desc': 'left half'},
                {'range': 'right', 'desc': 'right half'}
            ],
            5: [
                {'start': 3, 'end': 7, 'desc_start': '3rd', 'desc_end': '7th'},
                {'start': 4, 'end': 8, 'desc_start': '4th', 'desc_end': '8th'}
            ],
            6: [
                {'count': 2, 'side': 'left', 'desc': 'leftmost 2'},
                {'count': 2, 'side': 'right', 'desc': 'rightmost 2'}
            ]
        }
    }
    
    def __init__(self):
        self.light_gen = LightSequenceGenerator()
    
    def _calculate_final_state(self, num_lights: int, task_type: int, question_index: int) -> str:
        """
        Calculate the final state based on task type and question.
        
        Args:
            num_lights: Number of lights
            task_type: Type 1-6
            question_index: 0 or 1 (first or second question)
        
        Returns:
            Binary string representing final state
        """
        type_def = self.TYPE_DEFINITIONS[num_lights][task_type][question_index]
        final_state = ['0'] * num_lights
        
        if task_type == 1:  # Single point
            pos = type_def['positions'][0]
            final_state[pos - 1] = '1'
        
        elif task_type == 2:  # Multiple points
            for pos in type_def['positions']:
                final_state[pos - 1] = '1'
        
        elif task_type == 3:  # Mathematical pattern
            pattern = type_def['pattern']
            for i in range(num_lights):
                pos = i + 1  # 1-indexed
                if pattern == 'odd' and pos % 2 == 1:
                    final_state[i] = '1'
                elif pattern == 'even' and pos % 2 == 0:
                    final_state[i] = '1'
        
        elif task_type == 4:  # Spatial range
            range_type = type_def['range']
            half = num_lights // 2
            if range_type == 'left':
                for i in range(half):
                    final_state[i] = '1'
            else:  # right
                for i in range(half, num_lights):
                    final_state[i] = '1'
        
        elif task_type == 5:  # Continuous sequence
            start = type_def['start'] - 1  # Convert to 0-indexed
            end = type_def['end']  # Keep 1-indexed for range
            for i in range(start, end):
                final_state[i] = '1'
        
        elif task_type == 6:  # Relative position
            count = type_def['count']
            side = type_def['side']
            if side == 'left':
                for i in range(count):
                    final_state[i] = '1'
            else:  # right
                for i in range(num_lights - count, num_lights):
                    final_state[i] = '1'
        
        return ''.join(final_state)
    
    def _format_prompt(self, num_lights: int, task_type: int, question_index: int) -> str:
        """
        Format the prompt based on task type and question.
        
        Args:
            num_lights: Number of lights
            task_type: Type 1-6
            question_index: 0 or 1
        
        Returns:
            Formatted prompt string
        """
        type_def = self.TYPE_DEFINITIONS[num_lights][task_type][question_index]
        prompt_template = TYPE_PROMPTS[task_type][question_index]
        
        if task_type == 1:  # Single point
            return prompt_template.format(position_desc=type_def['desc'])
        
        elif task_type == 2:  # Multiple points
            return prompt_template.format(position_desc=type_def['desc'])
        
        elif task_type == 3:  # Mathematical pattern
            return prompt_template  # No formatting needed
        
        elif task_type == 4:  # Spatial range
            return prompt_template  # No formatting needed
        
        elif task_type == 5:  # Continuous sequence
            return prompt_template.format(
                start_desc=type_def['desc_start'],
                end_desc=type_def['desc_end']
            )
        
        elif task_type == 6:  # Relative position
            return prompt_template  # No formatting needed
        
        return prompt_template
    
    def generate_single_task(self, num_lights: int, state_idx: int, 
                            task_type: int, question_index: int,
                            task_id: str, output_dir: Path) -> LightSequenceTaskPair:
        """
        Generate a single light sequence task.
        
        Args:
            num_lights: Number of lights (4, 6, 8, or 10)
            state_idx: Index of initial state (0-13)
            task_type: Type 1-6
            question_index: 0 or 1
            task_id: Unique task ID
            output_dir: Directory to save images
        """
        # Get initial state
        initial_state = self.INITIAL_STATES[num_lights][state_idx]
        
        # Calculate final state
        final_state = self._calculate_final_state(num_lights, task_type, question_index)
        
        # Generate images
        first_path = output_dir / f"{task_id}_first.png"
        final_path = output_dir / f"{task_id}_final.png"
        
        self.light_gen.render_light_sequence(num_lights, initial_state, str(first_path))
        self.light_gen.render_light_sequence(num_lights, final_state, str(final_path))
        
        # Format prompt
        prompt = self._format_prompt(num_lights, task_type, question_index)
        
        # Create task pair
        task_pair = LightSequenceTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(first_path),
            final_image_path=str(final_path),
            task_category="Light Sequence",
            light_sequence_data={
                'num_lights': num_lights,
                'initial_state': initial_state,
                'final_state': final_state,
                'task_type': task_type,
                'question_index': question_index,
                'state_idx': state_idx
            },
            num_lights=num_lights,
            initial_state=initial_state,
            final_state=final_state,
            task_type=task_type,
            question_index=question_index
        )
        
        return task_pair
    
    def generate_dataset(self, num_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a dataset of light sequence reasoning tasks.
        
        Args:
            num_samples: If None, generate all 672 tasks. Otherwise, randomly sample.
        
        Returns:
            Dataset dictionary
        """
        all_tasks = []
        
        # Generate all tasks
        for num_lights in [4, 6, 8, 10]:
            for state_idx in range(14):  # 14 states per light count
                for task_type in range(1, 7):
                    for question_index in [0, 1]:
                        # Generate task ID: light_sequence_灯数_题目类型
                        task_id = f"light_sequence_{num_lights}_type{task_type}_s{state_idx:02d}_q{question_index}"
                        
                        # Use temporary directory
                        import tempfile
                        temp_dir = Path(tempfile.mkdtemp())
                        
                        task = self.generate_single_task(
                            num_lights=num_lights,
                            state_idx=state_idx,
                            task_type=task_type,
                            question_index=question_index,
                            task_id=task_id,
                            output_dir=temp_dir
                        )
                        all_tasks.append(task)
        
        # Sample if requested
        if num_samples is not None and num_samples < len(all_tasks):
            all_tasks = random.sample(all_tasks, num_samples)
        
        # Convert to dictionaries
        task_dicts = []
        for task in all_tasks:
            task_dict = {
                'id': task.id,
                'prompt': task.prompt,
                'first_image_path': task.first_image_path,
                'final_image_path': task.final_image_path,
                'task_category': task.task_category,
                'num_lights': task.num_lights,
                'initial_state': task.initial_state,
                'final_state': task.final_state,
                'task_type': task.task_type,
                'question_index': task.question_index,
                'light_sequence_data': task.light_sequence_data,
                'created_at': task.created_at
            }
            task_dicts.append(task_dict)
        
        # Create dataset dictionary
        dataset_dict = {
            'name': "Light Sequence Reasoning Dataset",
            'description': "Light sequence reasoning tasks for video model evaluation - spatial and mathematical pattern recognition",
            'pairs': task_dicts,
            'metadata': {
                "total_tasks": len(task_dicts),
                "light_counts": [4, 6, 8, 10],
                "task_types": list(range(1, 7)),
                "initial_states_per_count": 14,
                "generation_date": datetime.now().isoformat(),
                "task_categories": ["Light Sequence"]
            }
        }
        
        return dataset_dict


def render_light_sequence(num_lights: int, state: Union[str, List[int]], output_path: str) -> str:
    """Utility function to generate a light sequence image."""
    generator = LightSequenceGenerator()
    generator.render_light_sequence(num_lights, state, output_path)
    return output_path


def create_dataset(num_samples: Optional[int] = None) -> Dict[str, Any]:
    """Main function to create light sequence reasoning dataset - matches API of other tasks."""
    
    generator = LightSequenceTaskGenerator()
    return generator.generate_dataset(num_samples)

