"""
Sudoku Reasoning Task for VMEvalKit

Simple 3x3 Sudoku puzzle generation system for video model evaluation.
Follows the same data format as maze tasks with first/final frames and prompts.

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
class SudokuTaskPair:
    """
    Data structure for 3x3 sudoku video model evaluation.
    
    Contains:
    - prompt: Instructions for the video model
    - first_image: The puzzle with some numbers filled in
    - final_image: The complete solution
    """
    id: str
    prompt: str                      # What to ask the video model to do
    first_image_path: str           # The puzzle image (incomplete 3x3 sudoku)
    final_image_path: str           # The solution image (complete 3x3 sudoku) 
    task_category: str              # "Sudoku"
    sudoku_data: Dict[str, Any] = None  # Metadata
    difficulty: str = ""            # "easy", "medium", "hard"
    puzzle_array: List[int] = None  # The incomplete puzzle as 3x3 list
    solution_array: List[int] = None # The complete solution as 3x3 list
    num_given: int = 0              # Number of given digits (out of 9)
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class SudokuDataset:
    """Collection of SudokuTaskPair instances."""
    name: str
    description: str
    pairs: List[SudokuTaskPair]
    metadata: Dict[str, Any]
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> SudokuTaskPair:
        return self.pairs[idx]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'SudokuDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert dictionaries back to SudokuTaskPair objects
        pairs = []
        for pair_data in data['pairs']:
            pairs.append(SudokuTaskPair(**pair_data))
        
        data['pairs'] = pairs
        return cls(**data)


class Simple3x3SudokuGenerator:
    """Simple 3x3 Sudoku generator."""
    
    def generate_solved_sudoku(self):
        """Generate a complete, valid 3x3 sudoku solution."""
        # For 3x3, we need each row and column to contain 1, 2, 3 exactly once
        # We'll generate all possible 3x3 Latin squares and pick one randomly
        
        solutions = [
            # Solution 1
            [1, 2, 3,
             2, 3, 1, 
             3, 1, 2],
            
            # Solution 2  
            [1, 2, 3,
             3, 1, 2,
             2, 3, 1],
            
            # Solution 3
            [1, 3, 2,
             2, 1, 3,
             3, 2, 1],
            
            # Solution 4
            [1, 3, 2,
             3, 2, 1,
             2, 1, 3],
            
            # Solution 5
            [2, 1, 3,
             1, 3, 2,
             3, 2, 1],
            
            # Solution 6
            [2, 1, 3,
             3, 2, 1,
             1, 3, 2],
            
            # Solution 7
            [2, 3, 1,
             1, 2, 3,
             3, 1, 2],
            
            # Solution 8
            [2, 3, 1,
             3, 1, 2,
             1, 2, 3],
            
            # Solution 9
            [3, 1, 2,
             1, 2, 3,
             2, 3, 1],
            
            # Solution 10
            [3, 1, 2,
             2, 3, 1,
             1, 2, 3],
            
            # Solution 11
            [3, 2, 1,
             1, 3, 2,
             2, 1, 3],
            
            # Solution 12
            [3, 2, 1,
             2, 1, 3,
             1, 3, 2],
        ]
        
        return random.choice(solutions).copy()
    
    def create_puzzle(self, solution, difficulty_level=1):
        """Create a puzzle by removing numbers from a solution."""
        puzzle = solution.copy()
        
        # Difficulty levels for 3x3 (out of 9 total positions)
        # User requested only 1 missing number for all difficulties
        difficulty_map = {
            0: (1, 1),   # Easy: remove 1 number (leave 8)
            1: (1, 1),   # Medium: remove 1 number (leave 8) 
            2: (1, 1)    # Hard: remove 1 number (leave 8)
        }
        
        min_remove, max_remove = difficulty_map.get(difficulty_level, (1, 1))
        num_to_remove = random.randint(min_remove, max_remove)
        
        positions = list(range(9))
        random.shuffle(positions)
        
        for i in range(num_to_remove):
            puzzle[positions[i]] = None
        
        return puzzle
    
    def validate_solution(self, grid):
        """Validate that a 3x3 grid is a valid sudoku solution."""
        if len(grid) != 9:
            return False
        
        # Convert to 2D for easier checking
        grid_2d = [grid[i:i+3] for i in range(0, 9, 3)]
        
        # Check rows
        for row in grid_2d:
            if sorted(row) != [1, 2, 3]:
                return False
        
        # Check columns
        for col in range(3):
            column = [grid_2d[row][col] for row in range(3)]
            if sorted(column) != [1, 2, 3]:
                return False
        
        return True
    
    def create_board_image(self, sudoku_array, filepath):
        """Create and save a visual representation of the 3x3 sudoku board."""
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        # Create the grid
        for i in range(4):  # 4 lines for 3x3 grid
            ax.axhline(y=i, color='black', linewidth=2)
            ax.axvline(x=i, color='black', linewidth=2)
        
        # Fill in the numbers
        for i in range(3):
            for j in range(3):
                idx = i * 3 + j
                if sudoku_array[idx] is not None:
                    # Place number in center of cell
                    ax.text(j + 0.5, 2.5 - i, str(sudoku_array[idx]), 
                           fontsize=24, ha='center', va='center', 
                           fontweight='bold', color='blue')
                else:
                    # Empty cell - show light gray background
                    rect = patches.Rectangle((j, 2-i), 1, 1, 
                                           linewidth=0, facecolor='lightgray', alpha=0.3)
                    ax.add_patch(rect)
        
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches='tight', dpi=150)
        plt.close()


class SudokuTaskGenerator:
    """Main class for generating 3x3 sudoku reasoning tasks."""
    
    def __init__(self):
        self.sudoku_gen = Simple3x3SudokuGenerator()
        
    def generate_single_task(self, task_id: str, difficulty: int = 1) -> SudokuTaskPair:
        """Generate a single 3x3 sudoku task pair."""
        
        # Use temporary directory like other tasks
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Generate complete solution
        solution = self.sudoku_gen.generate_solved_sudoku()
        
        # Create puzzle by removing numbers
        puzzle = self.sudoku_gen.create_puzzle(solution, difficulty)
        
        # Count given numbers
        num_given = sum(1 for x in puzzle if x is not None)
        
        # Save images in temp directory
        puzzle_path = Path(temp_dir) / f"{task_id}_first.png"
        solution_path = Path(temp_dir) / f"{task_id}_final.png"
        
        self.sudoku_gen.create_board_image(puzzle, puzzle_path)
        self.sudoku_gen.create_board_image(solution, solution_path)
        
        # Create prompt based on difficulty
        difficulty_names = ["easy", "medium", "hard"]
        difficulty_name = difficulty_names[min(difficulty, 2)]
        
        # Use standardized prompt template from PROMPTS list
        prompt = PROMPTS[0].format(difficulty=difficulty_name)
        
        # Create task pair (return temp paths that will be moved by create_dataset.py)
        task_pair = SudokuTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(puzzle_path),
            final_image_path=str(solution_path),
            task_category="Sudoku",
            difficulty=difficulty_name,
            puzzle_array=puzzle,
            solution_array=solution,
            num_given=num_given,
            sudoku_data={
                "puzzle": puzzle,
                "solution": solution,
                "difficulty_level": difficulty,
                "num_given": num_given
            }
        )
        
        return task_pair
    
    def generate_dataset(self, num_samples: int = 10) -> Dict[str, Any]:
        """Generate a dataset of 3x3 sudoku tasks."""
        
        # Use all difficulty levels by default
        difficulties = [0, 1, 2]  # Easy, Medium, Hard
        
        tasks = []
        
        print(f"🧩 Generating {num_samples} 3x3 Sudoku tasks...")
        
        for i in range(num_samples):
            difficulty = random.choice(difficulties)
            task_id = f"sudoku_{i:04d}"
            
            try:
                task = self.generate_single_task(task_id, difficulty)
                tasks.append(task)
                print(f"✅ Generated task {i+1}/{num_samples}: {task_id} ({task.difficulty}) - {task.num_given}/9 given")
            except Exception as e:
                print(f"❌ Failed to generate task {task_id}: {e}")
                continue
        
        # Convert task pairs to dictionaries for consistency with other tasks
        task_dicts = []
        for task in tasks:
            task_dict = {
                'id': task.id,
                'prompt': task.prompt,
                'first_image_path': task.first_image_path,
                'final_image_path': task.final_image_path,
                'task_category': task.task_category,
                'difficulty': task.difficulty,
                'sudoku_data': task.sudoku_data,
                'puzzle_array': task.puzzle_array,
                'solution_array': task.solution_array,
                'num_given': task.num_given,
                'created_at': task.created_at
            }
            task_dicts.append(task_dict)
        
        # Create dataset dictionary for consistency with other tasks
        dataset_dict = {
            'name': "3x3 Sudoku Reasoning Dataset",
            'description': "Simple 3x3 Sudoku puzzles for video model reasoning evaluation",
            'pairs': task_dicts,
            'metadata': {
                "total_tasks": len(tasks),
                "difficulties": difficulties,
                "grid_size": "3x3",
                "generation_date": datetime.now().isoformat(),
                "task_categories": ["Sudoku"]
            }
        }
        
        # Note: We don't save the JSON file here - that's handled by create_dataset.py
        # This matches the pattern used by other tasks (maze, rotation, chess, raven)
        
        # Return the dictionary format for consistency with other tasks
        return dataset_dict


def generate_sudoku_board_image(sudoku_array: List[int], output_path: str) -> str:
    """Utility function to generate a 3x3 sudoku board image from array."""
    generator = Simple3x3SudokuGenerator()
    generator.create_board_image(sudoku_array, output_path)
    return output_path


def create_dataset(num_samples: int = 10) -> Dict[str, Any]:
    """Main function to create 3x3 sudoku dataset - matches API of other tasks."""
    
    generator = SudokuTaskGenerator()
    return generator.generate_dataset(num_samples)


# Dataset creation should only be done via vmevalkit/runner/create_dataset.py
# This module only provides the create_dataset() function as an API
