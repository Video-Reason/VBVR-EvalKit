"""
Sliding Puzzle Reasoning Task for VMEvalKit

Simple sliding puzzle generation system for video model evaluation.
Puzzles are designed to be near-complete, requiring only 1-2 moves to solve.

Author: VMEvalKit Team
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

# Import prompts from centralized location
from .PROMPTS import PROMPTS, DEFAULT_PROMPT_INDEX


@dataclass
class SlidingPuzzleTaskPair:
    """
    Data structure for sliding puzzle video model evaluation.
    
    Contains:
    - prompt: Instructions for the video model
    - first_image: The near-complete puzzle (1-2 moves from solution)
    - final_image: The complete solution
    """
    id: str
    prompt: str                      # What to ask the video model to do
    first_image_path: str           # The near-complete puzzle image
    final_image_path: str           # The solution image (complete puzzle)
    task_category: str              # "SlidingPuzzle"
    puzzle_data: Dict[str, Any] = None  # Metadata
    difficulty: str = ""            # "easy", "medium", "hard"
    puzzle_size: Tuple[int, int] = (0, 0)  # (3, 3), (4, 4), or (5, 5)
    initial_state: List[List[int]] = None  # The near-complete state matrix
    goal_state: List[List[int]] = None    # The complete solution matrix
    solution_length: int = 0        # Number of moves needed (1-3)
    num_moves_from_complete: int = 0  # Number of moves from complete state
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


class SlidingPuzzleGenerator:
    """
    Generator for sliding puzzle tasks.
    Creates near-complete puzzles (1-2 moves from solution).
    """
    
    def __init__(self, canvas_size: int = 400, temp_dir: Optional[str] = None):
        """
        Initialize the generator.
        
        Args:
            canvas_size: Size of the puzzle image in pixels
            temp_dir: Temporary directory for image generation (if None, uses system temp)
        """
        self.canvas_size = canvas_size
        if temp_dir:
            self.temp_dir = temp_dir
            Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
            self._cleanup_temp = False
        else:
            # Use system temporary directory, will be cleaned up automatically
            self.temp_dir = tempfile.mkdtemp(prefix="sliding_puzzle_")
            self._cleanup_temp = True
        self.rng = random.Random()
    
    def cleanup_temp_dir(self):
        """Clean up temporary directory if we created it.
        
        This should be called after dataset.py has finished copying files.
        """
        if self._cleanup_temp and Path(self.temp_dir).exists():
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
    
    def __del__(self):
        """Clean up temporary directory if we created it.
        
        Note: This is called when the object is garbage collected.
        We delay cleanup to allow dataset.py to copy files first.
        """
        # Try to cleanup, but don't fail if already cleaned up
        if self._cleanup_temp and Path(self.temp_dir).exists():
            try:
                import shutil
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass  # Ignore cleanup errors
    
    def create_goal_state(self, size: int) -> List[List[int]]:
        """
        Create the goal state (complete puzzle).
        
        Args:
            size: Puzzle size (3, 4, or 5)
            
        Returns:
            2D list representing the goal state
        """
        goal = []
        num = 1
        for i in range(size):
            row = []
            for j in range(size):
                if i == size - 1 and j == size - 1:
                    row.append(0)  # Empty space at bottom-right
                else:
                    row.append(num)
                    num += 1
            goal.append(row)
        return goal
    
    def find_blank(self, puzzle: List[List[int]]) -> Tuple[int, int]:
        """Find the position of the blank (0) in the puzzle."""
        for i in range(len(puzzle)):
            for j in range(len(puzzle[i])):
                if puzzle[i][j] == 0:
                    return (i, j)
        raise ValueError("No blank space found in puzzle")
    
    def get_valid_moves(self, puzzle: List[List[int]], blank_pos: Tuple[int, int]) -> List[str]:
        """
        Get valid move directions from the blank position.
        
        Args:
            puzzle: Current puzzle state
            blank_pos: (row, col) of blank space
            
        Returns:
            List of valid directions: ['up', 'down', 'left', 'right']
        """
        row, col = blank_pos
        size = len(puzzle)
        valid = []
        
        if row > 0:
            valid.append('up')
        if row < size - 1:
            valid.append('down')
        if col > 0:
            valid.append('left')
        if col < size - 1:
            valid.append('right')
        
        return valid
    
    def apply_move(self, puzzle: List[List[int]], blank_pos: Tuple[int, int], 
                   direction: str) -> Tuple[List[List[int]], Tuple[int, int]]:
        """
        Apply a move to the puzzle.
        
        Args:
            puzzle: Current puzzle state (will be modified)
            blank_pos: Current blank position
            direction: 'up', 'down', 'left', or 'right'
            
        Returns:
            (new_puzzle, new_blank_pos)
        """
        # Deep copy to avoid modifying original
        new_puzzle = [row[:] for row in puzzle]
        row, col = blank_pos
        
        # Determine swap position
        if direction == 'up':
            swap_row, swap_col = row - 1, col
        elif direction == 'down':
            swap_row, swap_col = row + 1, col
        elif direction == 'left':
            swap_row, swap_col = row, col - 1
        elif direction == 'right':
            swap_row, swap_col = row, col + 1
        else:
            raise ValueError(f"Invalid direction: {direction}")
        
        # Swap blank with adjacent tile
        new_puzzle[row][col], new_puzzle[swap_row][swap_col] = \
            new_puzzle[swap_row][swap_col], new_puzzle[row][col]
        
        return new_puzzle, (swap_row, swap_col)
    
    def generate_near_complete_puzzle(self, size: int, num_moves: int, 
                                     seed: Optional[int] = None) -> Tuple[List[List[int]], int]:
        """
        Generate a near-complete puzzle by starting from goal state and making N moves.
        
        Args:
            size: Puzzle size (3, 4, or 5)
            num_moves: Number of moves to make from goal state (1-2)
            seed: Random seed for reproducibility
            
        Returns:
            (puzzle_state, solution_length)
        """
        if seed is not None:
            self.rng.seed(seed)
        
        # Start from goal state
        goal = self.create_goal_state(size)
        puzzle = [row[:] for row in goal]  # Deep copy
        blank_pos = self.find_blank(puzzle)
        
        # Make num_moves random moves (reverse from goal)
        moves = []
        last_direction = None
        
        for _ in range(num_moves):
            valid_directions = self.get_valid_moves(puzzle, blank_pos)
            
            # Avoid moving back to previous position (unless it's the only option)
            if last_direction and len(valid_directions) > 1:
                # Remove the reverse direction
                reverse_map = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
                reverse = reverse_map.get(last_direction)
                if reverse in valid_directions:
                    valid_directions.remove(reverse)
            
            if not valid_directions:
                break
            
            direction = self.rng.choice(valid_directions)
            puzzle, blank_pos = self.apply_move(puzzle, blank_pos, direction)
            moves.append(direction)
            last_direction = direction
        
        # Solution length equals number of moves made (reverse them to solve)
        solution_length = len(moves)
        
        return puzzle, solution_length
    
    def render_puzzle(self, puzzle: List[List[int]], size: int, 
                     output_path: Union[str, Path], title: str = ""):
        """
        Render the puzzle as an image.
        
        Args:
            puzzle: 2D list representing puzzle state
            size: Puzzle size (3, 4, or 5)
            output_path: Path to save the image
            title: Optional title text
        """
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.set_xlim(0, size)
        ax.set_ylim(0, size)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Draw grid lines first
        for i in range(size + 1):
            # Vertical lines
            ax.plot([i, i], [0, size], '-', linewidth=2, color='#333333')
            # Horizontal lines
            ax.plot([0, size], [i, i], '-', linewidth=2, color='#333333')
        
        # Calculate tile size
        tile_size = 0.9  # Slightly smaller than 1 to show grid lines
        margin = (1 - tile_size) / 2
        
        # Adjust font size based on puzzle size (increased for better video recognition)
        if size == 3:
            fontsize = 32
        elif size == 4:
            fontsize = 24
        else:  # size == 5
            fontsize = 20
        
        # Draw tiles
        for i in range(size):
            for j in range(size):
                value = puzzle[i][j]
                x = j + margin
                y = size - 1 - i + margin  # Flip Y axis
                
                if value == 0:
                    # Empty space - white background (no tile, just grid lines)
                    # Don't draw anything, let the white background show through
                    pass
                else:
                    # Numbered tile - draw as colored rectangle
                    rect = patches.Rectangle(
                        (x, y), tile_size, tile_size,
                        linewidth=2, edgecolor='#333333', facecolor='#4A90E2'
                    )
                    ax.add_patch(rect)
                    # Add number
                    ax.text(x + tile_size/2, y + tile_size/2, str(value),
                           ha='center', va='center', fontsize=fontsize, 
                           color='white', weight='bold')
        
        # Add title if provided
        if title:
            ax.text(size/2, size + 0.1, title, ha='center', va='bottom',
                   fontsize=14, weight='bold')
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def generate_single_task(self, task_id: str, difficulty: str = "easy", 
                            seed: Optional[int] = None) -> SlidingPuzzleTaskPair:
        """
        Generate a single sliding puzzle task.
        
        Args:
            task_id: Unique identifier for the task
            difficulty: "easy", "medium", or "hard"
            seed: Random seed for reproducibility
            
        Returns:
            SlidingPuzzleTaskPair instance
        """
        if seed is not None:
            self.rng.seed(seed)
        
        # Determine puzzle size and number of moves based on difficulty
        # All difficulties use 2 moves
        num_moves = 2
        if difficulty == "easy":
            size = 3
        elif difficulty == "medium":
            size = 4
        else:  # hard = 5x5
            size = 5
        
        # Generate near-complete puzzle
        initial_state, solution_length = self.generate_near_complete_puzzle(
            size, num_moves, seed=seed
        )
        
        # Create goal state
        goal_state = self.create_goal_state(size)
        
        # Render images
        first_path = Path(self.temp_dir) / f"{task_id}_first.png"
        final_path = Path(self.temp_dir) / f"{task_id}_final.png"
        
        self.render_puzzle(initial_state, size, first_path)
        self.render_puzzle(goal_state, size, final_path)
        
        # Generate prompt with dynamic move count
        # Note: solution_length might be less than num_moves if generation was cut short
        actual_moves = solution_length
        prompt = PROMPTS[DEFAULT_PROMPT_INDEX].format(
            num_moves=actual_moves,
            plural="s" if actual_moves > 1 else ""
        )
        
        # Create task pair
        return SlidingPuzzleTaskPair(
            id=task_id,
            prompt=prompt,
            first_image_path=str(first_path),
            final_image_path=str(final_path),
            task_category="SlidingPuzzle",
            puzzle_data={
                "generation_method": "near_complete",
                "solution_length": solution_length,
                "num_moves_from_complete": num_moves
            },
            difficulty=difficulty,
            puzzle_size=(size, size),
            initial_state=initial_state,
            goal_state=goal_state,
            solution_length=solution_length,
            num_moves_from_complete=num_moves
        )


def create_dataset(num_samples: int = 50, difficulty_distribution: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Create sliding puzzle dataset with duplicate state detection.
    Supports both 1-step and 2-step moves for 3×3, 4×4, and 5×5 puzzles.
    
    Args:
        num_samples: Total number of task pairs to generate
        difficulty_distribution: Optional dict like {
            "easy_1step": 0.166,   # 3×3, 1 move
            "easy_2step": 0.166,   # 3×3, 2 moves
            "medium_1step": 0.166, # 4×4, 1 move
            "medium_2step": 0.166, # 4×4, 2 moves
            "hard_1step": 0.166,   # 5×5, 1 move
            "hard_2step": 0.166    # 5×5, 2 moves
        }
        
    Returns:
        Dictionary with 'pairs' key containing list of SlidingPuzzleTaskPair
    """
    if difficulty_distribution is None:
        # Default distribution - equal distribution across all 6 types
        difficulty_distribution = {
            "easy_1step": 1/6,   # 3×3, 1 move
            "easy_2step": 1/6,   # 3×3, 2 moves
            "medium_1step": 1/6, # 4×4, 1 move
            "medium_2step": 1/6, # 4×4, 2 moves
            "hard_1step": 1/6,   # 5×5, 1 move
            "hard_2step": 1/6    # 5×5, 2 moves
        }
    
    print(f"🧩 Creating Sliding Puzzle Dataset")
    print(f"   Total samples: {num_samples}")
    
    # Use system temporary directory
    # The temp directory will be cleaned up at the end of this function
    generator = SlidingPuzzleGenerator(temp_dir=None)
    pairs = []
    
    # Track seen states to avoid duplicates
    seen_states = set()
    
    # Helper function to convert state to hashable tuple
    def state_to_tuple(state):
        return tuple(tuple(row) for row in state)
    
    # Calculate number of samples per type
    easy_1step_count = int(num_samples * difficulty_distribution.get("easy_1step", 1/6))
    easy_2step_count = int(num_samples * difficulty_distribution.get("easy_2step", 1/6))
    medium_1step_count = int(num_samples * difficulty_distribution.get("medium_1step", 1/6))
    medium_2step_count = int(num_samples * difficulty_distribution.get("medium_2step", 1/6))
    hard_1step_count = int(num_samples * difficulty_distribution.get("hard_1step", 1/6))
    hard_2step_count = num_samples - easy_1step_count - easy_2step_count - medium_1step_count - medium_2step_count - hard_1step_count
    
    print(f"   3×3, 1 move: {easy_1step_count}")
    print(f"   3×3, 2 moves: {easy_2step_count}")
    print(f"   4×4, 1 move: {medium_1step_count}")
    print(f"   4×4, 2 moves: {medium_2step_count}")
    print(f"   5×5, 1 move: {hard_1step_count}")
    print(f"   5×5, 2 moves: {hard_2step_count}")
    
    # Generate tasks for all 6 types
    task_idx = 0
    max_retries_per_task = 50
    max_total_retries = 1000
    
    # Define task types: (size, num_moves, difficulty_name)
    task_types = [
        (3, 1, "easy_1step", easy_1step_count),
        (3, 2, "easy_2step", easy_2step_count),
        (4, 1, "medium_1step", medium_1step_count),
        (4, 2, "medium_2step", medium_2step_count),
        (5, 1, "hard_1step", hard_1step_count),
        (5, 2, "hard_2step", hard_2step_count),
    ]
    
    for size, num_moves, difficulty_name, count in task_types:
        generated = 0
        total_retries = 0
        consecutive_failures = 0
        
        while generated < count and total_retries < max_total_retries:
            task_id = f"sliding_puzzle_{task_idx:04d}"
            
            # Generate puzzle with specific size and moves
            initial_state, solution_length = generator.generate_near_complete_puzzle(
                size, num_moves, seed=task_idx * 1000 + total_retries
            )
            
            # Create goal state
            goal_state = generator.create_goal_state(size)
            
            # Render images
            first_path = Path(generator.temp_dir) / f"{task_id}_first.png"
            final_path = Path(generator.temp_dir) / f"{task_id}_final.png"
            
            generator.render_puzzle(initial_state, size, first_path)
            generator.render_puzzle(goal_state, size, final_path)
            
            # Generate prompt
            actual_moves = solution_length
            prompt = PROMPTS[DEFAULT_PROMPT_INDEX].format(
                num_moves=actual_moves,
                plural="s" if actual_moves > 1 else ""
            )
            
            # Create task pair
            pair = SlidingPuzzleTaskPair(
                id=task_id,
                prompt=prompt,
                first_image_path=str(first_path),
                final_image_path=str(final_path),
                task_category="SlidingPuzzle",
                puzzle_data={
                    "generation_method": "near_complete",
                    "solution_length": solution_length,
                    "num_moves_from_complete": num_moves
                },
                difficulty=difficulty_name,
                puzzle_size=(size, size),
                initial_state=initial_state,
                goal_state=goal_state,
                solution_length=solution_length,
                num_moves_from_complete=num_moves
            )
            
            # Check if state is unique
            state_key = (pair.puzzle_size, state_to_tuple(pair.initial_state))
            
            if state_key not in seen_states:
                seen_states.add(state_key)
                pairs.append(pair)
                generated += 1
                task_idx += 1
                consecutive_failures = 0
            else:
                total_retries += 1
                consecutive_failures += 1
                
                if consecutive_failures >= max_retries_per_task:
                    print(f"   ⚠️  Warning: Reached maximum unique states for {difficulty_name}. "
                          f"Generated {generated}/{count} unique tasks.")
                    break
        
        if generated < count:
            print(f"   ⚠️  Warning: Only generated {generated}/{count} unique {difficulty_name} tasks "
                  f"(possible states exhausted)")
    
    print(f"✅ Generated {len(pairs)} unique sliding puzzle tasks")
    print(f"   Unique states: {len(seen_states)}")
    
    # Convert dataclass instances to dictionaries for serialization
    pairs_dict = [asdict(pair) for pair in pairs]
    
    # Clean up temporary directory before returning
    # This should be done in the task-specific function, not in the main dataset.py
    generator.cleanup_temp_dir()
    
    # Return dataset without generator reference
    dataset = {
        "name": "sliding_puzzle_tasks",
        "description": f"Sliding puzzle dataset ({len(pairs)} pairs)",
        "pairs": pairs_dict
    }
    
    return dataset


def regenerate_image_from_state(state: List[List[int]], size: int, output_path: Union[str, Path]):
    """
    Regenerate puzzle image from state data.
    
    This is a helper function for dataset.py to regenerate images if needed.
    
    Args:
        state: 2D list representing the puzzle state
        size: Puzzle size (3, 4, or 5)
        output_path: Path where the image should be saved
    """
    generator = SlidingPuzzleGenerator()
    generator.render_puzzle(state, size, output_path)

