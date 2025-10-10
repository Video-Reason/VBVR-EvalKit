"""
Clean Maze Reasoning Tasks for Video Model Evaluation.

Two types of maze tasks:
1. KnowWhat Tasks: Algorithmic patterns with simple star/circle markers
2. Irregular Tasks: Professional mazes with green dot/flag icons

New Data Structure: prompt + first_image + final_image
"""

import json
import os
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# Import from KnowWhat submodule
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "submodules" / "KnowWhat"))
from core.maze_generator import (
    SHAPES, get_sample_mazes, init_random_start_end,
    WALL, PATH, POS, END
)

# Import from maze-dataset submodule
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "submodules" / "maze-dataset"))
from maze_dataset.maze.lattice_maze import LatticeMaze, TargetedLatticeMaze, SolvedMaze
from maze_dataset.plotting import MazePlot
from maze_dataset.generation import LatticeMazeGenerators

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


@dataclass
class MazeTaskPair:
    """
    New data structure for video model evaluation.
    
    Contains:
    - prompt: Instructions for the video model
    - first_image: The puzzle/problem to solve  
    - final_image: The correct answer/solution
    """
    id: str
    prompt: str                    # What to ask the video model to do
    first_image_path: str         # The puzzle image
    final_image_path: str         # The solution image
    task_category: str            # "KnowWhat" or "Irregular"
    maze_data: Dict[str, Any] = None  # Metadata
    difficulty: str = ""          # "easy", "medium", "hard"
    maze_size: Tuple[int, int] = (0, 0)
    start_pos: Tuple[int, int] = (0, 0)
    end_pos: Tuple[int, int] = (0, 0)
    solution_length: Optional[int] = None
    shape_type: str = ""          # For KnowWhat tasks only
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()


@dataclass
class MazeDataset:
    """Collection of MazeTaskPair instances."""
    name: str
    description: str
    pairs: List[MazeTaskPair]
    metadata: Dict[str, Any]
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(self, idx: int) -> MazeTaskPair:
        return self.pairs[idx]
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save dataset to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'MazeDataset':
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        pairs = [MazeTaskPair(**pair_data) for pair_data in data['pairs']]
        data['pairs'] = pairs
        
        return cls(**data)
    
    def filter_by_category(self, category: str) -> 'MazeDataset':
        """Filter by task category (KnowWhat or Irregular)."""
        filtered_pairs = [pair for pair in self.pairs if pair.task_category == category]
        return MazeDataset(
            name=f"{self.name}_filtered_{category}",
            description=f"Filtered version of {self.name} with only {category} tasks",
            pairs=filtered_pairs,
            metadata={**self.metadata, "filtered_by": category}
        )


class KnowWhatTaskGenerator:
    """
    Generator for KnowWhat algorithmic maze tasks.
    Uses simple star/circle markers, focuses on geometric patterns.
    """
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.maze_tasks_dir = self.data_root / "maze_tasks"
        self.generated_mazes_dir = self.data_root / "generated_mazes" 
        
        for dir_path in [self.maze_tasks_dir, self.generated_mazes_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.prompts = [
            "Move the blue star from start to the circle target.",
            "In this {shape}-pattern maze, move the star to the circle.",
            "Solve this {shape} maze by reaching the circle target (no path drawing).",
            "Start at the blue star and finish at the circle marker.",
        ]
    
    def knowwhat_to_lattice_maze(self, knowwhat_maze: np.ndarray) -> TargetedLatticeMaze:
        """Convert KnowWhat maze to maze-dataset format."""
        height, width = knowwhat_maze.shape
        
        start_pos = None
        end_pos = None
        
        for i in range(height):
            for j in range(width):
                if knowwhat_maze[i, j] == POS:  
                    start_pos = (i, j)
                elif knowwhat_maze[i, j] == END:  
                    end_pos = (i, j)
        
        if start_pos is None or end_pos is None:
            raise ValueError("Maze must have both start (POS=2) and end (END=3) positions")
        
        connection_list = np.zeros((2, height, width), dtype=bool)
        
        for i in range(height):
            for j in range(width):
                if knowwhat_maze[i, j] > 0:  
                    if i < height - 1 and knowwhat_maze[i + 1, j] > 0:
                        connection_list[0, i, j] = True
                    if j < width - 1 and knowwhat_maze[i, j + 1] > 0:
                        connection_list[1, i, j] = True
        
        targeted_maze = TargetedLatticeMaze(
            connection_list=connection_list,
            start_pos=start_pos,
            end_pos=end_pos
        )
        
        return targeted_maze
    
    def render_knowwhat_maze(self, maze: TargetedLatticeMaze, save_path: Path, show_solution: bool):
        """Render KnowWhat maze with markers only (no path)."""
        maze_plot = MazePlot(maze, unit_length=20)
        # Remove any automatic true path so no solution line is drawn
        maze_plot.true_path = None
        maze_plot.predicted_paths = []
        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        
        maze_plot.plot(fig_ax=(fig, ax), dpi=150, plain=True)
        
        # Add simple markers
        start_coord = maze_plot._rowcol_to_coord(maze.start_pos)
        end_coord = maze_plot._rowcol_to_coord(maze.end_pos)
        
        # Always show target circle at end (red for clarity)
        ax.plot(end_coord[0], end_coord[1], 'o', color='red', markersize=20, markeredgecolor='white', markeredgewidth=2)

        # Current position marker (blue star): at start for first frame, at end for final frame
        current = end_coord if show_solution else start_coord
        ax.plot(current[0], current[1], '*', color='blue', markersize=20, markeredgecolor='white', markeredgewidth=2)
        
        ax.set_aspect('equal')
        ax.axis('off')
        fig.patch.set_facecolor('white')
        fig.tight_layout(pad=0.1)
        
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def generate_knowwhat_pair(self, maze: np.ndarray, shape: str, size: Tuple[int, int], pair_id: str) -> MazeTaskPair:
        """Generate a KnowWhat task pair."""
        targeted_maze = self.knowwhat_to_lattice_maze(maze)
        
        # Generate first image (puzzle)
        first_path = self.generated_mazes_dir / f"{pair_id}_first.png"
        self.render_knowwhat_maze(targeted_maze, first_path, show_solution=False)
        
        # Generate final image (solution)
        final_path = self.generated_mazes_dir / f"{pair_id}_final.png"
        self.render_knowwhat_maze(targeted_maze, final_path, show_solution=True)
        
        prompt = random.choice(self.prompts).format(shape=shape)
        
        start_pos = tuple(map(int, np.argwhere(maze == POS)[0]))
        end_pos = tuple(map(int, np.argwhere(maze == END)[0]))
        
        return MazeTaskPair(
            id=pair_id,
            prompt=prompt,
            first_image_path=str(first_path),
            final_image_path=str(final_path),
            task_category="KnowWhat",
            maze_data={"maze_array": maze.tolist(), "generation_method": "knowwhat_algorithmic"},
            difficulty=self._determine_difficulty(size, shape),
            maze_size=size,
            start_pos=start_pos,
            end_pos=end_pos,
            shape_type=shape
        )
    
    def _determine_difficulty(self, size: Tuple[int, int], shape: str) -> str:
        """Determine difficulty for KnowWhat tasks."""
        size_factor = max(size)
        shape_complexity = {"square": 1, "cross": 2, "triangle": 2, "spiral": 3, "C": 3, "U": 3, "Z": 4, "N": 4}.get(shape, 2)
        
        if size_factor <= 5 and shape_complexity <= 2:
            return "easy"
        elif size_factor <= 7 and shape_complexity <= 3:
            return "medium"
        else:
            return "hard"


class IrregularTaskGenerator:
    """
    Generator for Irregular maze tasks using our own implementation of 
    a professional rendering approach (no external dependencies).
    """
    
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.maze_tasks_dir = self.data_root / "maze_tasks"
        self.generated_mazes_dir = self.data_root / "generated_mazes"
        
        for dir_path in [self.maze_tasks_dir, self.generated_mazes_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self._create_icons()
        
        self.prompts = [
            "Start at the green dot and end at the red flag (no path drawing).",
            "Move the green dot from start to the red flag destination.",
            "Show the green start marker at begin, then at the red flag in the final frame.",
            "Reach the red flag with the green dot; keep markers visible in both frames.",
        ]
    
    def _create_icons(self):
        """Create simple icons programmatically (no external files needed)."""
        # We'll create icons on-the-fly when needed instead of loading files
        pass
    
    def _create_green_circle_marker(self, ax, coord, grid_size):
        """Create green circle marker at given coordinate."""
        ax.plot(coord[0], coord[1], 'o', color='#22c55e', markersize=18, 
               markeredgecolor='white', markeredgewidth=2, zorder=10)
    
    def _create_red_flag_marker(self, ax, coord, grid_size):
        """Create red flag marker at given coordinate."""
        # Red triangular flag
        flag_size = grid_size * 0.4
        flag_x = coord[0] + flag_size * 0.3
        flag_y = coord[1]
        
        # Flag pole
        ax.plot([coord[0], coord[0]], [coord[1] - flag_size*0.5, coord[1] + flag_size*0.5], 
               color='#8b4513', linewidth=4, zorder=10)
        
        # Flag triangle
        triangle_x = [flag_x, flag_x + flag_size*0.6, flag_x]
        triangle_y = [flag_y + flag_size*0.3, flag_y, flag_y - flag_size*0.3]
        ax.fill(triangle_x, triangle_y, color='#ef4444', edgecolor='white', linewidth=1, zorder=11)
    
    def _create_trophy_marker(self, ax, coord, grid_size):
        """Create trophy marker at given coordinate."""
        trophy_size = grid_size * 0.3
        
        # Trophy base
        base_width = trophy_size * 0.8
        base_height = trophy_size * 0.2
        ax.add_patch(plt.Rectangle((coord[0] - base_width/2, coord[1] - trophy_size*0.4), 
                                 base_width, base_height, 
                                 facecolor='#f59e0b', edgecolor='#d97706', linewidth=1, zorder=10))
        
        # Trophy cup
        cup_width = trophy_size * 0.6
        cup_height = trophy_size * 0.5
        ax.add_patch(plt.Rectangle((coord[0] - cup_width/2, coord[1] - trophy_size*0.2), 
                                 cup_width, cup_height, 
                                 facecolor='#fbbf24', edgecolor='#d97706', linewidth=1, zorder=11))
    
    def generate_solved_maze(self, grid_n: int):
        """Generate a solved maze using Kruskal algorithm."""
        # Generate random maze and add start/end points
        lattice_maze = LatticeMazeGenerators.gen_kruskal(grid_shape=(grid_n, grid_n))
        
        # Add random start and end positions
        available_coords = []
        for i in range(grid_n):
            for j in range(grid_n):
                available_coords.append((i, j))
        
        start_pos = random.choice(available_coords)
        available_coords.remove(start_pos)
        end_pos = random.choice(available_coords)
        
        # Create targeted maze
        targeted_maze = TargetedLatticeMaze(
            connection_list=lattice_maze.connection_list,
            start_pos=start_pos,
            end_pos=end_pos
        )
        
        # Convert to solved maze
        solved_maze = SolvedMaze.from_targeted_lattice_maze(targeted_maze)
        
        return solved_maze
    
    def render_irregular_maze(self, solved_maze: SolvedMaze, save_path: Path, show_solution: bool):
        """Render Irregular maze showing markers only (no path)."""
        # Use their exact figure parameters
        fig_size_pixel = (832, 480)
        dpi = 100
        figsize = (fig_size_pixel[0]/dpi, fig_size_pixel[1]/dpi)
        
        maze_plot = MazePlot(solved_maze, unit_length=14)  # Their default
        # Remove any automatic true path so no solution line is drawn
        maze_plot.true_path = None
        maze_plot.predicted_paths = []
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        maze_plot.plot(fig_ax=(fig, ax), dpi=dpi, plain=True)
        
        # Calculate grid size for markers
        grid_size = maze_plot.unit_length * 0.8
        
        # Get coordinates
        start_coord = maze_plot._rowcol_to_coord(solved_maze.start_pos)
        end_coord = maze_plot._rowcol_to_coord(solved_maze.end_pos)
        
        # Always show the flag at the end position in both frames
        self._create_red_flag_marker(ax, end_coord, grid_size)

        # Current position: green dot at start in first frame, at end in final frame
        current = end_coord if show_solution else start_coord
        self._create_green_circle_marker(ax, current, grid_size)
        
        ax.set_aspect('equal')
        ax.axis('off')
        fig.patch.set_facecolor('white')
        fig.tight_layout(pad=0)
        
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
    
    def generate_irregular_pair(self, grid_n: int, pair_id: str) -> MazeTaskPair:
        """Generate an Irregular task pair using our implementation."""
        solved_maze = self.generate_solved_maze(grid_n)
        
        # Generate first image (puzzle)
        first_path = self.generated_mazes_dir / f"{pair_id}_first.png"
        self.render_irregular_maze(solved_maze, first_path, show_solution=False)
        
        # Generate final image (solution)
        final_path = self.generated_mazes_dir / f"{pair_id}_final.png"
        self.render_irregular_maze(solved_maze, final_path, show_solution=True)
        
        prompt = random.choice(self.prompts)
        
        return MazeTaskPair(
            id=pair_id,
            prompt=prompt,
            first_image_path=str(first_path),
            final_image_path=str(final_path),
            task_category="Irregular",
            maze_data={
                "generation_method": "kruskal_algorithm",
                "solution_length": len(solved_maze.solution)
            },
            difficulty=self._determine_difficulty(grid_n),
            maze_size=(grid_n, grid_n),
            start_pos=tuple(int(x) for x in solved_maze.start_pos),
            end_pos=tuple(int(x) for x in solved_maze.end_pos),
            solution_length=len(solved_maze.solution)
        )
    
    def _determine_difficulty(self, grid_n: int) -> str:
        """Determine difficulty for Irregular tasks."""
        if grid_n <= 4:
            return "easy"
        elif grid_n <= 6:
            return "medium"
        else:
            return "hard"
    
    def get_total_possible_mazes(self) -> int:
        """Estimated total possible with our approach (virtually unlimited)."""
        return 10000  # Conservative estimate - can generate many more


def create_knowwhat_dataset(num_samples: int = 20) -> MazeDataset:
    """Create KnowWhat dataset with simple star/circle markers."""
    # Apply robustness patch for KnowWhat generation
    try:
        from .maze_patch import patch_knowwhat_generation
        patch_knowwhat_generation()
    except ImportError:
        print("⚠️  Maze patch not available, using original generation")
    
    generator = KnowWhatTaskGenerator()
    pairs = []
    
    print(f"🧩 Generating {num_samples} KnowWhat algorithmic maze tasks...")
    
    for i in range(num_samples):
        try:
            shape = random.choice(SHAPES)
            size = random.choice([(5, 5), (7, 7), (9, 9)])
            
            # Generate single maze using KnowWhat
            mazes = get_sample_mazes(size, shape, k=1)
            maze = mazes[0]
            
            pair_id = f"knowwhat_{i:04d}"
            pair = generator.generate_knowwhat_pair(maze, shape, size, pair_id)
            pairs.append(pair)
            
        except Exception as e:
            print(f"Error generating KnowWhat maze {i}: {e}")
            continue
    
    dataset = MazeDataset(
        name="knowwhat_tasks",
        description=f"KnowWhat algorithmic maze tasks with star/circle markers ({len(pairs)} pairs)",
        pairs=pairs,
        metadata={"task_category": "KnowWhat", "marker_style": "star_circle"}
    )
    
    dataset.save(generator.maze_tasks_dir / "knowwhat_tasks.json")
    print(f"✓ Generated {len(pairs)} KnowWhat tasks")
    return dataset


def create_irregular_dataset(num_samples: int = 20) -> MazeDataset:
    """Create Irregular dataset using our own implementation."""
    generator = IrregularTaskGenerator()
    pairs = []
    
    total_possible = generator.get_total_possible_mazes()
    print(f"🎯 Generating {num_samples} Irregular maze tasks...")
    print(f"   (Estimated total possible: {total_possible:,} mazes)")
    
    for i in range(num_samples):
        try:
            # Use random grid sizes
            grid_n = random.choice([3, 4, 5, 6, 7, 8])
            
            pair_id = f"irregular_{i:04d}"
            pair = generator.generate_irregular_pair(grid_n, pair_id)
            pairs.append(pair)
            
        except Exception as e:
            print(f"Error generating Irregular maze {i}: {e}")
            continue
    
    dataset = MazeDataset(
        name="irregular_tasks", 
        description=f"Irregular maze tasks with professional green dot/flag rendering ({len(pairs)} pairs)",
        pairs=pairs,
        metadata={
            "task_category": "Irregular", 
            "total_possible": total_possible,
            "marker_style": "green_dot_flag",
            "generation_method": "kruskal_algorithm"
        }
    )
    
    dataset.save(generator.maze_tasks_dir / "irregular_tasks.json")
    print(f"✓ Generated {len(pairs)} Irregular tasks")
    return dataset


def create_combined_dataset(knowwhat_samples: int = 15, irregular_samples: int = 15) -> MazeDataset:
    """Create combined dataset with both task types."""
    print(f"🚀 Creating Combined Maze Dataset")
    print(f"   KnowWhat samples: {knowwhat_samples}")
    print(f"   Irregular samples: {irregular_samples}")
    print("=" * 70)
    
    # Generate both types
    knowwhat_dataset = create_knowwhat_dataset(knowwhat_samples)
    irregular_dataset = create_irregular_dataset(irregular_samples)
    
    # Combine
    all_pairs = knowwhat_dataset.pairs + irregular_dataset.pairs
    
    combined_dataset = MazeDataset(
        name="combined_maze_tasks",
        description=f"Combined dataset with {len(knowwhat_dataset)} KnowWhat + {len(irregular_dataset)} Irregular tasks",
        pairs=all_pairs,
        metadata={
            "knowwhat_count": len(knowwhat_dataset),
            "irregular_count": len(irregular_dataset),
            "irregular_total_possible": irregular_dataset.metadata.get("total_possible", 0)
        }
    )
    
    generator = KnowWhatTaskGenerator()
    combined_dataset.save(generator.maze_tasks_dir / "combined_maze_tasks.json")
    
    print(f"\n🎉 Combined dataset created with {len(combined_dataset)} total pairs!")
    print(f"   KnowWhat tasks: {len(knowwhat_dataset)} (algorithmic patterns)")
    print(f"   Irregular tasks: {len(irregular_dataset)} (professional rendering)")
    
    return combined_dataset


if __name__ == "__main__":
    # Create sample combined dataset
    dataset = create_combined_dataset(knowwhat_samples=5, irregular_samples=5)
