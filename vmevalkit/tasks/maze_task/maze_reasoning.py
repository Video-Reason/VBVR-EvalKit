"""
Clean Maze Reasoning Tasks for Video Model Evaluation.

Professional mazes with green dot/flag icons

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

# Import from maze-dataset submodule
import sys
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "submodules" / "maze-dataset"))
from maze_dataset.maze.lattice_maze import LatticeMaze, TargetedLatticeMaze, SolvedMaze
from maze_dataset.plotting import MazePlot
from maze_dataset.generation import LatticeMazeGenerators

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

# Import prompts from centralized location
from .PROMPTS import PROMPTS


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
    task_category: str            # "Maze"
    maze_data: Dict[str, Any] = None  # Metadata
    difficulty: str = ""          # "easy", "medium", "hard"
    maze_size: Tuple[int, int] = (0, 0)
    start_pos: Tuple[int, int] = (0, 0)
    end_pos: Tuple[int, int] = (0, 0)
    solution_length: Optional[int] = None
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
        
        # Convert dictionaries back to MazeTaskPair objects
        pairs = []
        for pair_data in data['pairs']:
            pairs.append(MazeTaskPair(**pair_data))
        
        data['pairs'] = pairs
        return cls(**data)
    
    def filter_by_category(self, category: str) -> 'MazeDataset':
        """Filter by task category."""
        filtered_pairs = [p for p in self.pairs if p.task_category == category]
        return MazeDataset(
            name=f"{self.name}_filtered_{category}",
            description=f"Filtered dataset: {category} tasks only",
            pairs=filtered_pairs,
            metadata={**self.metadata, "filter": category}
        )


class MazeTaskGenerator:
    """
    Generator for maze tasks using our own implementation of 
    a professional rendering approach (no external dependencies).
    """
    
    def __init__(self, data_root: str = "data/questions"):
        self.data_root = Path(data_root)
        # Don't create intermediate folders anymore
        import tempfile
        self.temp_dir = tempfile.mkdtemp()
        
        self._create_icons()
        
        # Use standardized prompt from PROMPTS list
        self.prompt = PROMPTS[0]
    
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
    
    def render_maze(self, solved_maze: SolvedMaze, save_path: Path, show_solution: bool):
        """Render maze showing markers only (no path)."""
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
    
    def generate_maze_pair(self, grid_n: int, pair_id: str) -> MazeTaskPair:
        """Generate a maze task pair using our implementation."""
        solved_maze = self.generate_solved_maze(grid_n)
        
        # Generate first image (puzzle) in temp directory
        first_path = Path(self.temp_dir) / f"{pair_id}_first.png"
        self.render_maze(solved_maze, first_path, show_solution=False)
        
        # Generate final image (solution) in temp directory
        final_path = Path(self.temp_dir) / f"{pair_id}_final.png"
        self.render_maze(solved_maze, final_path, show_solution=True)
        
        prompt = PROMPTS[0]  # Use standardized prompt
        
        return MazeTaskPair(
            id=pair_id,
            prompt=prompt,
            first_image_path=str(first_path),
            final_image_path=str(final_path),
            task_category="Maze",
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
        """Determine difficulty for maze tasks."""
        if grid_n <= 4:
            return "easy"
        elif grid_n <= 6:
            return "medium"
        else:
            return "hard"
    
    def get_total_possible_mazes(self) -> int:
        """Estimated total possible with our approach (virtually unlimited)."""
        return 10000  # Conservative estimate - can generate many more


# Removed create_maze_dataset_direct - merged into create_dataset below


def create_dataset(num_samples: int = 30) -> Dict[str, Any]:
    """Create maze dataset - main entry point matching other domains."""
    print(f"🧩 Creating Maze Dataset")
    print(f"   Total samples: {num_samples}")
    
    start_time = datetime.now()
    
    # Generate maze tasks directly (merged from create_maze_dataset_direct)
    generator = MazeTaskGenerator()
    pairs = []
    
    total_possible = generator.get_total_possible_mazes()
    print(f"🎯 Generating {num_samples} Simplified maze tasks (3x3 grids only)...")
    print(f"   (Estimated total possible: {total_possible:,} mazes)")
    
    for i in range(num_samples):
        try:
            # Use simplified grid sizes (SIMPLIFIED: only 3x3 grids)
            grid_n = 3
            
            pair_id = f"maze_{i:04d}"
            pair = generator.generate_maze_pair(grid_n, pair_id)
            pairs.append(pair)
            
        except Exception as e:
            print(f"Error generating maze {i}: {e}")
            continue
    
    # Create dataset object
    dataset = MazeDataset(
        name="maze_tasks", 
        description=f"Simplified maze tasks (3x3 grids only) with green dot/flag rendering ({len(pairs)} pairs)",
        pairs=pairs,
        metadata={
            "task_category": "Maze", 
            "total_possible": total_possible,
            "marker_style": "green_dot_flag",
            "generation_method": "kruskal_algorithm",
            "grid_size": "3x3_only"
        }
    )
    
    print(f"✓ Generated {len(pairs)} maze tasks")
    
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print(f"\n✅ Dataset creation complete!")
    print(f"   Total tasks: {len(dataset)}")
    print(f"   Time elapsed: {elapsed:.1f}s")
    
    # Convert to dictionary format like other domains
    return {
        "name": dataset.name,
        "description": dataset.description,
        "pairs": [asdict(pair) for pair in dataset.pairs],
        "metadata": dataset.metadata,
        "created_at": dataset.created_at
    }


# Dataset creation should only be done via vmevalkit/runner/create_dataset.py
# This module only provides the create_dataset() function as an API