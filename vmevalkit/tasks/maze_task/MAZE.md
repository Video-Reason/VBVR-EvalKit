# Maze Reasoning Tasks

Video models need to demonstrate path-finding and spatial reasoning capabilities by navigating mazes.

## Task Overview

Professional-style mazes with green dot and red flag markers.

**Key Features:**
- High-quality maze rendering
- Green dot as current position
- Red flag as target
- Various difficulty levels based on maze size

## Data Structure

Each maze task follows this structure:

```python
@dataclass
class MazeTaskPair:
    id: str                    # Unique identifier (e.g., "maze_0001")
    prompt: str                # Instructions for the video model
    first_image_path: str     # Path to the puzzle image
    final_image_path: str     # Path to the solution image
    task_category: str         # "Maze"
    maze_data: Dict           # Metadata about the maze
    difficulty: str           # "easy", "medium", or "hard"
    maze_size: Tuple[int, int]  # Grid dimensions
    start_pos: Tuple[int, int]   # Starting position
    end_pos: Tuple[int, int]     # Target position
    solution_length: int         # Number of steps in solution
```

## Visual Format

**First Frame (Puzzle):**
- Green dot at start position
- Red flag at target position

**Final Frame (Solution):**
- Green dot moved to target position (overlapping red flag)
- Red flag remains visible

## Prompts

Example prompts for mazes:
- "Navigate the green dot through the maze corridors (avoiding walls) from its starting position to reach the red flag."
- "Guide the green dot through the open paths to the red flag destination. Walls block movement, corridors allow passage."
- "Move the green dot from its starting position through the maze paths to the red flag. Navigate only through corridors."

## Dataset Creation

### Quick Start

```python
from vmevalkit.tasks.maze_task import create_dataset

# Create maze dataset (standard interface)
maze_dataset = create_dataset(num_samples=30)
```

### Filtering

```python
# Filter by difficulty
easy_mazes = [p for p in dataset['pairs'] if p['difficulty'] == "easy"]

# Filter by maze size
small_mazes = [p for p in dataset['pairs'] if p['maze_size'][0] <= 5]
```

## File Organization

```
data/questions/
├── maze_task/
│   ├── maze_tasks.json          # Complete dataset
│   └── temp/                    # Temporary generation files
└── maze_images/
    ├── maze_0000_first.png      # Maze puzzle images
    ├── maze_0000_final.png      # Maze solution images
    └── ...
```

## Technical Details

### Maze Generation

Mazes use the Kruskal algorithm for generation:

- **Algorithm**: Kruskal's minimum spanning tree
- **Grid sizes**: 3x3 (simplified for easier evaluation)
- **Rendering**: Professional maze visualization with matplotlib
- **Markers**: 
  - Green circle for current position
  - Red flag for target
  - Professional styling with borders and shadows

### Difficulty Levels

Based on maze grid size:
- **Easy**: 3x3 grids (current simplified implementation)
- **Medium**: Reserved for future expansion
- **Hard**: Reserved for future expansion

## Best Practices

1. **Balanced Datasets**: Create datasets with varied difficulty levels
2. **Validation**: Ensure all mazes have valid solutions
3. **Consistency**: Use consistent visual styling across all mazes
4. **Documentation**: Include metadata for analysis and filtering

## Success Metrics

- **Path validity**: Does the solution follow open corridors?
- **Optimal path**: Is the solution the shortest possible path?
- **Visual consistency**: Does the green dot properly reach the flag?

## Notes

- Mazes are generated dynamically with guaranteed solvability
- Each maze has exactly one valid solution path
- Visual rendering is optimized for clarity and consistency