# Maze Reasoning Task Documentation

## Overview

The Maze Reasoning Task evaluates video generation models' ability to demonstrate spatial reasoning and problem-solving by generating videos that show navigation from a start position to an end position in a maze. This task is part of VMEvalKit's reasoning evaluation suite.

## Task Types

### 1. KnowWhat Tasks
- **Description**: Algorithmic mazes with geometric patterns (squares, spirals, triangles, etc.)
- **Visual Style**: Simple star and circle markers
- **Markers**:
  - 🔵 Blue star: Current position (moves from start to end)
  - 🔴 Red circle: Target position (remains fixed)
- **Patterns**: Square, Cross, Triangle, Spiral, C, U, Z, N shapes
- **Sizes**: 5x5, 7x7, 9x9 grids

### 2. Irregular Tasks
- **Description**: Professional-style mazes with complex paths
- **Visual Style**: Green dot and red flag markers
- **Markers**:
  - 🟢 Green dot: Current position (moves from start to end)
  - 🚩 Red flag: Target position (remains fixed)
- **Generation**: Kruskal's algorithm for maze generation
- **Sizes**: 3x3 to 8x8 grids

## Data Structure

### MazeTaskPair
Each task consists of a pair of images and a text prompt:

```python
@dataclass
class MazeTaskPair:
    id: str                    # Unique identifier (e.g., "knowwhat_0001")
    prompt: str                # Instructions for the video model
    first_image_path: str      # Path to the puzzle image (start state)
    final_image_path: str      # Path to the solution image (end state)
    task_category: str         # "KnowWhat" or "Irregular"
    maze_data: Dict[str, Any]  # Metadata including maze array
    difficulty: str            # "easy", "medium", or "hard"
    maze_size: Tuple[int, int] # Grid dimensions
    start_pos: Tuple[int, int] # Starting coordinates
    end_pos: Tuple[int, int]   # Ending coordinates
    solution_length: Optional[int]  # Path length (for Irregular)
    shape_type: str            # Shape pattern (for KnowWhat)
```

### MazeDataset
A collection of maze task pairs with metadata:

```python
@dataclass
class MazeDataset:
    name: str                  # Dataset identifier
    description: str           # Human-readable description
    pairs: List[MazeTaskPair]  # List of task pairs
    metadata: Dict[str, Any]   # Additional information
```

## Visual Representation

### First Frame (Puzzle)
- Shows the maze with markers at their starting positions
- KnowWhat: Blue star at start, red circle at target
- Irregular: Green dot at start, red flag at target

### Final Frame (Solution)
- Shows the completed state
- KnowWhat: Blue star moved to target position (overlapping red circle)
- Irregular: Green dot moved to flag position

### Important: No Path Drawing
The task explicitly avoids drawing solution paths. Only the current position marker moves between frames. This tests the model's ability to understand and demonstrate navigation without explicit path visualization.

## Prompts

### KnowWhat Prompts
- "Move the blue star from start to the circle target."
- "In this {shape}-pattern maze, move the star to the circle."
- "Solve this {shape} maze by reaching the circle target (no path drawing)."
- "Start at the blue star and finish at the circle marker."

### Irregular Prompts
- "Start at the green dot and end at the red flag (no path drawing)."
- "Move the green dot from start to the red flag destination."
- "Show the green start marker at begin, then at the red flag in the final frame."
- "Reach the red flag with the green dot; keep markers visible in both frames."

## Usage

### Generating Datasets

```python
from vmevalkit.tasks.maze_task import (
    create_knowwhat_dataset,
    create_irregular_dataset,
    create_combined_dataset
)

# Generate individual datasets
knowwhat_dataset = create_knowwhat_dataset(num_samples=20)
irregular_dataset = create_irregular_dataset(num_samples=20)

# Generate combined dataset
combined_dataset = create_combined_dataset(
    knowwhat_samples=15,
    irregular_samples=15
)
```

### Loading Existing Datasets

```python
from vmevalkit.tasks.maze_task import MazeDataset

# Load from JSON files
dataset = MazeDataset.load("data/questions/maze_tasks/combined_maze_tasks.json")

# Filter by category
knowwhat_only = dataset.filter_by_category("KnowWhat")
irregular_only = dataset.filter_by_category("Irregular")

# Access individual pairs
for pair in dataset.pairs:
    print(f"Task: {pair.id}")
    print(f"Prompt: {pair.prompt}")
    print(f"First image: {pair.first_image_path}")
    print(f"Final image: {pair.final_image_path}")
```

## File Structure

```
data/
├── maze_tasks/
│   ├── knowwhat_tasks.json      # KnowWhat dataset
│   ├── irregular_tasks.json     # Irregular dataset
│   └── combined_maze_tasks.json # Combined dataset
└── generated_mazes/
    ├── knowwhat_0000_first.png  # KnowWhat puzzle images
    ├── knowwhat_0000_final.png  # KnowWhat solution images
    ├── irregular_0000_first.png # Irregular puzzle images
    └── irregular_0000_final.png # Irregular solution images
```

## Evaluation Criteria

Video models are evaluated on their ability to:

1. **Spatial Understanding**: Recognize the maze structure and valid paths
2. **Marker Recognition**: Identify start (star/green dot) and target (circle/flag) positions
3. **Movement Logic**: Generate smooth, logical movement from start to end
4. **Constraint Adherence**: Respect maze walls and avoid invalid paths
5. **Visual Consistency**: Maintain marker appearance and maze structure throughout

## Difficulty Levels

### KnowWhat Difficulty
- **Easy**: Small grids (5x5) with simple shapes (square, cross)
- **Medium**: Medium grids (7x7) with moderate shapes (triangle, spiral)
- **Hard**: Large grids (9x9) with complex shapes (Z, N patterns)

### Irregular Difficulty
- **Easy**: Small grids (3x3, 4x4)
- **Medium**: Medium grids (5x5, 6x6)
- **Hard**: Large grids (7x7, 8x8)

## Implementation Details

### KnowWhat Maze Generation

KnowWhat mazes use pre-generated maze files from the KnowWhat submodule to ensure reliability:

- **Source**: `/submodules/KnowWhat/data/experiment_mazes/`
- **Available mazes**: 
  - 5x5: 5 mazes per shape (30 total)
  - 7x7: 30 mazes per shape (180 total)
  - Total: 210 pre-validated mazes
- **Benefits**:
  - Guaranteed validity (all mazes have solutions)
  - Consistency with original KnowWhat experiments
  - No generation failures
  - Better performance (no on-the-fly generation)

The implementation renders mazes directly from numpy arrays using matplotlib, avoiding complex format conversions.

### Irregular Maze Generation

Irregular mazes are generated dynamically using:
- **Algorithm**: Kruskal's algorithm via maze-dataset library
- **Rendering**: Custom implementation with matplotlib
- **Markers**: Programmatically drawn (no external icon files needed)

## Notes

- All generated images use white backgrounds with black maze walls
- Markers are designed to be visually distinct and easily recognizable
- The task focuses on endpoint reasoning rather than path-following
- KnowWhat mazes are pre-validated, ensuring 100% generation success rate
