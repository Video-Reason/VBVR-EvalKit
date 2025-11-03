# Maze Reasoning Tasks

Video models need to demonstrate path-finding and spatial reasoning capabilities by navigating mazes.

## Task Overview

The maze task challenges video models to demonstrate spatial reasoning by moving a green dot through a maze from its starting position to reach a red flag at the target location. This task evaluates:

- **Spatial Navigation**: Understanding maze topology and valid paths
- **Path Planning**: Finding routes through complex corridors
- **Visual Reasoning**: Interpreting walls, paths, and markers
- **Goal-Directed Movement**: Reaching the target efficiently

**Key Features:**
- High-quality maze rendering using matplotlib
- Green dot as current position marker
- Red flag as target destination
- Professional visualization with shadows and borders
- Guaranteed solvable mazes using Kruskal's algorithm
- Automatic path validation and solution generation

## Architecture

### Core Components

1. **MazeTaskGenerator**: Main class for generating maze tasks
   - Handles maze generation using external maze-dataset submodule
   - Renders professional-quality visualizations
   - Creates marker icons programmatically
   - Manages temporary file storage

2. **MazeTaskPair**: Data structure for maze problems
   - Stores puzzle and solution images
   - Contains metadata for evaluation
   - Tracks difficulty and solution metrics

3. **MazeDataset**: Collection manager for task pairs
   - Provides filtering and organization
   - Supports JSON serialization
   - Enables batch operations

## Data Structure

### MazeTaskPair

Each maze task follows this comprehensive structure:

```python
@dataclass
class MazeTaskPair:
    id: str                      # Unique identifier (e.g., "maze_0001")
    prompt: str                  # Instructions for the video model
    first_image_path: str        # Path to the puzzle image (start state)
    final_image_path: str        # Path to the solution image (end state)
    task_category: str           # Always "Maze" for this task type
    maze_data: Dict[str, Any]    # Additional metadata:
                                # - generation_method: "kruskal_algorithm"
                                # - solution_length: Number of steps in optimal path
    difficulty: str              # "easy", "medium", or "hard" based on grid size
    maze_size: Tuple[int, int]   # Grid dimensions (rows, columns)
    start_pos: Tuple[int, int]   # Starting cell coordinates (row, col)
    end_pos: Tuple[int, int]     # Target cell coordinates (row, col)
    solution_length: int         # Number of steps in the solution path
    created_at: str              # ISO timestamp of creation
```

### MazeDataset

Container for multiple maze tasks with metadata:

```python
@dataclass
class MazeDataset:
    name: str                    # Dataset identifier
    description: str             # Human-readable description
    pairs: List[MazeTaskPair]    # Collection of task pairs
    metadata: Dict[str, Any]     # Dataset-level metadata:
                                # - task_category: "Maze"
                                # - total_possible: Estimated unique mazes
                                # - marker_style: "green_dot_flag"
                                # - generation_method: "kruskal_algorithm"
                                # - grid_size: "3x3_only" (current limitation)
    created_at: str              # ISO timestamp
```

## Visual Format

### Frame Specifications

**Image Properties:**
- Resolution: 832×480 pixels
- DPI: 100
- Format: PNG with white background
- Aspect ratio: Preserved with equal scaling

**First Frame (Puzzle State):**
- Green dot (●) at starting position
- Red flag (🚩) at target position
- White cells represent open paths
- Black lines represent walls
- Professional borders and shadows

**Final Frame (Solution State):**
- Green dot moved to target position (overlapping red flag)
- Red flag remains visible underneath
- Path taken is implicit (not drawn)
- Represents successful navigation

### Marker Rendering

Markers are created programmatically using matplotlib:

```python
# Green dot marker
ax.plot(coord[0], coord[1], 'o', 
        color='#22c55e',           # Tailwind green-500
        markersize=18,
        markeredgecolor='white',
        markeredgewidth=2,
        zorder=10)

# Red flag marker (composite design)
# Flag pole
ax.plot([x, x], [y-size*0.5, y+size*0.5], 
        color='#8b4513',           # Brown pole
        linewidth=4, zorder=10)

# Flag triangle
ax.fill(triangle_x, triangle_y,
        color='#ef4444',           # Tailwind red-500
        edgecolor='white',
        linewidth=1, zorder=11)
```

## Maze Generation

### Algorithm Details

The system uses **Kruskal's Minimum Spanning Tree Algorithm** for maze generation:

1. **Initialization**: Create a grid of disconnected cells
2. **Edge Creation**: List all possible walls between cells
3. **Random Selection**: Randomly order the walls
4. **Union-Find**: Connect cells if they're in different sets
5. **Completion**: Continue until all cells are connected

This ensures:
- Every maze has exactly one solution
- All cells are reachable
- No loops or cycles exist
- Uniform randomness in structure

### Implementation

```python
def generate_solved_maze(self, grid_n: int):
    """Generate a solved maze using Kruskal algorithm."""
    # Generate maze structure
    lattice_maze = LatticeMazeGenerators.gen_kruskal(
        grid_shape=(grid_n, grid_n)
    )
    
    # Select random start and end positions
    available_coords = [(i, j) for i in range(grid_n) 
                       for j in range(grid_n)]
    start_pos = random.choice(available_coords)
    available_coords.remove(start_pos)
    end_pos = random.choice(available_coords)
    
    # Create targeted maze with solution
    targeted_maze = TargetedLatticeMaze(
        connection_list=lattice_maze.connection_list,
        start_pos=start_pos,
        end_pos=end_pos
    )
    
    # Convert to solved maze (finds optimal path)
    solved_maze = SolvedMaze.from_targeted_lattice_maze(targeted_maze)
    return solved_maze
```

## Prompts

### Standardized Prompt

The maze task uses a carefully designed standardized prompt:

```
"Move the green dot from its starting position through the maze paths to the red flag. Navigate only through open spaces (white)."
```

This prompt clearly specifies:
- **Subject**: Green dot (the agent to be moved)
- **Action**: Move through maze paths
- **Goal**: Reach the red flag
- **Constraints**: Navigate only through white (open) spaces
- **Implicit**: Avoid black walls, find valid path

### Prompt Management

Prompts are centralized in `PROMPTS.py` for easy modification:

```python
# vmevalkit/tasks/maze_task/PROMPTS.py
PROMPTS = [
    "Move the green dot from its starting position through the maze paths to the red flag. Navigate only through open spaces (white).",
    # Additional prompt variations can be added here for experiments
]

DEFAULT_PROMPT_INDEX = 0  # Use first prompt by default
```

## Dataset Creation

### Quick Start

```python
from vmevalkit.tasks.maze_task import create_dataset

# Create standard dataset
maze_dataset = create_dataset(num_samples=30)

# Access the generated pairs
for pair in maze_dataset['pairs']:
    print(f"ID: {pair['id']}")
    print(f"Difficulty: {pair['difficulty']}")
    print(f"Solution length: {pair['solution_length']}")
```

### Advanced Usage

```python
from vmevalkit.tasks.maze_task import MazeTaskGenerator, MazeDataset

# Create custom generator
generator = MazeTaskGenerator(data_root="custom/data/path")

# Generate individual maze pairs
maze_pair = generator.generate_maze_pair(
    grid_n=3,           # Grid size (currently limited to 3)
    pair_id="custom_001"
)

# Create custom dataset
pairs = []
for i in range(50):
    pair = generator.generate_maze_pair(3, f"maze_{i:04d}")
    pairs.append(pair)

dataset = MazeDataset(
    name="custom_maze_dataset",
    description="Custom maze collection",
    pairs=pairs,
    metadata={
        "task_category": "Maze",
        "experiment": "difficulty_scaling"
    }
)

# Save dataset
dataset.save("path/to/dataset.json")

# Load existing dataset
loaded_dataset = MazeDataset.load("path/to/dataset.json")
```

### Filtering and Analysis

```python
# Filter by difficulty
easy_mazes = [p for p in dataset['pairs'] if p['difficulty'] == "easy"]
medium_mazes = [p for p in dataset['pairs'] if p['difficulty'] == "medium"]

# Filter by solution length
short_paths = [p for p in dataset['pairs'] if p['solution_length'] <= 4]
long_paths = [p for p in dataset['pairs'] if p['solution_length'] >= 6]

# Filter by maze size
small_mazes = [p for p in dataset['pairs'] if p['maze_size'][0] <= 3]

# Analyze dataset statistics
import numpy as np
solution_lengths = [p['solution_length'] for p in dataset['pairs']]
print(f"Average solution length: {np.mean(solution_lengths):.2f}")
print(f"Min/Max solution length: {min(solution_lengths)}/{max(solution_lengths)}")

# Group by start/end positions
from collections import defaultdict
position_groups = defaultdict(list)
for pair in dataset['pairs']:
    key = (pair['start_pos'], pair['end_pos'])
    position_groups[key].append(pair)
```

## File Organization

### Directory Structure

```
data/
├── questions/                      # Generated datasets
│   ├── maze_task/
│   │   ├── maze_tasks.json        # Main dataset file
│   │   └── maze_[ID]/             # Individual task folders
│   │       ├── first_frame.png    # Puzzle image
│   │       ├── final_frame.png    # Solution image
│   │       └── metadata.json      # Task metadata
│   └── temp/                      # Temporary generation files
│
├── outputs/                        # Model inference results
│   └── [experiment_name]/
│       └── [model_name]/
│           └── maze_task/
│               └── maze_[ID]/
│                   ├── video/     # Generated videos
│                   ├── question/  # Input images
│                   └── metadata.json
│
└── evaluations/                    # Evaluation results
    └── [experiment_name]/
        └── [model_name]/
            └── maze_task/
                └── maze_[ID]/
                    ├── gpt4o-eval.json
                    └── human-eval.json
```

## Technical Implementation

### Dependencies

The maze task relies on:

1. **maze-dataset submodule**: Core maze generation algorithms
   - `LatticeMaze`: Basic maze structure
   - `TargetedLatticeMaze`: Maze with start/end points
   - `SolvedMaze`: Maze with computed solution path
   - `LatticeMazeGenerators`: Generation algorithms (Kruskal)
   - `MazePlot`: Visualization utilities

2. **Python libraries**:
   ```python
   matplotlib   # Visualization and rendering
   numpy        # Array operations
   cv2          # Image processing (optional)
   Pillow       # Image manipulation (optional)
   ```

### Maze Rendering Pipeline

```python
def render_maze(self, solved_maze: SolvedMaze, save_path: Path, show_solution: bool):
    """Render maze with markers."""
    # Configure figure dimensions
    fig_size_pixel = (832, 480)
    dpi = 100
    figsize = (fig_size_pixel[0]/dpi, fig_size_pixel[1]/dpi)
    
    # Create maze plot
    maze_plot = MazePlot(solved_maze, unit_length=14)
    maze_plot.true_path = None  # Don't draw solution path
    maze_plot.predicted_paths = []
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    maze_plot.plot(fig_ax=(fig, ax), dpi=dpi, plain=True)
    
    # Add markers
    grid_size = maze_plot.unit_length * 0.8
    start_coord = maze_plot._rowcol_to_coord(solved_maze.start_pos)
    end_coord = maze_plot._rowcol_to_coord(solved_maze.end_pos)
    
    # Always show flag at end
    self._create_red_flag_marker(ax, end_coord, grid_size)
    
    # Move green dot based on frame
    current = end_coord if show_solution else start_coord
    self._create_green_circle_marker(ax, current, grid_size)
    
    # Configure and save
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor('white')
    fig.tight_layout(pad=0)
    fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
```

### Difficulty Scaling

Current implementation uses simplified 3×3 grids, but the architecture supports expansion:

```python
def _determine_difficulty(self, grid_n: int) -> str:
    """Determine difficulty based on maze complexity."""
    if grid_n <= 4:
        return "easy"     # 3×3, 4×4 grids
    elif grid_n <= 6:
        return "medium"   # 5×5, 6×6 grids
    else:
        return "hard"     # 7×7+ grids
    
    # Future: Could also consider:
    # - Solution path length
    # - Number of dead ends
    # - Branching factor
    # - Distance from optimal
```

## Integration with VMEvalKit

### Runner Integration

The maze task integrates with the VMEvalKit runner system through:

```python
# vmevalkit/runner/create_dataset.py
DOMAIN_REGISTRY = {
    'maze': {
        'emoji': '🌀',
        'name': 'Maze',
        'description': 'Spatial reasoning and navigation planning',
        'module': 'vmevalkit.tasks.maze_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    # ... other tasks
}
```

### Model Inference

Video models process maze tasks through:

1. **Input**: First frame (puzzle) + text prompt
2. **Processing**: Model generates video showing navigation
3. **Output**: Video ending at solution state
4. **Storage**: Structured output in experiment folders

### Evaluation

Maze solutions are evaluated using:

1. **GPT-4O Evaluation** (`vmevalkit.eval.gpt4o_eval`):
   - Compares final frame with ground truth
   - Checks if green dot reached red flag
   - Validates path completeness
   
2. **Human Evaluation** (`vmevalkit.eval.human_eval`):
   - Manual verification of solution correctness
   - Path validity assessment
   - Visual quality rating

## Success Metrics

### Primary Metrics

1. **Solution Correctness** (1-5 scale):
   - 5: Perfect - green dot exactly on red flag
   - 4: Mostly correct - very close to target
   - 3: Partial - significant progress made
   - 2: Minimal - little progress toward goal
   - 1: Failed - no understanding of task

2. **Path Validity**:
   - Does the path stay within white corridors?
   - Are walls respected (no passing through)?
   - Is the movement continuous?

3. **Path Efficiency**:
   - Is the path optimal (shortest)?
   - Unnecessary detours or backtracking?
   - Direct vs. meandering approach

### Secondary Metrics

- **Visual Consistency**: Smooth marker movement
- **Frame Coherence**: Logical progression between frames
- **Task Understanding**: Correct interpretation of start/goal

## Best Practices

### Dataset Creation

1. **Balance Complexity**: Even with 3×3 grids, vary start/end positions
2. **Ensure Diversity**: Generate sufficient samples for statistical validity
3. **Validate Generation**: Check all mazes have valid solutions
4. **Document Metadata**: Include all relevant information for analysis

### Visualization

1. **Consistent Styling**: Maintain uniform appearance across all mazes
2. **Clear Markers**: Ensure markers are easily distinguishable
3. **High Contrast**: Use strong color differences for clarity
4. **Professional Appearance**: Clean borders, shadows, proper spacing

### Evaluation

1. **Multiple Evaluators**: Use both automatic and human evaluation
2. **Clear Criteria**: Define success metrics precisely
3. **Error Analysis**: Track common failure modes
4. **Statistical Rigor**: Sufficient samples for significance

## Performance Considerations

### Generation Performance

- Maze generation: ~0.1s per maze (3×3 grid)
- Image rendering: ~0.2s per image pair
- Dataset creation: ~10s for 30 samples
- Memory usage: Minimal (~50MB for 100 mazes)

### Optimization Strategies

1. **Batch Generation**: Create multiple mazes in parallel
2. **Caching**: Reuse maze structures when possible
3. **Lazy Loading**: Load images only when needed
4. **Cleanup**: Remove temporary files after generation

```python
# Example: Batch generation with cleanup
import tempfile
import shutil

with tempfile.TemporaryDirectory() as temp_dir:
    generator = MazeTaskGenerator()
    generator.temp_dir = temp_dir
    
    # Generate mazes
    pairs = [generator.generate_maze_pair(3, f"maze_{i:04d}") 
             for i in range(100)]
    
    # Process pairs...
    # Temp files automatically cleaned up
```

## Troubleshooting

### Common Issues

1. **Import Error for maze-dataset**:
   ```python
   # Ensure submodule is initialized
   git submodule update --init --recursive
   ```

2. **Missing Dependencies**:
   ```bash
   pip install matplotlib numpy
   ```

3. **Rendering Issues**:
   - Check matplotlib backend configuration
   - Verify DPI settings match system capabilities
   - Ensure sufficient memory for image generation

4. **Path Finding Failures**:
   - All generated mazes are guaranteed solvable
   - If issues occur, check maze-dataset submodule version

### Debugging

Enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Generate with verbose output
generator = MazeTaskGenerator()
maze = generator.generate_solved_maze(3)
print(f"Start: {maze.start_pos}, End: {maze.end_pos}")
print(f"Solution: {maze.solution}")
print(f"Path length: {len(maze.solution)}")
```

## Future Enhancements

### Planned Features

1. **Variable Difficulty**:
   - Support for larger grid sizes (5×5, 7×7, 10×10)
   - Adjustable complexity parameters
   - Multiple difficulty modes

2. **Advanced Maze Types**:
   - Circular/hexagonal mazes
   - 3D maze representations
   - Multi-level mazes

3. **Enhanced Visualization**:
   - Animated solution paths
   - Heat maps of common paths
   - Alternative marker styles

4. **Evaluation Improvements**:
   - Automatic path validation
   - Efficiency scoring
   - Partial credit for progress

5. **Dataset Variations**:
   - Themed maze designs
   - Obstacle variations
   - Multiple goals/checkpoints

### Extension Points

The architecture supports extensions through:

1. **Custom Generators**: Subclass `MazeTaskGenerator`
2. **Alternative Algorithms**: Replace Kruskal with other methods
3. **New Markers**: Add custom marker rendering functions
4. **Prompt Variations**: Extend `PROMPTS.py` with new instructions
5. **Metadata Extensions**: Add custom fields to `MazeTaskPair`

## References

### Algorithm Resources
- Kruskal's Algorithm: [Wikipedia](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm)
- Maze Generation: [Jamis Buck's Blog](http://weblog.jamisbuck.org/2011/2/7/maze-generation-algorithm-recap)
- Union-Find: [CP-Algorithms](https://cp-algorithms.com/data_structures/disjoint_set_union.html)

### Related Components
- Main Task Registry: `/vmevalkit/runner/create_dataset.py`
- Evaluation System: `/vmevalkit/eval/`
- Model Inference: `/vmevalkit/runner/inference.py`
- Other Tasks: `/vmevalkit/tasks/`

## Notes

- Current implementation limited to 3×3 grids for simplified evaluation
- Each maze guaranteed to have exactly one solution path
- Visual rendering optimized for clarity and consistency
- Temporary files automatically cleaned up after generation
- All coordinates use (row, column) convention consistently
- Maze-dataset submodule must be properly initialized