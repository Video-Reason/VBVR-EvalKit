# 3×3 Sudoku Task for VMEvalKit

## 📊 Overview

The 3×3 Sudoku task evaluates video generation models' capacity for **logical reasoning** and **constraint satisfaction** through simplified Sudoku puzzles. This task tests whether models can:

1. **Understand visual puzzles** - Parse grid structure and identify numbers
2. **Apply logical rules** - Enforce row/column uniqueness constraints  
3. **Generate solution sequences** - Produce videos showing step-by-step solving
4. **Demonstrate reasoning** - Show systematic problem-solving approach

The task uses **3×3 Latin squares** (each row and column contains digits 1, 2, 3 exactly once) as a minimal test case for logical reasoning capabilities.

## 🎯 Task Description

### Input Components
- **First Frame**: A 3×3 grid with some cells filled (blue numbers) and empty cells (light gray background)
- **Prompt**: Text instruction to solve the puzzle following Sudoku rules
- **Format**: 600×600px PNG image at 150 DPI with clear grid lines

### Expected Output
- **Video Sequence**: Animation showing the puzzle being solved
- **Final Frame**: Complete 3×3 grid with all 9 cells filled correctly
- **Solution Path**: Clear demonstration of logical solving steps

### Core Constraints
- Each **row** must contain digits 1, 2, and 3 exactly once
- Each **column** must contain digits 1, 2, and 3 exactly once
- Total of 9 cells arranged in a 3×3 grid

## 🎨 Visual Design Specification

### Grid Layout
```
┌───┬───┬───┐
│ 1 │   │ 3 │  ← Row constraint: needs 2
├───┼───┼───┤
│   │ 2 │   │  
├───┼───┼───┤
│ 3 │   │ 2 │
└───┴───┴───┘
  ↑
Column constraint: needs 2
```

### Visual Elements
| Component | Specification | Purpose |
|-----------|--------------|---------|
| **Grid Lines** | Black, 2px width | Clear cell boundaries |
| **Numbers** | Blue, bold, 24pt font | High visibility |
| **Empty Cells** | Light gray (alpha=0.3) | Visual distinction |
| **Canvas** | White background | Maximum contrast |
| **Aspect Ratio** | 1:1 square | Uniform cell sizing |

## 🧩 Mathematical Foundation

### Latin Square Properties
The task uses **3×3 Latin squares** - the smallest non-trivial case with exactly **12 valid solutions**:

```python
# All 12 possible 3×3 Latin squares (rows listed)
Solutions = [
    [1,2,3, 2,3,1, 3,1,2],  # Solution 1
    [1,2,3, 3,1,2, 2,3,1],  # Solution 2
    [1,3,2, 2,1,3, 3,2,1],  # Solution 3
    [1,3,2, 3,2,1, 2,1,3],  # Solution 4
    [2,1,3, 1,3,2, 3,2,1],  # Solution 5
    [2,1,3, 3,2,1, 1,3,2],  # Solution 6
    [2,3,1, 1,2,3, 3,1,2],  # Solution 7
    [2,3,1, 3,1,2, 1,2,3],  # Solution 8
    [3,1,2, 1,2,3, 2,3,1],  # Solution 9
    [3,1,2, 2,3,1, 1,2,3],  # Solution 10
    [3,2,1, 1,3,2, 2,1,3],  # Solution 11
    [3,2,1, 2,1,3, 1,3,2],  # Solution 12
]
```

### Complexity Analysis
- **Solution Space**: 12 valid complete grids
- **Puzzle Space**: 2^9 = 512 possible masked configurations
- **Valid Puzzles**: Must have unique solution derivable from constraints

## 🎚️ Difficulty Levels

> **Note**: Current implementation uses **1 missing number** for all difficulties (see line 170-174 in `sudoku_reasoning.py`)

### Implemented Configuration
```python
difficulty_map = {
    0: (1, 1),  # Easy: 8 given, 1 missing
    1: (1, 1),  # Medium: 8 given, 1 missing  
    2: (1, 1)   # Hard: 8 given, 1 missing
}
```

### Proposed Enhancement
For more varied difficulty:
```python
difficulty_map = {
    0: (1, 2),   # Easy: 7-8 given numbers
    1: (3, 5),   # Medium: 4-6 given numbers
    2: (6, 7)    # Hard: 2-3 given numbers
}
```

## 🧠 Reasoning Requirements

### Level 1: Visual Understanding
- **Grid Parsing**: Identify 3×3 structure and cell positions
- **Number Recognition**: Distinguish digits 1, 2, 3 from empty cells
- **Spatial Mapping**: Track row/column indices (0-2)

### Level 2: Rule Application
- **Row Scanning**: Check which numbers are missing in each row
- **Column Scanning**: Check which numbers are missing in each column
- **Constraint Intersection**: Find cells with unique valid values

### Level 3: Solution Strategies

#### Strategy A: Direct Placement
```
Scenario: Row has [1, _, 3] 
Analysis: Missing digit is 2
Action: Place 2 in empty cell
```

#### Strategy B: Elimination
```
Scenario: Empty cell at position (1,1)
Row check: Row 1 missing {1,3}
Column check: Column 1 missing {2,3}
Intersection: Only 3 satisfies both → Place 3
```

#### Strategy C: Constraint Propagation
```
1. Fill obvious cells (single candidates)
2. Update constraints after each placement
3. Repeat until solved
```

## 📝 Data Structure

### Core Task Representation
```python
@dataclass
class SudokuTaskPair:
    # Identification
    id: str                      # Unique task identifier (e.g., "sudoku_0042")
    task_category: str           # Always "Sudoku"
    
    # Task Content  
    prompt: str                  # Instruction text for the model
    first_image_path: str        # Path to puzzle image (partial grid)
    final_image_path: str        # Path to solution image (complete grid)
    
    # Puzzle Data
    puzzle_array: List[int]      # 9-element array (None for empty cells)
    solution_array: List[int]    # Complete 9-element solution
    num_given: int              # Number of pre-filled cells (1-8)
    
    # Metadata
    difficulty: str              # "easy", "medium", or "hard"
    sudoku_data: Dict[str, Any]  # Additional puzzle metadata
    created_at: str             # ISO timestamp of generation
```

### Array Representation
Cells are indexed 0-8 in row-major order:
```
Grid positions:     Flat array indices:
┌───┬───┬───┐      
│ 0 │ 1 │ 2 │      puzzle_array = [
├───┼───┼───┤        cell_0, cell_1, cell_2,  # Row 0
│ 3 │ 4 │ 5 │        cell_3, cell_4, cell_5,  # Row 1  
├───┼───┼───┤        cell_6, cell_7, cell_8   # Row 2
│ 6 │ 7 │ 8 │      ]
└───┴───┴───┘
```

## 🔄 Generation Pipeline

### 1. Solution Selection
```python
def generate_solved_sudoku():
    solutions = [...]  # All 12 valid Latin squares
    return random.choice(solutions).copy()
```

### 2. Puzzle Creation
```python
def create_puzzle(solution, difficulty_level):
    puzzle = solution.copy()
    positions_to_remove = random.sample(range(9), num_to_remove)
    for pos in positions_to_remove:
        puzzle[pos] = None
    return puzzle
```

### 3. Image Generation
```python
def create_board_image(sudoku_array, filepath):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Draw grid (4 lines for 3×3)
    for i in range(4):
        ax.axhline(y=i, color='black', linewidth=2)
        ax.axvline(x=i, color='black', linewidth=2)
    
    # Place numbers and mark empty cells
    for i in range(3):
        for j in range(3):
            if sudoku_array[i*3 + j] is not None:
                ax.text(...)  # Blue number
            else:
                ax.add_patch(...)  # Gray background
```

### 4. Validation
```python
def validate_solution(grid):
    # Check row constraints
    for row in rows:
        if sorted(row) != [1, 2, 3]: return False
    
    # Check column constraints  
    for col in columns:
        if sorted(col) != [1, 2, 3]: return False
        
    return True
```

## 📊 Evaluation Framework

### Primary Metrics
| Metric | Description | Scoring |
|--------|-------------|---------|
| **Correctness** | Final grid satisfies all constraints | Binary (0/1) |
| **Completeness** | All 9 cells filled | Count (0-9) |
| **Validity** | No constraint violations | Binary (0/1) |
| **Visual Clarity** | Clear number placement in video | Scale (1-5) |

### Evaluation Methods

#### 1. Automatic (GPT-4O Vision)
```python
# Evaluation prompt for GPT-4O
prompt = """
Compare the final frame with expected solution.
Check if all numbers match and constraints are satisfied.
Rate 1-5 based on correctness and clarity.
"""
```

#### 2. Human Evaluation
- Visual inspection of generated video
- Manual verification of solution correctness
- Assessment of reasoning demonstration

#### 3. Programmatic Validation
```python
def evaluate_solution(generated, expected):
    # Extract final grid from video
    final_grid = extract_final_frame(generated)
    
    # Parse numbers from image
    parsed = parse_sudoku_grid(final_grid)
    
    # Validate constraints
    return validate_solution(parsed) and parsed == expected
```

## 🚀 Usage Examples

### Basic Dataset Generation
```python
from vmevalkit.tasks.sudoku_task import create_dataset

# Generate 50 puzzles with mixed difficulties
dataset = create_dataset(num_samples=50)

# Access individual tasks
for task in dataset['pairs']:
    print(f"Task {task['id']}: {task['num_given']}/9 given numbers")
    print(f"Puzzle: {task['puzzle_array']}")
    print(f"Solution: {task['solution_array']}")
```

### Custom Generation
```python
from vmevalkit.tasks.sudoku_task import SudokuTaskGenerator

generator = SudokuTaskGenerator()

# Generate single task with specific difficulty
task = generator.generate_single_task(
    task_id="custom_001",
    difficulty=2  # Hard level
)
```

### Integration with Runner
```python
# Via command line
python vmevalkit/runner/create_dataset.py \
    --pairs-per-domain 100 \
    --random-seed 42

# Programmatic
from vmevalkit.runner.create_dataset import generate_domain_to_folders

tasks = generate_domain_to_folders(
    domain_name="sudoku",
    num_samples=100,
    output_base=Path("data/questions"),
    random_seed=42
)
```

## 🔧 Technical Implementation

### Dependencies
- **NumPy**: Array operations and mathematical computations
- **Matplotlib**: Grid visualization and image generation
- **Pillow (PIL)**: Image processing and saving
- **Dataclasses**: Structured data representation

### File Structure
```
vmevalkit/tasks/sudoku_task/
├── __init__.py           # Module exports and initialization
├── PROMPTS.py           # Standardized prompt templates
├── sudoku_reasoning.py  # Core implementation
└── SUDOKU.md           # This documentation
```

### Key Classes
| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `Simple3x3SudokuGenerator` | Puzzle generation logic | `generate_solved_sudoku()`, `create_puzzle()` |
| `SudokuTaskGenerator` | Task orchestration | `generate_single_task()`, `generate_dataset()` |
| `SudokuTaskPair` | Data structure | Dataclass with validation |
| `SudokuDataset` | Collection management | `save()`, `load()` |

## 🎯 Model Performance Insights

### Expected Behaviors

#### Successful Solution
```
Frame 1: Shows puzzle with missing numbers
Frame 2-N: Progressive filling of empty cells
Final Frame: Complete valid solution
```

#### Common Failure Modes
1. **Constraint Violation**: Placing duplicate numbers in row/column
2. **Incomplete Solution**: Some cells remain empty
3. **Visual Artifacts**: Unclear or distorted numbers
4. **Random Guessing**: No logical progression shown

### Performance Factors
- **Visual Acuity**: Ability to parse grid structure
- **Working Memory**: Tracking constraints across cells
- **Logical Reasoning**: Systematic rule application
- **Sequential Planning**: Ordered solution steps

## 📈 Future Enhancements

### Proposed Improvements
1. **Variable Difficulty**: Implement true difficulty scaling with 2-7 missing numbers
2. **Multiple Solutions**: Test handling of puzzles with non-unique solutions
3. **4×4 Grids**: Extend to larger grids (digits 1-4) for increased complexity
4. **Solution Paths**: Track and evaluate solving strategy used
5. **Hint System**: Provide partial solutions as additional input
6. **Error Recovery**: Test correction of intentionally wrong placements

### Research Questions
- Can models learn solving strategies from examples?
- Do models show consistent solving patterns?
- How does performance scale with grid size?
- Can models explain their reasoning process?

## 🔗 Related Resources

### VMEvalKit Tasks
- [Chess Task](../chess_task/CHESS.md) - Strategic reasoning
- [Maze Task](../maze_task/MAZE.md) - Spatial navigation
- [Raven Task](../raven_task/RAVEN.md) - Pattern completion
- [Rotation Task](../rotation_task/ROTATION.md) - 3D visualization

### External References
- [Latin Squares Mathematics](https://en.wikipedia.org/wiki/Latin_square)
- [Sudoku Complexity Theory](https://en.wikipedia.org/wiki/Mathematics_of_Sudoku)
- [Constraint Satisfaction Problems](https://en.wikipedia.org/wiki/Constraint_satisfaction_problem)

## 📝 Citation

```bibtex
@misc{vmevalkit2024sudoku,
  title={3×3 Sudoku Task for Video Model Reasoning Evaluation},
  author={VMEvalKit Team},
  year={2024},
  howpublished={VMEvalKit: Video Model Evaluation Toolkit}
}
```

---
*Generated as part of VMEvalKit - A comprehensive toolkit for evaluating video generation models' reasoning capabilities.*