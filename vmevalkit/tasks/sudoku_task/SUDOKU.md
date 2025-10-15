# 3x3 Sudoku Task for VMEvalKit

## Overview

The 3x3 Sudoku task evaluates video models' ability to solve simple logical puzzles through visual reasoning. Models must understand basic Sudoku rules, identify missing numbers, and demonstrate step-by-step solutions through generated video sequences.

## Task Description

**Input**: A partially filled 3x3 Sudoku grid with numbers 1, 2, 3
**Output**: Complete solution showing all cells filled according to Sudoku rules
**Prompt**: Instructions to solve the puzzle following Sudoku constraints

## Visual Format

- **Grid**: 3x3 Sudoku board with clear grid lines
- **Numbers**: Simple black digits (1, 2, 3) on white background
- **Layout**: Clean, high-contrast visualization with grid lines
- **Style**: Clear typography optimized for video model parsing

## Difficulty Levels

1. **Easy** (Level 0): 8 given numbers out of 9 total (only 1 missing)
2. **Medium** (Level 1): 4-5 given numbers out of 9 total  
3. **Hard** (Level 2): 2-3 given numbers out of 9 total

## Reasoning Requirements

### Visual Understanding
- Parse partially filled 3x3 Sudoku grid from image
- Identify given numbers vs. empty cells
- Understand grid structure and cell positions

### Logical Rules
- **Row constraint**: Each row must contain digits 1, 2, 3 exactly once
- **Column constraint**: Each column must contain digits 1, 2, 3 exactly once

### Solution Strategies
- **Direct placement**: Only one number possible in a cell
- **Elimination**: Use row/column constraints to narrow possibilities
- **Logical deduction**: Apply systematic constraint satisfaction

## Data Structure

```python
@dataclass
class SudokuTaskPair:
    id: str                      # Unique task identifier
    prompt: str                  # Solution instructions
    first_image_path: str        # Puzzle image (incomplete 3x3)
    final_image_path: str        # Solution image (complete 3x3)
    task_category: str           # "Sudoku"
    difficulty: str              # "easy", "medium", "hard"
    puzzle_array: List[int]      # 9-element flat array representation
    solution_array: List[int]    # Complete solution array
    num_given: int              # Count of given digits (out of 9)
    sudoku_data: Dict[str, Any] # Additional metadata
```

## Generation Process

1. **Solution Generation**: Pick random valid 3x3 Latin square (all 12 possibilities)
2. **Puzzle Creation**: Remove numbers based on difficulty level  
3. **Image Generation**: Create clean puzzle and solution images with grid
4. **Validation**: Ensure solution is valid Latin square

## Evaluation Criteria

### Correctness
- ✅ All cells filled with valid numbers (1, 2, 3)
- ✅ No violations of row/column constraints
- ✅ Matches expected solution exactly

### Visual Reasoning
- ✅ Identifies empty vs. filled cells correctly
- ✅ Recognizes simple digit patterns accurately
- ✅ Understands 3x3 grid structure

### Logical Process
- ✅ Shows step-by-step reasoning in video
- ✅ Follows systematic solving approach
- ✅ Demonstrates constraint satisfaction logic

## Usage Example

```python
from vmevalkit.tasks.sudoku_task import create_dataset

# Generate dataset with mixed difficulties
dataset = create_dataset(
    num_tasks=10,
    difficulties=[0, 1, 2],  # Easy, Medium, Hard
    output_dir="data/questions/sudoku_task"
)

print(f"Generated {len(dataset)} 3x3 Sudoku tasks")
```

## Applications

- **Logic Testing**: Evaluate basic constraint satisfaction abilities
- **Visual Processing**: Test simple digit recognition in grid contexts  
- **Sequential Reasoning**: Assess step-by-step problem solving
- **Pattern Recognition**: Identify basic logical solving patterns

## Related Work

Inspired by the MNIST Sudoku Generator: https://github.com/kairess/mnist_sudoku_generator

Simplified to 3x3 Latin squares for focused evaluation of logical reasoning in video models.
