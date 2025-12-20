% Symmetry Completion Task Specification

# Overview

The Symmetry Completion domain evaluates whether video generation models can complete visual patterns in grids by filling in missing cells based on pattern rules. This task tests pattern recognition, logical reasoning, and visual completion capabilities.

Each sample provides:

- First frame: Grid with partial pattern (some cells filled, some missing).
- Final frame: Complete pattern (all cells filled according to the pattern rule).
- Prompt: Instruction telling the model to complete the pattern by filling missing cells.
- Metadata: Pattern type, grid size, missing positions, difficulty, etc.

Models must render a video that fills in the missing cells to complete the pattern while maintaining a fixed camera view.

# Visual Structure

- Canvas: 768×512 px PNG (light gray background #f8fafc).
- Grid: 3×3, 4×4, or 5×5 cells (centered on canvas).
- Cells: White background with black borders.
- Filled cells: Black (#1e293b) fill.
- Missing cells: Empty (white) in first frame, filled in final frame.

# Pattern Types

The task supports 16 different pattern types, including both simple patterns and Raven-style progressive patterns:

## Simple Patterns

1. **Checkerboard**: Alternating black and white cells (chessboard pattern).
2. **Horizontal Stripes**: Each row has the same color (alternating rows).
3. **Vertical Stripes**: Each column has the same color (alternating columns).
4. **Diagonal**: Main diagonal and/or anti-diagonal filled.
5. **Border**: Outer border filled, center empty.
6. **Cross**: Center row and center column filled.
7. **Corners**: Four corner cells filled.
8. **Center**: Center region filled (size depends on grid size).

## Raven-Style Progressive Patterns

9. **Row Increment**: Each row has an incrementing number of filled cells (row 0: 1 cell, row 1: 2 cells, etc.).
10. **Column Increment**: Each column has an incrementing number of filled cells (col 0: 1 cell, col 1: 2 cells, etc.).
11. **Row Position Shift**: Each row has one filled cell, position shifts right across rows.
12. **Column Position Shift**: Each column has one filled cell, position shifts down across columns.
13. **Combined Row+Col**: Combination of row increment and column position shift.
14. **Diagonal Increment**: Main diagonal and anti-diagonal filled.
15. **Horizontal Symmetry**: Top half mirrors bottom half (horizontal symmetry).
16. **Vertical Symmetry**: Left half mirrors right half (vertical symmetry).

# Difficulty Levels

| Difficulty | Grid Size | Missing Cells | Pattern Types |
|------------|-----------|---------------|---------------|
| easy       | 3×3, 4×4  | 30-40%        | Checkerboard, Stripes, Row/Col Increment, Position Shift |
| medium     | 4×4, 5×5  | 40-50%        | Checkerboard, Stripes, Diagonal, Border, Cross, Raven-style patterns |
| hard       | 5×5, 6×6  | 50-60%        | All pattern types (16 types) |

# Prompt Template

Single comprehensive prompt:

```
Complete this pattern by filling in the missing grid cells. Observe the existing pattern and determine the rule that governs how cells should be filled. Fill in all empty cells to complete the pattern. Keep the camera view fixed in the top-down perspective and maintain all existing cells unchanged. Stop the video when the pattern is fully completed.
```

Key properties:
- Emphasize pattern observation and rule identification
- Fixed top-down camera
- Fill missing cells only
- Completion when pattern is fully filled

# Data Files per Question

```
data/questions/symmetry_completion_task/symmetry_completion_XXXX/
├── first_frame.png
├── final_frame.png
├── prompt.txt
└── question_metadata.json
```

- `first_frame.png` – incomplete pattern (some cells missing)
- `final_frame.png` – complete pattern (all cells filled)
- `prompt.txt` – instruction text
- `question_metadata.json` – includes:
  - `task_id`, `domain`, `difficulty`
  - `pattern_type`, `grid_size`
  - `num_missing`, `missing_positions`
  - `full_pattern` (complete pattern as 2D array)
  - `canvas_size`, `camera` info

# Generation Pipeline

`symmetry_completion_reasoning.py` contains:

1. **SymmetryCompletionRenderer**: Draws first/final PNGs using matplotlib
   - First frame: grid with incomplete pattern (left half visible, right half missing)
   - Final frame: grid with complete symmetric pattern
2. **SymmetryCompletionGenerator**:
   - Generates patterns based on type and grid size
   - Selects missing positions based on difficulty
   - Ensures uniqueness through signature checking
   - Builds prompt and metadata, writes files
3. **create_dataset**:
   - Accepts `num_samples`, `difficulties`, `seed`
   - Loops to create question folders and returns standard dataset dict

# Scaling Strategy

The task focuses on **symmetry completion** and can generate 50+ unique samples through:

- **4 symmetry pattern types**: Different symmetric visual patterns
- **4 grid sizes**: 4×4, 6×6, 8×8, 10×10 (even numbers only for perfect symmetry)
- **3 difficulty levels**: Easy, medium, hard
- **Missing ratio ranges**: Different percentages of right half missing (30-40%, 50-60%, 70-80%)
- **Multiple missing positions**: Each (pattern, size, difficulty) combination can have different missing positions in the right half
- **Total combinations**: 4 × 4 × 3 × multiple positions = 200+ possible combinations

See `SCALING_PLAN.md` for detailed scaling strategy and future expansion plans.

# Uniqueness

## Task Uniqueness (Avoiding Duplicates)

Each task is uniquely identified by:
- Pattern type
- Grid size
- Missing positions
- Full pattern structure

Signatures are generated to avoid duplicates within a batch.

## Solution Uniqueness (Ensuring Single Correct Answer)

The task ensures that each `first_frame` has a **unique** `final_frame` solution through:

1. **Pattern-Specific Missing Position Selection**:
   - Row/Col Increment patterns: Each row/column retains at least one visible cell to show the increment rule
   - Position Shift patterns: First and last rows/columns are kept visible to observe the shift
   - Symmetry patterns: Key positions defining symmetry are preserved
   - Other patterns: Random selection with sufficient information guarantee

2. **Uniqueness Verification**:
   - Each pattern type has specific verification rules
   - Ensures sufficient visible cells (40-50% minimum depending on pattern complexity)
   - Verifies that key pattern-defining information is visible
   - Retries with adjusted missing counts if verification fails

3. **Minimum Visible Cell Requirements**:
   - Simple patterns (checkerboard, stripes): ≥40% visible
   - Complex patterns (increment, symmetry): ≥50% visible
   - Combined patterns: ≥50% visible

4. **Verification Examples**:
   - Checkerboard: Must see at least 2 adjacent cells to determine alternating pattern
   - Increment patterns: Each row/column must have at least one visible filled cell
   - Symmetry patterns: At least 40% of defining half must be visible
   - Position shift: At least 2-3 rows/columns must be visible to observe shift

This ensures that models can uniquely determine the pattern rule from the incomplete grid, leading to a single correct solution.

# Integration

1. Generate samples: `python examples/create_questions.py --task symmetry_completion --pairs-per-domain 50`
2. Register task in `vmevalkit/runner/TASK_CATALOG.py`:
   ```python
   'symmetry_completion': {
       'name': 'Symmetry Completion',
       'description': 'Complete visual patterns in grids by filling missing cells',
       'module': 'vmevalkit.tasks.symmetry_completion_task',
       'create_function': 'create_dataset',
       'process_dataset': lambda dataset, num_samples: dataset['pairs']
   }
   ```

# Quality Checklist

- [ ] All patterns are centered in grid
- [ ] First frame shows incomplete pattern with missing cells
- [ ] Final frame shows complete pattern
- [ ] Missing positions are clearly visible (empty cells)
- [ ] Pattern rules are consistent and solvable
- [ ] Metadata includes all pattern information
- [ ] Task registered in `TASK_CATALOG` before running pipeline

