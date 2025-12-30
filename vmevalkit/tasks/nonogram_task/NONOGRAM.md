% Nonogram Puzzle Task Specification

# Overview

The Nonogram domain evaluates whether video generation models can solve nonogram puzzles by filling in grid cells according to row and column hints. This task tests constraint satisfaction reasoning, logical deduction, and pattern recognition capabilities.

Each sample provides:

- First frame: Blank grid with row and column hints (numbers indicating consecutive filled blocks).
- Final frame: Complete solution with all cells correctly filled to reveal the hidden pattern.
- Prompt: Instruction telling the model to solve the puzzle by filling cells according to hints.
- Metadata: Grid size, pattern type, hints, difficulty, etc.

Models must render a video that fills in the grid cells progressively based on logical reasoning from the hints, revealing the complete pattern.

# Visual Structure

- Canvas: 768×512 px PNG (light gray background #f8fafc).
- Grid: N×N cells (cell size: 35px) with light gray borders.
- Row hints: Numbers displayed on the left side of the grid, indicating consecutive filled block lengths in each row.
- Column hints: Numbers displayed above the grid, indicating consecutive filled block lengths in each column.
- Filled cells: Dark gray (#1e293b) when solution is shown.
- Empty cells: White background.

# Difficulty Levels

| Difficulty | Grid Size | Pattern Types | Fill Density | Description |
|------------|-----------|---------------|--------------|-------------|
| easy       | 5×5 to 6×6 | cross, square, diagonal, checkerboard | 25-35% | Small grids with simple, recognizable patterns. Fewer, longer consecutive blocks for simpler hints. |
| medium     | 7×7 to 10×10 | cross, square, circle, letter_t, checkerboard | 40-50% | Medium grids with moderate complexity. Balanced pattern types. |
| hard       | 12×12 to 15×15 | circle, letter_t, random | 55-70% | Large grids with complex patterns. Higher density and more fragmented hints. |

## Difficulty Distribution

The dataset uses a balanced distribution:
- **Easy**: 40% of samples
- **Medium**: 40% of samples  
- **Hard**: 20% of samples

This ensures a good mix of difficulty levels for comprehensive evaluation.

# Pattern Types

The task generates various pattern types:

- **Cross**: Vertical and horizontal lines forming a cross
- **Square**: Hollow square border
- **Circle**: Circular pattern (approximated)
- **Checkerboard**: Alternating filled/empty cells
- **Letter T**: Letter T shape
- **Diagonal**: Diagonal line pattern
- **Random**: Random pattern with controlled density

# Prompt Template

Single comprehensive prompt:

```
Solve this nonogram puzzle by filling in the grid cells according to the row and column hints. The numbers on the left indicate the lengths of consecutive filled blocks in each row, and the numbers on top indicate the lengths of consecutive filled blocks in each column. Fill in the cells to reveal the hidden pattern. Keep the camera view fixed in the top-down perspective and maintain the grid structure unchanged. Stop the video when all cells are correctly filled and the complete pattern is revealed.
```

Key properties:
- Emphasize constraint satisfaction (row/column hints)
- Fixed top-down camera
- Progressive cell filling
- Completion when pattern is revealed

# Data Files per Question

```
data/questions/nonogram_task/nonogram_XXXX/
├── first_frame.png
├── final_frame.png
├── prompt.txt
└── question_metadata.json
```

- `first_frame.png` – blank grid with hints (no filled cells)
- `final_frame.png` – complete solution (all cells filled correctly)
- `prompt.txt` – instruction text
- `question_metadata.json` – includes:
  - `task_id`, `domain`, `difficulty`
  - `grid_size` (N×N)
  - `pattern_type` (cross, square, circle, etc.)
  - `row_hints` and `col_hints` arrays
  - `filled_cells`, `total_cells`, `fill_percentage`
  - `canvas_size`, `camera` info

# Generation Pipeline

`nonogram_reasoning.py` contains:

1. **NonogramRenderer**: Draws first/final PNGs using matplotlib
   - First frame: blank grid with hints displayed outside
   - Final frame: complete solution with filled cells
2. **NonogramGenerator**:
   - Samples grid size based on difficulty
   - Generates pattern (cross, square, circle, etc.)
   - Calculates row and column hints from pattern
   - Ensures uniqueness of generated puzzles
   - Builds prompt and metadata, writes files
3. **create_dataset**:
   - Accepts `num_samples`, `difficulties`, `seed`
   - Loops to create question folders and returns standard dataset dict

# Hint Calculation

Hints are calculated by analyzing consecutive filled blocks:

- **Row hints**: For each row, count consecutive filled cells and list their lengths.
  - Example: `[1,0,1,1,0]` → `[1, 2]` (one block of length 1, one block of length 2)
- **Column hints**: Same logic applied to each column.
- Empty rows/columns have hint `[0]`.

# Integration

1. Generate samples: `python examples/create_questions.py --task nonogram --pairs-per-domain 50`
2. Register task in `vmevalkit/runner/TASK_CATALOG.py`:
   ```python
   'nonogram': {
       'name': 'Nonogram',
       'description': 'Constraint satisfaction puzzle with row/column hints',
       'module': 'vmevalkit.tasks.nonogram_task',
       'create_function': 'create_dataset',
       'process_dataset': lambda dataset, num_samples: dataset['pairs']
   }
   ```

# Quality Checklist

- [ ] Grid size matches difficulty (5×5 to 15×15)
- [ ] First frame shows blank grid with hints outside
- [ ] Final frame shows complete solution with all cells filled
- [ ] Row hints are displayed on the left, column hints on top
- [ ] Hints correctly represent the pattern
- [ ] Pattern is recognizable (cross, square, circle, etc.)
- [ ] Metadata includes all hints and pattern information
- [ ] Task registered in `TASK_CATALOG` before running pipeline

