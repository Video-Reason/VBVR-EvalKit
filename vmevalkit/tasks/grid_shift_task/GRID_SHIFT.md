# Grid Shift Task

## Overview

The Grid Shift task evaluates whether a model can apply a uniform translation
to all blocks in a grid without leaving the grid bounds. All blocks share the
same color.

## Task Structure

Each task pair contains:
- **First Frame**: 6x6 grid with three colored blocks
- **Final Frame**: All blocks shifted 1-2 steps in a random direction
- **Prompt**: Instruction describing direction and number of steps

## Rules

- Grid size: 6x6
- Blocks: exactly 3
- Color: all blocks share the same color
- Direction: random per task (up, down, left, right)
- Steps: 1 or 2
- No block leaves the grid (initial positions are sampled to guarantee this)

## Data Structure

```json
{
  "id": "grid_shift_0001",
  "prompt": "...",
  "first_image_path": "path/to/first_frame.png",
  "final_image_path": "path/to/final_frame.png",
  "task_category": "GridShift",
  "grid_shift_data": {
    "grid_size": 6,
    "num_blocks": 3,
    "color": "blue",
    "direction": "right",
    "steps": 2,
    "positions": [[0, 1], [2, 3], [4, 0]],
    "shifted_positions": [[0, 3], [2, 5], [4, 2]]
  },
  "difficulty": "medium",
  "created_at": "2025-01-01T00:00:00Z"
}
```

## Usage

```bash
python examples/create_questions.py --task grid_shift --pairs-per-domain 50
```
