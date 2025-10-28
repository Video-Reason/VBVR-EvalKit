# Paper Assets - Sora Score=5 Examples

This directory contains one exemplary Sora output (score=5) per task type.

## Directory Structure

```
paper/assets/
├── chess_task/
│   ├── sora_chess_example_12frames.png   # 12-frame temporal decomposition
│   ├── sora_chess_example_12frames.eps   # Vector format for paper
│   ├── first_frame.png                   # Original input image
│   ├── final_frame.png                   # Target/solution image  
│   ├── prompt.txt                        # Text prompt used
│   └── question_metadata.json            # Task metadata
├── maze_task/
│   └── ... (same structure)
├── raven_task/
│   └── ... (same structure)
├── rotation_task/
│   └── ... (same structure)
└── sudoku_task/
    └── ... (same structure)
```

## Examples Used

- **Chess**: chess_0001 (Score: 5/5)
- **Maze**: maze_0001 (Score: 5/5)
- **Raven**: raven_0001 (Score: 5/5)
- **Rotation**: rotation_0014 (Score: 5/5)
- **Sudoku**: sudoku_0000 (Score: 5/5)

## Frame Decomposition Details

Each 12-frame sequence shows:
- Frames extracted at: 0, 21, 43, 65, 87, 109, 130, 152, 174, 196, 218, 239
- Total video duration: 8 seconds at 30 FPS
- Frame labels show actual frame indices
- Publication-ready at 300 DPI
