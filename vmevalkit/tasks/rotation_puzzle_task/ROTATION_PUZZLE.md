% Rotation Puzzle Task Specification

# Overview

The Rotation Puzzle domain evaluates whether video generation models can solve pipe puzzle problems by rotating squares to connect pipe paths. This task is similar to pipe puzzles but uses geometric shapes (squares) that can be rotated.

Each sample provides:

- First frame: Four squares with pipe patterns rotated to different angles, paths not connected.
- Final frame: All squares rotated to correct angles, pipe paths form a continuous connection.
- Prompt: Instruction telling the model to rotate the squares to connect the pipe paths.
- Metadata: Square positions, initial/target angles, pipe patterns, difficulty, etc.

Models must render a video that rotates the squares smoothly to connect all pipe paths while maintaining a fixed camera view.

# Visual Structure

- Canvas: 768×512 px PNG (light gray background #f8fafc).
- Four squares arranged in a 2×2 grid, centered on canvas.
- Each square contains pipe paths (L-shapes) that can connect to adjacent squares.
- Square size: 140px, spacing: 20px between squares.
- Pipe paths: Blue (#3b82f6) with connections on top, right, bottom, and/or left sides.
- No labels, no connecting lines, no borders - just the four squares.

# Difficulty Levels

| Difficulty | Rotations Needed | Description |
|------------|------------------|-------------|
| easy       | 1-2 squares      | Simple: only 1-2 squares need rotation |
| medium     | 2-3 squares      | Moderate: 2-3 squares need rotation |
| hard       | 3-4 squares      | Complex: 3-4 squares need rotation |

# Prompt Template

Single comprehensive prompt:

```
Solve this rotation puzzle by rotating the four squares to connect the pipe paths. Each square can be rotated 90 degrees clockwise or counterclockwise. Rotate the squares so that all pipe paths connect to form a continuous path. Keep the camera view fixed in the top-down perspective and maintain all square positions unchanged. Stop the video when all pipes are connected and the puzzle is solved.
```

Key properties:
- Emphasize rotation actions (90° clockwise/counterclockwise)
- Fixed top-down camera
- Continuous path connection
- Completion when puzzle is solved

# Data Files per Question

```
data/questions/rotation_puzzle_task/rotation_puzzle_XXXX/
├── first_frame.png
├── final_frame.png
├── prompt.txt
└── question_metadata.json
```

- `first_frame.png` – unsolved state (squares at initial rotations)
- `final_frame.png` – solved state (squares at target rotations, paths connected)
- `prompt.txt` – instruction text
- `question_metadata.json` – includes:
  - `task_id`, `domain`, `difficulty`
  - `num_squares` (always 4)
  - `squares` array with position, initial_angle, target_angle, pipe_pattern
  - `canvas_size`, `camera` info

# Generation Pipeline

`rotation_puzzle_reasoning.py` contains:

1. **RotationPuzzleRenderer**: Draws first/final PNGs using matplotlib
   - First frame: squares at initial rotation angles
   - Final frame: squares at target rotation angles (all 0° for solved state)
2. **RotationPuzzleGenerator**:
   - Creates 4 squares with L-shaped pipe patterns
   - Assigns initial angles based on difficulty
   - Ensures puzzle is solvable (target angles all 0°)
   - Builds prompt and metadata, writes files
3. **create_dataset**:
   - Accepts `num_samples`, `difficulties`, `seed`
   - Loops to create question folders and returns standard dataset dict

# Pipe Pattern Types

Each square has an L-shaped pipe pattern with connections on two adjacent sides:
- Top-Right L: connections on top and right
- Right-Bottom L: connections on right and bottom
- Bottom-Left L: connections on bottom and left
- Left-Top L: connections on left and top

Patterns are shuffled and assigned to squares randomly.

# Integration

1. Generate samples: `python examples/create_questions.py --task rotation_puzzle --pairs-per-domain 50`
2. Register task in `vmevalkit/runner/TASK_CATALOG.py`:
   ```python
   'rotation_puzzle': {
       'name': 'Rotation Puzzle',
       'description': 'Pipe puzzle with rotatable squares',
       'module': 'vmevalkit.tasks.rotation_puzzle_task',
       'create_function': 'create_dataset',
       'process_dataset': lambda dataset, num_samples: dataset['pairs']
   }
   ```

# Quality Checklist

- [ ] All squares are centered in 2×2 grid
- [ ] First frame shows squares at initial rotations
- [ ] Final frame shows squares at target rotations (all 0°)
- [ ] Pipe paths are visible and correctly rotated
- [ ] No angle numbers, labels, or connecting lines
- [ ] Metadata includes all square positions and angles
- [ ] Task registered in `TASK_CATALOG` before running pipeline

