% Dot to Dot Puzzle Task Specification

# Overview

The Dot to Dot domain evaluates whether video generation models can connect numbered dots in sequential order to reveal a complete image. This task tests sequential reasoning, number ordering, and visual pattern completion capabilities.

Each sample provides:

- First frame: Numbered dots scattered across the canvas with no connections.
- Final frame: All dots connected in numerical order, revealing the complete pattern.
- Prompt: Instruction telling the model to connect dots from 1 to N in sequence.
- Metadata: Dot positions, numbers, pattern type, difficulty, etc.

Models must render a video that draws lines between consecutive numbered dots while maintaining a fixed camera view.

# Visual Structure

- Canvas: 768×512 px PNG (white background).
- Dots: Black circles with white numbers (1, 2, 3, ...).
- Lines: Blue lines connecting dots in numerical order.
- Pattern types:
  - `star`: Star shape with alternating outer/inner points
  - `heart`: Heart shape using parametric equations
  - `circle`: Circular arrangement
  - `triangle`: Triangular perimeter
  - `square`: Square perimeter
  - `spiral`: Spiral pattern from center outward

# Difficulty Levels

| Difficulty | Number of Dots | Pattern Complexity |
|------------|----------------|-------------------|
| easy       | 10-20          | Simple shapes (circle, square, triangle) |
| medium     | 20-40          | Moderate shapes (star, heart) |
| hard       | 40-60          | Complex patterns (spiral, detailed shapes) |

# Prompt Template

Prompts emphasize sequential connection and fixed camera:

```
Connect the numbered dots in order from 1 to {max_number}. Draw a continuous line connecting each dot to the next one in sequence. Keep the camera view fixed and maintain the dot positions unchanged. Stop the video when all dots are connected and the complete image is revealed.
```

Key properties:
- Emphasize numerical order (1, 2, 3, ...)
- Continuous line drawing without lifting
- Fixed camera view
- Completion when pattern is revealed

# Data Files per Question

```
data/questions/dot_to_dot_task/dot_to_dot_XXXX/
├── first_frame.png
├── final_frame.png
├── prompt.txt
└── question_metadata.json
```

- `first_frame.png` – dots with numbers, no lines
- `final_frame.png` – complete pattern with all dots connected
- `prompt.txt` – instruction text
- `question_metadata.json` – includes:
  - `task_id`, `domain`, `difficulty`
  - `num_dots`, `max_number`
  - `dots` array with number and position for each dot
  - `canvas_size`, `camera` info

# Generation Pipeline

`dot_to_dot_reasoning.py` contains:

1. **DotToDotRenderer**: Draws first/final PNGs using matplotlib
   - First frame: dots with numbers only
   - Final frame: dots connected with blue lines
2. **DotToDotGenerator**:
   - Samples number of dots based on difficulty
   - Generates pattern (star, heart, circle, triangle, square, spiral)
   - Creates dot positions following pattern equations
   - Builds prompt and metadata, writes files
3. **create_dataset**:
   - Accepts `num_samples`, `difficulties`, `seed`
   - Loops to create question folders and returns standard dataset dict

# Pattern Generation

Patterns are generated using parametric equations or geometric distributions:

- **Star**: Alternating outer/inner radius points
- **Heart**: Parametric heart equation
- **Circle**: Uniform angular distribution
- **Triangle**: Points distributed along three sides
- **Square**: Points distributed along four sides
- **Spiral**: Archimedean spiral from center

# Integration

1. Generate samples: `python examples/create_questions.py --task dot_to_dot --pairs-per-domain 50`
2. Register task in `vmevalkit/runner/TASK_CATALOG.py`:
   ```python
   'dot_to_dot': {
       'name': 'Dot to Dot',
       'description': 'Sequential dot connection to reveal patterns',
       'module': 'vmevalkit.tasks.dot_to_dot_task',
       'create_function': 'create_dataset',
       'process_dataset': lambda dataset, num_samples: dataset['pairs']
   }
   ```

# Quality Checklist

- [ ] Dots are numbered sequentially from 1 to N
- [ ] First frame shows dots without lines
- [ ] Final frame shows complete connected pattern
- [ ] Lines connect dots in correct numerical order
- [ ] Pattern is recognizable (star, heart, etc.)
- [ ] Metadata includes all dot positions and numbers
- [ ] Task registered in `TASK_CATALOG` before running pipeline

