% Shape Sorter Task Specification

# Overview

The Shape Sorter domain evaluates whether video generation models can plan and execute multi-step shape matching actions from a fixed top-down perspective. The task is inspired by infant cognition shape-sorting toys, but simplified to a flat 2D board.

Each sample provides:

- First frame: Colored shape cards staged on one side of the board, outlines on the opposite side.
- Final frame: All shapes aligned inside their matching outlines.
- Prompt: Instruction telling the model to slide each card into the corresponding slot and stop once the board is solved.
- Metadata: Shape list, colors, layout variant, sizes, camera information, etc.

Models must render a video that smoothly moves the cards from their starting positions to the matching outlines without teleportation or viewpoint changes.

# Visual Structure

- Canvas: 768×512 px PNG (white background, thin mid-board divider).
- Left region: Colored cards (initial positions). Right region: Outline slots.
- Layout variants:
  - `line`: cards/slots arranged in a straight column.
  - `staggered`: loose two-column ordering.
  - `grid`: 2×2 arrangement.
  - `scatter`: lightly randomized positions within constrained bounds.
- Shape library (currently 6 types):
  - `circle`, `square`, `triangle`, `star`, `hexagon`, `diamond`.
- Colors chosen from a palette of six high-contrast hues.
- Shape counts vary from 2–6 per question; higher counts imply tighter spacing and more complex layout.

# Prompt Template

Prompts are short, imperative-style instructions, e.g.:

```
Solve the flat shape sorter puzzle exactly as shown. Starting from the unsolved first frame, drag the colored cards across the board and place them into the matching outlines on the right. Match the blue circle, orange triangle, green square, and finally the red star card. Keep the board orientation unchanged and end once all outlines are packed tightly.
```

Key properties:
- Emphasize sliding motions, fixed top-down camera, and ending only when every slot is filled.
- Insert shape summary to remind the model which pieces to move.
- Same prompt works across difficulties; paraphrase variations can be added later.

# Data Files per Question

```
data/questions/shape_sorter_task/shape_sorter_XXXX/
├── first_frame.png
├── final_frame.png
├── prompt.txt
└── question_metadata.json
```

- `first_frame.png` – unsolved state (cards left, slots right).
- `final_frame.png` – solved state (all cards inside slots). Background is identical to the first frame for consistency.
- `prompt.txt` – single-line instruction (UTF-8).
- `question_metadata.json` – includes:
  - `task_id`, `domain`, `difficulty`
  - `shape_count`, `shape_library`, `layout_variant`
  - `shapes` array with per-piece shape, color, start/target coordinates, size
  - `colors`, `canvas_size`, `camera` info

# Generation Pipeline

`shape_sorter_reasoning.py` contains:

1. **Renderer**: draws first/final PNGs using matplotlib primitives (circles, polygons, regular polygons).
2. **ShapeSorterGenerator**:
   - Samples number of shapes based on difficulty.
   - Chooses layout variant (line/staggered/grid/scatter) and computes non-overlapping positions.
   - Samples shapes/colors, builds prompt and metadata, writes files.
3. **create_dataset**:
   - Accepts `num_samples`, `difficulties`, `seed`.
   - Loops to create question folders and returns the standard dataset dict with `pairs`.

# Difficulty Mapping (current)

| Difficulty | Shapes | Layout trend |
|------------|--------|--------------|
| easy       | 2–3    | line / staggered |
| medium     | 3–5    | staggered / grid |
| hard       | 5–6    | scatter (multi-row, jittered) |

Future improvements can fix deterministic shape counts per tier or add more layout templates.

# Integration

1. Generate samples: `python vmevalkit/runner/create_dataset.py --pairs-per-domain 10 --domains shape_sorter`
2. Register task inside `vmevalkit/runner/TASK_CATALOG.py`:
   ```python
   'shape_sorter': {
       'name': 'Shape Sorter',
       'description': '2D shape matching under fixed camera',
       'module': 'vmevalkit.tasks.shape_sorter_task',
       'create_function': 'create_dataset',
       'process_dataset': lambda dataset, num_samples: dataset['pairs']
   }
   ```
3. Include this markdown file in documentation references.

# Quality Checklist

- [ ] Prompt references correct shape list (no empty string).
- [ ] `.png` files exist and share the same background dimensions.
- [ ] Metadata includes `layout_variant`, `shape_count`, `shapes` array with matching coordinates.
- [ ] For each question, card positions and outline positions are non-overlapping and inside the visible regions.
- [ ] Task registered in `TASK_CATALOG` before running full pipeline.

