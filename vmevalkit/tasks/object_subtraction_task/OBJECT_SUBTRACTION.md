# Object Subtraction Task Documentation

## Overview

The Object Subtraction Task evaluates video generation models' ability to demonstrate **selective attention**, **inhibitory control**, and **causal consistency** by generating videos that show the removal of specific objects while keeping others stationary.

This task introduces a scalable, cognitively layered reasoning benchmark where models must:
- Understand selective removal instructions
- Remove specific objects based on explicit or abstract rules
- Keep non-target objects unchanged (do not do anything to other objects)
- Generate physically plausible transitions

## Task Description

### Core Challenge

Models must:
1. **Parse Instructions**: Understand which objects to remove based on the prompt
2. **Selective Removal**: Remove only the specified objects
3. **Spatial Invariance**: Do not do anything to other objects (keep them unchanged)
4. **Generate Transition**: Create smooth video showing objects being removed

### Visual Format

- **Canvas Size**: 256x256 pixels (default)
- **Objects**: 5-8 objects per scene (default)
- **Object Types**: Colored shapes (cubes, spheres, pyramids, cones)
- **Colors**: Red, green, blue, yellow, orange, purple
- **Shapes**: Cube (square), sphere (circle), pyramid (triangle), cone (trapezoid)

## Cognitive Levels

### Level 1: Explicit Specificity ✅ (Implemented)

**Task Type:**  
Remove objects defined by **explicit visual attributes** (color or shape only).

**Prompt Examples:**
- "Remove all {color} objects from the scene. Do not do anything to other objects." (颜色: red, green, blue, yellow, orange, purple)
- "Remove all {shape} objects from the scene. Do not do anything to other objects." (形状: cube, sphere, pyramid, cone)

**Example Scene:**
- **First Frame**: White background with 5-7 colored shapes (red cubes, green spheres, blue pyramids)
- **Final Frame**: Only non-red objects remain, identical positions

**Rule Structure:**
```python
{
  "level": "L1",
  "rule_type": "color",  # or "shape"
  "remove_color": "red",  # or "remove_shape": "cube"
  "target_object_ids": [0, 1]  # Explicit object IDs to remove
}
```

**Object Size:**  
All objects in L1 have uniform size (30 pixels) to focus on color and shape attributes only.

**Cognitive Focus:**  
Visual recognition · Simple selection · Static invariance

### Level 2: Enumerated Selection ✅ (Implemented)

**Task Type:**  
Remove multiple **explicitly listed** objects by color and shape.

**Prompt Examples:**
- "Remove the red cube, the green sphere, and the blue pyramid from the scene. Do not do anything to other objects."
- "Remove the orange pyramid and the red cone from the scene. Do not do anything to other objects."

### Level 3: Relational Reference ✅ (Implemented)

**Task Type:**  
Remove objects using **spatial relations** (edge-based positions) instead of explicit labels.

**Prompt Examples:**
- "Remove the leftmost object. Do not do anything to other objects."
- "Remove the {N} leftmost objects. Do not do anything to other objects."
- "Remove the rightmost object. Do not do anything to other objects."
- "Remove the {N} rightmost objects. Do not do anything to other objects."
- "Remove the topmost object. Do not do anything to other objects."
- "Remove the {N} topmost objects. Do not do anything to other objects."
- "Remove the bottommost object. Do not do anything to other objects."
- "Remove the {N} bottommost objects. Do not do anything to other objects."

### Level 4: Conceptual Abstraction ✅ (Implemented)

**Task Type:**  
Remove objects based on **semantic or conceptual properties** (outlier detection).

**Prompt Examples:**
- "Remove the object that looks different from the others. Do not do anything to other objects."

**Note:** All L4 tasks use the same unified prompt, regardless of the specific type of outlier (combination, shape consistency, or color consistency). This makes the task more abstract - the model needs to figure out what makes the object different.

## Data Structure

### ObjectSubtractionTaskPair

Each task consists of:
```python
{
    "id": "object_subtraction_l1_0001",
    "prompt": "Remove all red objects...",
    "first_image_path": "path/to/first_frame.png",
    "final_image_path": "path/to/final_frame.png",
    "task_category": "ObjectSubtraction",
    "level": "L1",
    "object_subtraction_data": {
        "objects": [...],  # All objects with id, color, shape, x, y, size, area
        "rule": {...},     # Rule definition
        "remove_object_ids": [0, 1],
        "keep_object_ids": [2, 3, 4],
        "num_objects": 5,
        "num_removed": 2,
        "num_kept": 3
    },
    "difficulty": "easy",
    "created_at": "2025-01-XX..."
}
```

## Implementation Details

### Object Generation

- **Collision Detection**: Ensures objects don't overlap
- **Grid Fallback**: If random placement fails, uses grid-based layout
- **Deterministic**: Uses seeds for reproducibility
- **Uniform Size**: All objects across all levels (L1, L2, L3, L4) have uniform size (30 pixels) to focus on other attributes (color, shape, position, conceptual properties)

### Rule Generation

**Level 1:**
- **Color-based**: Selects a color and finds all objects with that color
- **Shape-based**: Selects a shape and finds all objects with that shape
- **Uniqueness**: Each rule explicitly lists `target_object_ids` for unambiguous removal
- **Object Size**: All objects have uniform size (30 pixels) to focus on color and shape attributes

**Level 2:**
- **Enumerated Selection**: Removes 1-3 explicitly listed objects by color and shape combination
- **Object Size**: All objects have uniform size (30 pixels)

**Level 3:**
- **Spatial Relations**: Removes objects based on edge-based spatial positions (leftmost, rightmost, topmost, bottommost only)
- **Object Size**: All objects have uniform size (30 pixels)

**Level 4:**
- **Combination Outlier Detection**: Removes the object that looks different from others (based on color+shape combination majority)
- **Shape Consistency Outlier**: Removes the object with different shape (majority has same shape but different colors)
- **Color Consistency Outlier**: Removes the object with different color (majority has same color but different shapes)
- **Object Size**: All objects have uniform size (30 pixels)
- **Unified Prompt**: All L4 tasks use the same prompt regardless of outlier type

### Image Rendering

- **Library**: matplotlib
- **Shapes**:
  - Cube: Rectangle
  - Sphere: Circle
  - Pyramid: Equilateral triangle
  - Cone: Trapezoid
- **Colors**: Standard color mapping (red, green, blue, yellow, orange, purple)

## Usage

### Generate Dataset

```python
from vmevalkit.tasks.object_subtraction_task import create_dataset

# Generate 50 tasks (Level 1 only)
dataset = create_dataset(num_samples=50, levels=["L1"])

# Generate tasks for multiple levels
dataset = create_dataset(num_samples=100, levels=["L1", "L2", "L3", "L4"])

# Generate tasks for all levels with deterministic seed
dataset = create_dataset(num_samples=100, levels=["L1", "L2", "L3", "L4"], random_seed=42)
```

### Command Line

```bash
# Generate questions using the standard VMEvalKit script
python examples/create_questions.py --task object_subtraction --pairs-per-domain 50
```

## Evaluation Metrics (Planned)

| Metric                  | Description                                          |
| ----------------------- | ---------------------------------------------------- |
| final_object_match      | IoU overlap between generated and target final frame |
| removed_object_count    | Correct number of removed items                      |
| kept_object_stability   | Avg. displacement ≤ 3 px                             |
| motion_continuity       | Smooth optical flow (no teleportation)               |
| rule_accuracy          | Removed items satisfy logical rule in metadata       |

## Integration with VMEvalKit

### Domain Registry

Registered in `vmevalkit/runner/TASK_CATALOG.py`:
```python
'object_subtraction': {
    'name': 'Object Subtraction',
    'description': 'Selective object removal with multi-level cognitive reasoning',
    'module': 'vmevalkit.tasks.object_subtraction_task',
    'create_function': 'create_dataset',
    'process_dataset': lambda dataset, num_samples: dataset['pairs']
}
```

### Output Structure

Each generated question follows the standard VMEvalKit format:
```
data/questions/object_subtraction_task/{task_id}/
├── first_frame.png
├── final_frame.png
├── prompt.txt
└── question_metadata.json
```

## Future Extensions

- [x] Implement Level 2 (Enumerated Selection)
- [x] Implement Level 3 (Relational Reference)
- [x] Implement Level 4 (Conceptual Abstraction)
- [ ] Add evaluation metrics
- [ ] Support for more object types and colors
- [ ] Configurable canvas sizes
- [ ] Animation styles (fade out, slide out, etc.)

## References

- [GitHub Issue #54](https://github.com/hokindeng/VMEvalKit/issues/54) - Original task proposal



