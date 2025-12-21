# Majority Color Task

## Overview

The Majority Color task evaluates whether a model can identify the most frequent
color in a scene and recolor every object to match that majority color. Object
positions, shapes, and sizes must remain unchanged.

## Task Structure

Each task pair contains:
- **First Frame**: Colored shapes with a unique majority color
- **Final Frame**: All objects recolored to the majority color
- **Prompt**: Instruction to recolor all objects to the most common color

## Object Setup

- **Canvas**: 256x256 pixels
- **Colors**: Red, green, blue, yellow, orange, purple
- **Shapes**: Cube (square), sphere (circle), pyramid (triangle), cone (trapezoid)
- **Placement**: Random placement with collision checks; grid fallback if needed

## Difficulty

Difficulty varies by number of objects and number of colors:
- **Easy**: 6-8 objects, 3 colors
- **Medium**: 8-10 objects, 3-4 colors
- **Hard**: 10-12 objects, 4-5 colors

The majority color is always strictly more than half of all objects.

## Data Structure

Each pair has the standard fields plus task-specific metadata:

```json
{
  "id": "majority_color_0001",
  "prompt": "...",
  "first_image_path": "path/to/first_frame.png",
  "final_image_path": "path/to/final_frame.png",
  "task_category": "MajorityColor",
  "majority_color_data": {
    "objects": [...],
    "majority_color": "red",
    "color_counts": {"red": 5, "green": 2, "blue": 1},
    "num_objects": 8,
    "num_colors": 3
  },
  "difficulty": "easy",
  "created_at": "2025-01-01T00:00:00Z"
}
```

## Usage

Generate tasks with:

```bash
python examples/create_questions.py --task majority_color --pairs-per-domain 50
```
