# Counting Objects Task

A unified task for counting different geometric shapes (circles, pentagons, and squares) in generated images. This task combines three separate counting tasks into a single, cohesive framework with difficulty levels.

## Overview

This task tests video generation models' ability to create videos that demonstrate counting various geometric objects. Each shape type has its own generation logic preserved from the original implementations, with added difficulty classification.

## Difficulty Levels

### Easy
- **Circles**: 5 black circles
- **Pentagons**: 5 black pentagons  
- **Squares**: 2 nested squares

### Medium
- **Circles**: 5 colored circles OR 6-7 circles (any color)
- **Pentagons**: 5 colored pentagons OR 6-7 pentagons (any color)
- **Squares**: 3 nested squares

### Hard
- **Circles**: 8-9 circles (any color)
- **Pentagons**: 8-9 pentagons (any color)
- **Squares**: 4-5 nested squares

## Shape Types

### 1. Circles
- **Count Range**: 5-9 circles per image
- **Variations**: 
  - Two radius sizes (r=5, r=10)
  - Black or multi-colored (tab10 colormap)
  - Various arrangements (odd/even count patterns)
- **Parameters**: 
  - DPI: 300 (high quality)
  - Line thickness: 1
  - Canvas size: 5x5 units

### 2. Pentagons
- **Count Range**: 5-9 pentagons per image
- **Variations**:
  - Two side lengths (r=5, r=10)
  - Black or multi-colored (tab10 colormap)
  - Various arrangements based on row patterns
- **Parameters**:
  - DPI: 300 (high quality)
  - Line thickness: 1
  - Canvas size: 5x5 units

### 3. Squares
- **Count Range**: Variable (depends on nesting depth)
- **Variations**:
  - Nesting depths: 2-5 levels
  - Random center positions
  - Random initial sizes (8-18 units)
  - Multiple line thicknesses (2, 3, 4)
- **Parameters**:
  - Reduction factor: 0.75
  - Padding: 0.75 units
  - Canvas size: 30x30 units

## Task Structure

Each sample includes:
- **First Frame**: Clean image showing all objects without annotations
- **Last Frame**: Same image with a "Total: N" text overlay indicating the count
- **Text Position**: Randomly varies between 'top', 'middle', and 'bottom'
- **Ground Truth**: Exact count of objects in the image

## Usage

### Generate All Shape Types

```python
from counting_objects import create_dataset

# Generate 10 samples per shape type (30 total)
dataset = create_dataset(num_samples=10)

print(f"Total samples: {dataset['total_samples']}")
print(f"Shape types: {dataset['shape_types']}")
```

### Generate Specific Difficulties

```python
# Generate only easy and medium difficulty samples
dataset = create_dataset(
    num_samples=10,
    difficulties=['easy', 'medium']
)

# Generate only hard samples
dataset = create_dataset(
    num_samples=10,
    difficulties=['hard']
)
```

### Generate Specific Shape Types

```python
# Generate only circles and pentagons
dataset = create_dataset(
    num_samples=15,
    shape_types=['circles', 'pentagons']
)

# Generate hard circles only
dataset = create_dataset(
    num_samples=20,
    shape_types=['circles'],
    difficulties=['hard']
)
```

### Generate Individual Shape Types

```python
from counting_objects import (
    create_circles_dataset,
    create_pentagons_dataset,
    create_squares_dataset
)

# Generate only circles
circles = create_circles_dataset(num_samples=20)

# Generate only pentagons
pentagons = create_pentagons_dataset(num_samples=20)

# Generate only squares
squares = create_squares_dataset(num_samples=20)
```

## Sample Data Structure

Each sample contains:

```python
{
    "sample_id": "circles_0001",              # Unique sample identifier
    "prompt": "Create a video to show...",     # Generation prompt
    "first_frame": "circles_1_first.png",      # Path to first frame
    "last_frame": "circles_1_last.png",        # Path to last frame
    "ground_truth_count": 7,                   # Correct object count
    "text_position": "top",                    # Location of text overlay
    "shape_type": "circles",                   # Type of shape
    "difficulty": "medium",                    # Difficulty level (easy/medium/hard)
    "metadata": {                              # Shape-specific metadata
        "diameter": 0.1,
        "centers": [[x1, y1], [x2, y2], ...],
        "dpi": 300,
        "linewidth": 1,
        ...
    },
    "id": "counting_circles_0000",            # VMEvalKit ID
    "domain": "counting_objects",              # Task domain
    "first_image_path": "/tmp/.../...",        # Absolute path to first frame
    "final_image_path": "/tmp/.../...",        # Absolute path to last frame
}
```

## Evaluation

Video generation models should:

1. **Start with the first frame** (no objects highlighted)
2. **Sequentially highlight or animate each object** to facilitate counting
3. **End with the last frame** showing the total count

Evaluation criteria:
- **Correctness**: Does the video accurately count all objects?
- **Clarity**: Are individual objects clearly distinguishable during counting?
- **Sequence**: Does the counting proceed in a logical order?
- **Accuracy**: Does the final count match the ground truth?

## Original Sources

This combined task preserves the original generation logic from:
- **Circles**: [create_circles.py](https://github.com/tin-xai/simple_task_video_reasoning/blob/main/CountingCircles/create_circles.py)
- **Pentagons**: [create_pentagons.py](https://github.com/tin-xai/simple_task_video_reasoning/blob/main/CountingCircles/create_pentagons.py)
- **Squares**: [create_squares.py](https://github.com/tin-xai/simple_task_video_reasoning/blob/main/CountingSquares/create_squares.py)

## Key Features

1. **Unified Interface**: Single `create_dataset()` function for all shape types
2. **Modular Design**: Separate functions for each shape type allow independent usage
3. **Preserved Logic**: All original generation algorithms remain intact
4. **Consistent Output**: Standardized data structure across all shape types
5. **Flexible Generation**: Generate all shapes or specific subsets as needed
6. **Difficulty Levels**: Each sample is classified as easy, medium, or hard based on objective criteria

## Difficulty Classification

Difficulty levels are automatically assigned based on the following criteria, and **only the requested difficulties are generated** (not filtered after generation):

### Circles & Pentagons
The difficulty is determined by a combination of:
- **Object count**: More objects = harder
- **Color complexity**: Multi-colored objects are harder than monochrome

| Difficulty | Count | Color |
|------------|-------|-------|
| Easy       | 5     | Any |
| Medium     | 5 OR 6-7 | Any |
| Hard       | 8-9   | Any |

### Squares
The difficulty is based solely on the nesting depth:

| Difficulty | Nesting Depth |
|------------|---------------|
| Easy       | 2 levels      |
| Medium     | 3 levels      |
| Hard       | 4-5 levels    |

### Efficient Generation

When you request specific difficulties, the generator **only creates samples matching those difficulties**:

- `difficulties=['easy']` → Only generates 5 black objects and 2-level nests
- `difficulties=['hard']` → Only generates 8-9 objects and 4-5-level nests  
- `difficulties=['easy', 'hard']` → Skips medium entirely

This is more efficient than generating everything and filtering afterwards.

### Filtering by Difficulty

You can filter generated samples by difficulty:

```python
# Generate only easy samples
easy_dataset = create_dataset(num_samples=10, difficulties=['easy'])

# Generate easy and hard (skip medium)
dataset = create_dataset(num_samples=10, difficulties=['easy', 'hard'])

# Generate all difficulties (default)
dataset = create_dataset(num_samples=10)  # or difficulties=['easy', 'medium', 'hard']
```

## Parameters Summary

| Shape Type | Count Range | DPI | Thickness | Canvas Size | Color Options |
|------------|-------------|-----|-----------|-------------|---------------|
| Circles    | 5-9         | 300 | 1         | 5×5         | Black/Multi   |
| Pentagons  | 5-9         | 300 | 1         | 5×5         | Black/Multi   |
| Squares    | 2-5 (depth) | -   | 2-4       | 30×30       | Black only    |

## Implementation Notes

- **High Quality**: Uses DPI=300 for circles and pentagons for clear, high-resolution images
- **No Overlap**: Careful positioning ensures objects don't overlap
- **Reproducibility**: All generation parameters are stored in metadata
- **Text Overlay**: Final frame includes count annotation at random position
- **Temporary Storage**: Images saved to temporary directory managed by `tempfile`

## Example Scenarios

### Easy Cases
- 5 black circles in regular arrangement
- 2 nested squares with clear spacing

### Hard Cases
- 9 multi-colored circles in dense arrangement
- 5 nested squares with small differences in size
- Overlapping pentagon arrangements

## Dependencies

```python
matplotlib>=3.5.0
numpy>=1.20.0
```

## Citation

If you use this task in your research, please cite:

```
Tin's Simple Task Video Reasoning
https://github.com/tin-xai/simple_task_video_reasoning
```

