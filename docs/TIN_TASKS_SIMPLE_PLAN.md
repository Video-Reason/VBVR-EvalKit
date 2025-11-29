# Tin's Tasks Integration - Simplified Plan

**Philosophy**: Use Tin's original code with MINIMAL modifications. Just wrap it with VMEvalKit interface.

---

## 🎯 Core Strategy

Tin's code already does everything we need:
- ✅ Generates first_frame and last_frame images
- ✅ Creates metadata JSON with prompts and ground truth
- ✅ Has proper image generation logic
- ✅ Includes all necessary parameters

**We just need to**:
1. Copy his scripts into VMEvalKit task folders
2. Wrap with `create_dataset()` function
3. Change output from fixed directories to tempfile
4. Return VMEvalKit format (dict with 'pairs' key)

---

## 📦 Four Tasks from Tin's Repo

| Task Name | Source File | What to Keep | Lines of Code |
|-----------|-------------|--------------|---------------|
| **Counting Circles** | `CountingCircles/create_circles.py` | Everything | ~250 |
| **Counting Pentagons** | `CountingCircles/create_pentagons.py` | Everything | ~230 |
| **Counting Squares** | `CountingSquares/create_squares.py` | Everything | ~148 |
| **Letter Counting** | `FindingWords/create_strings.py` | Everything | ~159 |
| **Subway Pathfinding** | `PathFinding/create_subway.py` | Everything | ~307 |

---

## 🔧 Minimal Modifications Required

### Change #1: Wrap in function
**Before** (Tin's code):
```python
# Direct script execution
test_samples = []
for thickness in [0.5, 1]:
    for d in dpi:
        # ... generation logic ...
        test_samples.append(test_sample)

# Save to file
with open("./OlympicLikeLogo/test_samples.json", "w") as fp:
    json.dump(test_samples, fp, indent=2)
```

**After** (VMEvalKit):
```python
def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """Generate counting circles dataset using Tin's original logic."""
    test_samples = []
    
    for thickness in [0.5, 1]:
        for d in dpi:
            # ... EXACT SAME generation logic from Tin's code ...
            test_samples.append(test_sample)
            
            if num_samples and len(test_samples) >= num_samples:
                break
        if num_samples and len(test_samples) >= num_samples:
            break
    
    return {
        "name": "counting_circles_tasks",
        "pairs": test_samples  # Use Tin's exact data structure
    }
```

### Change #2: Use tempfile instead of fixed directories
**Before**:
```python
os.makedirs("./OlympicLikeLogo", exist_ok=True)
filepath = "./OlympicLikeLogo/" + filename + '.png'
```

**After**:
```python
import tempfile
temp_dir = tempfile.mkdtemp()
filepath = os.path.join(temp_dir, filename + '.png')
```

### Change #3: Keep Tin's data structure
**No changes needed!** His structure already has:
- `sample_id`
- `prompt`
- `first_frame` / `last_frame`
- `ground_truth_count` (or equivalent)
- `metadata`

Just add `domain` field:
```python
test_sample = {
    "sample_id": f"sample_{sample_idx + 1:04d}",
    "prompt": f"Create a video to show how to count the number of circles",
    "first_frame": f"{first_frame_id}.png",
    "last_frame": f"{last_frame_id}.png",
    "ground_truth_count": num,
    "domain": "counting_circles",  # ADD THIS LINE ONLY
    "text_position": text_pos,
    "metadata": {
        # ... Tin's original metadata ...
    }
}
```

---

## 📁 Implementation Plan

### Task 1: Counting Circles
```
vmevalkit/tasks/counting_circles_task/
├── __init__.py                  # Export create_dataset
├── counting_circles.py          # Copy from Tin's create_circles.py + minimal wrapper
└── COUNTING_CIRCLES.md          # Brief description
```

**Action**: 
1. Copy `create_circles.py` → `counting_circles.py`
2. Move all generation code into `create_dataset()` function
3. Change `"./OlympicLikeLogo/"` → `tempfile.mkdtemp()`
4. Add `"domain": "counting_circles"` to each test_sample
5. Return `{"name": "...", "pairs": test_samples}`

### Task 2: Counting Pentagons
```
vmevalkit/tasks/counting_pentagons_task/
├── __init__.py
├── counting_pentagons.py        # Copy from Tin's create_pentagons.py + minimal wrapper
└── COUNTING_PENTAGONS.md
```

**Action**: Same as Task 1, but with pentagons

### Task 3: Counting Squares
```
vmevalkit/tasks/counting_squares_task/
├── __init__.py
├── counting_squares.py          # Copy from Tin's create_squares.py + minimal wrapper
└── COUNTING_SQUARES.md
```

**Action**: Same pattern - wrap Tin's code

### Task 4: Letter Counting
```
vmevalkit/tasks/letter_counting_task/
├── __init__.py
├── letter_counting.py           # Copy from Tin's create_strings.py + minimal wrapper
└── LETTER_COUNTING.md
```

**Action**: Same pattern

### Task 5: Subway Pathfinding
```
vmevalkit/tasks/subway_pathfinding_task/
├── __init__.py
├── subway_pathfinding.py        # Copy from Tin's create_subway.py + minimal wrapper
└── SUBWAY_PATHFINDING.md
```

**Action**: Same pattern

---

## 📝 Detailed Example: Counting Circles

### File: `vmevalkit/tasks/counting_circles_task/__init__.py`
```python
from .counting_circles import create_dataset
__all__ = ['create_dataset']
```

### File: `vmevalkit/tasks/counting_circles_task/counting_circles.py`
```python
"""
Counting Circles Task - Adapted from Tin's simple_task_video_reasoning
Original: https://github.com/tin-xai/simple_task_video_reasoning/blob/main/CountingCircles/create_circles.py

Minimal modifications to fit VMEvalKit interface.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import numpy as np
import random
import json
from matplotlib import cm
import os
import tempfile
from typing import Dict, Any

# ============================================
# Tin's Original Functions (UNCHANGED)
# ============================================

def hue_to_rgb(hue):
    rgb = hsv_to_rgb([hue, 1, 1])
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))

def get_colors_from_colormap(colormap_name, num_colors):
    colormap = cm.get_cmap(colormap_name, num_colors)
    colors = [colormap(i) for i in range(num_colors)]
    return colors

def draw_circles(dpi, size, radius, centers, colors, thickness, add_text=False, 
                 total_count=None, text_position='top', filename=None, output_dir=None):
    """Tin's original draw_circles function with output_dir parameter added."""
    
    assert len(centers) == len(colors)
    h=5
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, h)
    ax.set_ylim(0, h)
    ax.axis("off")

    for center, color in zip(centers, colors):
        circle1_plot = plt.Circle((center[0] * h, center[1] * h), radius * h, 
                                  color=color, fill=False, linewidth=thickness)
        ax.add_artist(circle1_plot)

    # Add text if requested (for last frame)
    if add_text and total_count is not None:
        text_str = f"Total: {total_count}"
        fontsize = 20
        if text_position == 'top':
            ax.text(h/2, h * 0.95, text_str, fontsize=fontsize, ha='center', va='top', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        elif text_position == 'bottom':
            ax.text(h/2, h * 0.05, text_str, fontsize=fontsize, ha='center', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:  # middle
            ax.text(h/2, h/2, text_str, fontsize=fontsize, ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    filepath = os.path.join(output_dir, filename + '.png')
    fig.savefig(filepath, bbox_inches='tight', dpi=dpi, pad_inches=0)
    plt.close(fig)
    return filename

# ============================================
# VMEvalKit Wrapper
# ============================================

def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    """
    Generate counting circles dataset using Tin's original generation logic.
    
    Args:
        num_samples: Number of samples to generate (None = generate all variations)
        
    Returns:
        Dataset dictionary with 'pairs' key containing task data
    """
    
    # Create temp directory for images
    temp_dir = tempfile.mkdtemp()
    
    # Tin's original parameters
    size = 500
    dpi = [100, 200, 300]
    num_circles = [5, 6, 7, 8, 9]
    dist = 0.1
    
    test_samples = []
    text_positions = ['top', 'middle', 'bottom']
    sample_idx = 0

    # ============================================
    # Tin's Original Generation Logic (UNCHANGED)
    # ============================================
    
    for thickness in [0.5, 1]:
        for d in dpi:
            for r in [5, 10]:
                rad = 0.5 / r
                for num in num_circles:
                    for colors in [['black'] * num, get_colors_from_colormap('tab10', num)]:
                        
                        if num % 2 != 0:
                            centers = []
                            row_1 = (num + 1) // 2
                            row_2 = row_1 - 1
                            
                            y = 0.6
                            x = 0.5
                            
                            ratio = dist * rad
                            min_dist = rad * 2.0 + ratio
                            
                            if row_1 * rad * 2 + row_2 * ratio >= 1:
                                continue
                            
                            # Tin's original center calculation logic
                            if row_1 == 3:
                                centers.append([x, y])
                                centers.append([x - min_dist, y])
                                centers.append([x + min_dist, y])
                                centers.append([x - rad - ratio/2, y - rad])
                                centers.append([x + rad + ratio/2, y - rad])
                            elif row_1 == 5:
                                centers.append([x, y])
                                centers.append([x - min_dist, y])
                                centers.append([x + min_dist, y])
                                centers.append([x - 2 * min_dist, y])
                                centers.append([x + 2 * min_dist, y])
                                centers.append([x - rad - ratio / 2, y - rad])
                                centers.append([x + rad + ratio / 2, y - rad])
                                centers.append([x - rad - ratio - min_dist, y - rad])
                                centers.append([x + rad + ratio + min_dist, y - rad])
                            elif row_1 == 2:
                                centers.append([x - rad - ratio/2, y])
                                centers.append([x + rad + ratio/2, y])
                                centers.append([x, y - rad])
                            else:
                                centers.append([x - rad - ratio/2, y])
                                centers.append([x + rad + ratio/2, y])
                                centers.append([x - rad - ratio/2 - min_dist, y])
                                centers.append([x + rad + ratio/2 + min_dist, y])
                                centers.append([x, y - rad])
                                centers.append([x + min_dist, y - rad])
                                centers.append([x - min_dist, y - rad])
                            
                            # Generate frames using Tin's logic
                            text_pos = text_positions[sample_idx % len(text_positions)]
                            first_frame_id = draw_circles(d, size, rad, centers, colors, thickness, 
                                                         add_text=False, 
                                                         filename=f"{sample_idx + 1}_first",
                                                         output_dir=temp_dir)
                            
                            last_frame_id = draw_circles(d, size, rad, centers, colors, thickness, 
                                                        add_text=True, 
                                                        total_count=num, text_position=text_pos,
                                                        filename=f"{sample_idx + 1}_last",
                                                        output_dir=temp_dir)
                            
                            # Tin's original data structure + domain field
                            test_sample = {
                                "sample_id": f"sample_{sample_idx + 1:04d}",
                                "id": f"counting_circles_{sample_idx:04d}",  # VMEvalKit ID
                                "prompt": f"Create a video to show how to count the number of circles",
                                "first_frame": f"{first_frame_id}.png",
                                "last_frame": f"{last_frame_id}.png",
                                "first_image_path": os.path.join(temp_dir, f"{first_frame_id}.png"),
                                "final_image_path": os.path.join(temp_dir, f"{last_frame_id}.png"),
                                "ground_truth_count": num,
                                "domain": "counting_circles",  # Added for VMEvalKit
                                "task_category": "Visual Counting",  # Added for VMEvalKit
                                "text_position": text_pos,
                                "metadata": {
                                    "diameter": rad * 2,
                                    "centers": centers,
                                    "distance": dist,
                                    "dpi": d,
                                    "canvas_size": 5.0,
                                    "linewidth": thickness,
                                    "colors": [c if isinstance(c, str) else f"rgba{tuple(c)}" for c in colors]
                                }
                            }
                            test_samples.append(test_sample)
                            sample_idx += 1
                            
                            # Check if we've reached requested number
                            if num_samples and len(test_samples) >= num_samples:
                                break
                        
                        else:
                            # Tin's even number logic (abbreviated for brevity - keep ALL original logic)
                            row_1 = num // 2
                            row_2 = row_1
                            y = 0.6
                            x = 0.5
                            ratio = dist * rad
                            min_dist = rad * 2.0 + ratio
                            
                            if row_2 * min_dist + 2 * rad >= 1:
                                continue
                                
                            for i in range(2):
                                centers = []
                                # ... (keep all of Tin's logic here) ...
                                
                                if num_samples and len(test_samples) >= num_samples:
                                    break
                        
                        if num_samples and len(test_samples) >= num_samples:
                            break
                    if num_samples and len(test_samples) >= num_samples:
                        break
                if num_samples and len(test_samples) >= num_samples:
                    break
            if num_samples and len(test_samples) >= num_samples:
                break
        if num_samples and len(test_samples) >= num_samples:
            break
    
    return {
        "name": "counting_circles_tasks",
        "pairs": test_samples,
        "source": "tin_tasks",
        "total_samples": len(test_samples)
    }
```

---

## 📋 Registration in TASK_CATALOG.py

Add these entries (keep Tin's task names):

```python
DOMAIN_REGISTRY = {
    # ... existing tasks ...
    
    'counting_circles': {
        'name': 'Counting Circles',
        'description': 'Count circles in Olympic-like arrangements (Tin\'s task)',
        'module': 'vmevalkit.tasks.counting_circles_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    
    'counting_pentagons': {
        'name': 'Counting Pentagons',
        'description': 'Count pentagons in arranged patterns (Tin\'s task)',
        'module': 'vmevalkit.tasks.counting_pentagons_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    
    'counting_squares': {
        'name': 'Counting Nested Squares',
        'description': 'Count all squares including nested ones (Tin\'s task)',
        'module': 'vmevalkit.tasks.counting_squares_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    
    'letter_counting': {
        'name': 'Letter Counting',
        'description': 'Count letter occurrences in words (Tin\'s task)',
        'module': 'vmevalkit.tasks.letter_counting_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    
    'subway_pathfinding': {
        'name': 'Subway Pathfinding',
        'description': 'Navigate through subway networks (Tin\'s task)',
        'module': 'vmevalkit.tasks.subway_pathfinding_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
}
```

---

## 🎯 Evaluation Guidance

Add to `gpt4o_eval.py` and `internvl.py` (simple scoring based on Tin's ground truth):

```python
TASK_GUIDANCE = {
    # ... existing ...
    
    'counting_circles': {
        'description': 'Count circles - Tin\'s task',
        'scoring': 'Award 1 point if count matches ground_truth_count, 0 otherwise.'
    },
    
    'counting_pentagons': {
        'description': 'Count pentagons - Tin\'s task',
        'scoring': 'Award 1 point if count matches ground_truth_count, 0 otherwise.'
    },
    
    'counting_squares': {
        'description': 'Count nested squares - Tin\'s task',
        'scoring': 'Award 1 point if count matches ground_truth_count, 0 otherwise.'
    },
    
    'letter_counting': {
        'description': 'Count letter occurrences - Tin\'s task',
        'scoring': 'Award 1 point if count matches ground_truth_count, 0 otherwise.'
    },
    
    'subway_pathfinding': {
        'description': 'Navigate subway paths - Tin\'s task',
        'scoring': 'Award 1 point if reaches destination_station, 0 otherwise.'
    },
}
```

---

## ✅ Implementation Checklist

For each task:
- [ ] Create module directory
- [ ] Copy Tin's original `.py` file
- [ ] Wrap generation code in `create_dataset()` function
- [ ] Change output directory to `tempfile.mkdtemp()`
- [ ] Add `"domain"` field to each sample
- [ ] Add `"id"`, `"first_image_path"`, `"final_image_path"` fields
- [ ] Create `__init__.py` with export
- [ ] Add entry to `TASK_CATALOG.py`
- [ ] Add evaluation guidance
- [ ] Test: `python vmevalkit/runner/create_dataset.py --domains {task} --pairs-per-domain 5`

---

## 🚀 Quick Implementation

**Estimated time**: 2-3 hours for all 5 tasks (mostly copy-paste + small modifications)

**Steps**:
1. Copy Tin's 5 scripts
2. Add wrapper function to each
3. Change directory paths
4. Add domain fields
5. Register in catalog
6. Test

**That's it!** No complex Pydantic models, no rewriting image generation, no custom difficulty levels. Just use Tin's excellent code as-is.

---

## 📊 Expected Output

Using Tin's generation parameters:
- **Counting Circles**: ~160 samples (if generating all)
- **Counting Pentagons**: ~12 samples
- **Counting Squares**: ~120 samples
- **Letter Counting**: ~1000+ samples (many words × letters × DPI)
- **Subway Pathfinding**: ~180 samples

Can use `num_samples` parameter to limit generation for testing.

