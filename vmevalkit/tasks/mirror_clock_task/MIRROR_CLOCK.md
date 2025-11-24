# Mirror Clock Reasoning Task

## Overview

The Mirror Clock Reasoning Task evaluates video generation models' ability to combine **spatial reasoning** (mirror transformations) with **temporal reasoning** (future time prediction) by determining what time a clock will show after advancing by a specified duration, starting from a mirrored clock image.

## Task Description

### Objective
Generate a video that demonstrates:
1. Understanding of mirror reflection (spatial reasoning)
2. Identifying the original time from the mirrored image
3. Calculating future time by adding a specified duration (temporal reasoning)

### Cognitive Concept
This task combines **two levels of reasoning**:

**Level 1: Spatial Reasoning**
- The input shows a clock viewed through a mirror (horizontally flipped)
- The model must understand this is a mirror reflection
- Numbers and hands are reversed left-right

**Level 2: Temporal Reasoning**
- After identifying the original time, add a specified time duration
- Calculate the resulting future time
- Display the answer

**Example**:
- Mirror shows: Flipped 3:00
- Prompt: "Add 2 hours"
- Steps: Recognize 3:00 → Add 2 hours → Answer: 5:00

### Input
- **First Frame**: A horizontally flipped clock (as seen in a mirror)
- **Text Prompt**: "This is a mirrored clock. If the original clock moves forward by [X hours/minutes], what time will it show?"

### Expected Output
- **Generated Video**: Animation showing:
  1. Recognition of the mirror reflection
  2. Identification of original time
  3. Adding the specified duration
  4. Final answer: the future time
- **Final Frame**: Clock showing the calculated future time

## Examples

### Easy Example
- **Mirrored Clock**: Shows flipped 3:00
- **Prompt**: "This is a mirrored clock. If the original clock moves forward by 2 hours, what time will it show?"
- **Reasoning Steps**:
  1. Recognize mirror flip → Original is 3:00
  2. Add 2 hours → 3:00 + 2:00 = 5:00
- **Answer**: 5:00

### Medium Example
- **Mirrored Clock**: Shows flipped 2:30
- **Prompt**: "After 1 hour and 30 minutes passes on the original clock, what will be the new time?"
- **Reasoning Steps**:
  1. Recognize mirror flip → Original is 2:30
  2. Add 1:30 → 2:30 + 1:30 = 4:00
- **Answer**: 4:00

### Hard Example
- **Mirrored Clock**: Shows flipped 3:25
- **Prompt**: "From this mirrored clock, determine the original time, then add 2 hours and 45 minutes. What is the result?"
- **Reasoning Steps**:
  1. Recognize mirror flip → Original is 3:25
  2. Add 2:45 → 3:25 + 2:45 = 6:10
- **Answer**: 6:10

## Difficulty Levels

### Easy
- **Starting Time**: Exact hours (e.g., 3:00, 6:00, 9:00)
- **Time Delta**: Full hours only (1-3 hours)
- **Cognitive Load**: Low - simple hour addition
- **Example**: 3:00 + 2 hours = 5:00
- **Distribution**: ~33% of dataset

### Medium
- **Starting Time**: Any time with minutes
- **Time Delta**: Hours with 30-minute intervals (e.g., 1:00, 1:30, 2:00)
- **Cognitive Load**: Medium - requires minute carryover
- **Example**: 2:30 + 1:30 = 4:00
- **Distribution**: ~33% of dataset

### Hard
- **Starting Time**: Any precise time
- **Time Delta**: Any combination (0-3 hours, 0-59 minutes)
- **Cognitive Load**: High - complex arithmetic with potential hour rollovers
- **Example**: 3:25 + 2:45 = 6:10 (requires minute carryover)
- **Distribution**: ~33% of dataset

## Cognitive Abilities Tested

1. **Spatial Reasoning**: Understanding horizontal mirror transformations
2. **Visual Processing**: Interpreting mirrored clock hand positions
3. **Mental Arithmetic**: Adding hours and minutes with carryover
4. **Multi-Step Reasoning**: Combining spatial and temporal reasoning
5. **Abstract Thinking**: Managing two transformations (mirror + time addition)
6. **Attention to Detail**: Precise calculations with minute-level accuracy

## Evaluation Criteria

### Success Metrics
- ✅ Correct future time displayed in final frame
- ✅ Demonstrates understanding of mirror transformation
- ✅ Shows correct time calculation (original + delta)
- ✅ Logical reasoning sequence visible in video

### Failure Cases
- ❌ Wrong future time shown
- ❌ Failed to recognize mirror transformation
- ❌ Incorrect time arithmetic
- ❌ Only shows one step (mirror OR time addition, not both)

## Dataset Statistics

- **Total Variations**: Unlimited (procedurally generated)
- **Default Dataset Size**: 50 tasks
- **Difficulty Distribution**:
  - Easy: ~33% (exact hours)
  - Medium: ~33% (5-minute intervals)
  - Hard: ~33% (any minute)
- **Time Range**: 12-hour clock (1:00 - 12:59)

## Usage

### Generate Dataset

```python
from vmevalkit.tasks.mirror_clock_task import create_dataset

# Generate 50 balanced tasks
dataset = create_dataset(num_samples=50, balanced=True, seed=42)

# Generate 100 random tasks
dataset = create_dataset(num_samples=100, balanced=False)
```

### Generate Single Task

```python
from vmevalkit.tasks.mirror_clock_task import create_single_task

# Create task with specific time
task = create_single_task(
    task_id="test_mirror_001",
    hours=3,
    minutes=15
)

print(f"Original: {task.mirror_clock_data['original_time']['formatted']}")
# Output:
# Original: 3:15
```

## File Structure

```
data/questions/mirror_clock_task/
├── mirror_clock_0000/
│   ├── first_frame.png          # Mirror view clock
│   ├── final_frame.png          # Actual time clock
│   ├── prompt.txt               # Text instruction
│   └── question_metadata.json   # Task metadata
├── mirror_clock_0001/
└── ...
```

## Metadata Schema

```json
{
  "original_time": {
    "hours": 3,
    "minutes": 0,
    "formatted": "3:00"
  },
  "time_delta": {
    "hours": 2,
    "minutes": 0,
    "formatted": "2 hours"
  },
  "future_time": {
    "hours": 5,
    "minutes": 0,
    "formatted": "5:00"
  },
  "difficulty": "easy",
  "task_type": "mirror_with_future_prediction"
}
```

## Implementation Notes

### Clock Rendering
- **Clock Face**: Traditional analog clock with numbers 1-12
- **Hour Hand**: Shorter, thicker hand (50% of radius)
- **Minute Hand**: Longer, thinner hand (70% of radius)
- **Visual Clarity**: Clear contrast, readable numbers, precise angles

### Task Logic
- **Step 1**: Draw original clock showing the starting time
- **Step 2**: Flip horizontally using PIL's `Image.FLIP_LEFT_RIGHT` (first frame)
- **Step 3**: Calculate future time by adding hours and minutes
- **Step 4**: Draw future clock (final frame)
- **Prompt**: Dynamically generated with specific time delta

### Reproducibility
- Use `seed` parameter for deterministic generation
- Consistent angle calculations across runs
- Balanced difficulty distribution with controlled randomization

## Educational Value

This task helps evaluate whether video models can:
1. **Combine Multiple Reasoning Types**: Spatial + temporal reasoning in one task
2. **Multi-Step Problem Solving**: Execute a sequence of transformations
3. **Abstract Representation**: Understand symbolic time representation
4. **Arithmetic with Context**: Apply time addition rules (60-minute hours, 24-hour days)

## Real-World Applications

- **Navigation**: Reading reversed signs/displays
- **Medical Imaging**: Interpreting mirrored scans
- **Design**: Understanding symmetry and reflection
- **Physics**: Mirror optics and transformations

## Version History

- **v1.0.0** (2025): Initial release with three difficulty levels
