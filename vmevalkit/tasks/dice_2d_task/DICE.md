# 2D Dice Opposite Face Reasoning Task

## Overview

The 2D Dice Opposite Face Reasoning Task evaluates video generation models' ability to perform **spatial reasoning** and **logical deduction** by determining the opposite face of a standard dice using the fundamental rule that opposite faces always sum to 7.

## Task Description

### Objective
Generate a video that demonstrates:
1. Understanding of standard dice rules (spatial reasoning)
2. Applying the opposite faces rule: face + opposite = 7
3. Calculating and displaying the correct opposite face

### Cognitive Concept
This task tests **logical deduction** with a spatial constraint:

**Core Rule**: On a standard 6-sided dice, opposite faces always sum to 7
- 1 is opposite to 6
- 2 is opposite to 5
- 3 is opposite to 4

**Example**:
- Shown face: 3 dots
- Rule: 3 + opposite = 7
- Answer: opposite = 4

### Input
- **First Frame**: A 2D rendered dice face showing a number (1-6)
- **Text Prompt**: "This dice face shows {number} dots. What number is on the opposite face?"

### Expected Output
- **Generated Video**: Animation showing:
  1. Recognition of the shown number
  2. Application of the opposite faces rule
  3. Calculation: shown + opposite = 7
  4. Final answer: the opposite face
- **Final Frame**: 2D dice face showing the opposite number

## Examples

### Easy Example
- **Shown Face**: ⚂ (3 dots)
- **Prompt**: "This dice face shows 3 dots. What number is on the opposite face?"
- **Reasoning Steps**:
  1. Identify shown face → 3
  2. Apply rule: 3 + opposite = 7
  3. Calculate: opposite = 7 - 3 = 4
- **Answer**: ⚃ (4 dots)

### Medium Example
- **Shown Face**: ⚀ (1 dot)
- **Prompt**: "Using the rule that opposite faces sum to 7, if one face shows 1, what is the opposite?"
- **Reasoning Steps**:
  1. Shown face = 1
  2. Rule: opposite faces sum to 7
  3. Calculate: opposite = 7 - 1 = 6
- **Answer**: ⚅ (6 dots)

### Hard Example
- **Shown Face**: ⚄ (5 dots)
- **Prompt**: "Given that opposite faces of a dice always sum to 7, and one face shows 5, what number appears on the opposite face?"
- **Reasoning Steps**:
  1. Identify constraint: sum = 7
  2. Shown face = 5
  3. Solve: 5 + x = 7 → x = 2
- **Answer**: ⚁ (2 dots)

## Difficulty Levels

### Easy
- **Task**: Direct opposite face question
- **Cognitive Load**: Low - simple subtraction (7 - shown)
- **Example**: "Show 3 → What is opposite?" → Answer: 4
- **Distribution**: ~33% of dataset

### Medium
- **Task**: Explicit rule application
- **Cognitive Load**: Medium - requires understanding the rule
- **Example**: "Using sum = 7 rule, show 2 → opposite?" → Answer: 5
- **Distribution**: ~33% of dataset

### Hard
- **Task**: Complex reasoning with constraints
- **Cognitive Load**: High - multi-step logical deduction
- **Example**: "Given constraint, show 6 → opposite?" → Answer: 1
- **Distribution**: ~33% of dataset

## Cognitive Abilities Tested

1. **Spatial Reasoning**: Understanding 3D object properties (opposite faces)
2. **Pattern Recognition**: Recognizing dice dot patterns (1-6)
3. **Logical Deduction**: Applying the opposite faces rule
4. **Arithmetic**: Simple addition/subtraction (sum to 7)
5. **Rule Application**: Using constraints to solve problems
6. **Visual Processing**: Interpreting 2D representations of 3D objects

## Evaluation Criteria

### Success Metrics
- ✅ Correct opposite face displayed in final frame
- ✅ Demonstrates understanding of the sum-to-7 rule
- ✅ Shows correct calculation process
- ✅ Logical reasoning sequence visible in video

### Failure Cases
- ❌ Wrong opposite face shown
- ❌ Failed to apply the sum-to-7 rule
- ❌ Incorrect arithmetic
- ❌ No reasoning process shown

## Dataset Statistics

- **Total Variations**: Limited to 6 unique face pairs (1↔6, 2↔5, 3↔4)
- **Default Dataset Size**: 50 tasks
- **Difficulty Distribution**:
  - Easy: ~33% (direct questions)
  - Medium: ~33% (rule-based)
  - Hard: ~33% (constraint-based)
- **Face Coverage**: All faces 1-6 represented equally

## Usage

### Generate Dataset

```python
from vmevalkit.tasks.dice_2d_task import create_dataset

# Generate 50 balanced tasks
dataset = create_dataset(num_samples=50, balanced=True, seed=42)

# Generate 100 random tasks
dataset = create_dataset(num_samples=100, balanced=False)
```

### Generate Single Task

```python
from vmevalkit.tasks.dice_2d_task import create_single_task

# Create task showing face 3
task = create_single_task(
    task_id="test_dice_001",
    shown_face=3,
    difficulty="easy"
)

print(f"Shown: {task.dice_data['shown_face']}")
print(f"Answer: {task.dice_data['answer_face']}")
# Output:
# Shown: 3
# Answer: 4
```

## File Structure

```
data/questions/dice_2d_task/
├── dice_0000/
│   ├── first_frame.png          # Shown dice face
│   ├── final_frame.png          # Opposite dice face
│   ├── prompt.txt               # Text instruction
│   └── question_metadata.json   # Task metadata
├── dice_0001/
└── ...
```

## Metadata Schema

```json
{
  "shown_face": 3,
  "answer_face": 4,
  "difficulty": "easy",
  "task_type": "direct_opposite",
  "rule": "opposite_faces_sum_to_7"
}
```

## Implementation Notes

### Dice Face Rendering
- **Face Size**: 300x300 pixels
- **Dot Patterns**: Standard dice patterns for 1-6
- **Visual Style**: Clean white face with black dots, rounded corners
- **Dot Size**: 14px radius for clarity

### Task Logic
- **Step 1**: Render shown face (1-6)
- **Step 2**: Calculate opposite using rule: opposite = 7 - shown
- **Step 3**: Render opposite face (answer)
- **Prompt**: Dynamically generated with shown number

### Dice Rules
- **Opposite Sum**: Always equals 7
- **Face Pairs**: (1,6), (2,5), (3,4)
- **Standard Dice**: Follows international dice conventions

### Reproducibility
- Use `seed` parameter for deterministic generation
- Balanced difficulty distribution available
- Consistent rendering across runs

## Educational Value

This task helps evaluate whether video models can:
1. **Understand 3D Constraints**: Apply rules about 3D objects from 2D views
2. **Logical Reasoning**: Use deductive reasoning with constraints
3. **Pattern Application**: Apply learned rules to new instances
4. **Arithmetic Reasoning**: Perform simple calculations in context

## Version History

- **v1.0.0** (2025): Initial release with three difficulty levels
