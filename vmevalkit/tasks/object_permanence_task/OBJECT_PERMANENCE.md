# Object Permanence Task Documentation

## Overview

The Object Permanence Task evaluates video generation models' ability to demonstrate **object permanence reasoning** - understanding that objects continue to exist and remain unchanged when occluded. This task tests whether models understand that objects maintain their properties (position, color, shape) even when temporarily hidden by an occluder.

This task introduces a cognitively appropriate benchmark where models must:
- Understand that objects exist even when occluded
- Maintain object properties when occluder moves
- Demonstrate object permanence reasoning

## Task Description

### Core Challenge

Models must:
1. **Understand Scene**: Recognize objects in the initial scene
2. **Process Occlusion**: Understand that occluder movement doesn't affect objects
3. **Maintain Consistency**: Keep objects completely unchanged (position, color, shape)
4. **Generate Transition**: Create video showing occluder moving while objects remain unchanged

### Visual Format

- **Canvas Size**: 256×256 pixels (default)
- **Objects**: 1-3 objects per scene (based on difficulty)
- **Object Types**: Colored shapes (cubes, spheres, pyramids, cones)
- **Colors**: Red, green, blue, yellow, orange, purple
- **Occluder**: Opaque gray panel that moves from left to right

## Design Philosophy

### Why Object Permanence?

**Key Concept**: Object permanence is a fundamental cognitive ability - understanding that objects continue to exist even when they cannot be seen.

**Rationale**:
- ✅ **Tests Core Cognitive Ability**: Evaluates understanding of object persistence
- ✅ **Real-World Relevance**: Models need to understand objects exist when occluded
- ✅ **Clear Success Criteria**: Objects must remain unchanged (easy to verify)
- ✅ **Psychology-Based**: Based on classic object permanence experiments

### Difficulty Levels

#### Easy
- **Objects**: 1 object
- **Occluder**: Moves from left to right, passes over object
- **Test Focus**: Single object permanence understanding
- **Example**: Red ball in center, gray panel moves from left to right

#### Medium
- **Objects**: 2 objects
- **Occluder**: Moves from left to right, passes over both objects
- **Test Focus**: Multiple object permanence understanding
- **Example**: Red ball and blue ball, gray panel moves from left to right

#### Hard
- **Objects**: 3 objects
- **Occluder**: Moves from left to right, passes over all objects
- **Test Focus**: Complex object permanence with multiple objects
- **Example**: Red ball, blue ball, and green ball, gray panel moves from left to right

## Visual Structure

### First Frame
- **Objects**: Fully visible, positioned in scene
- **Occluder**: On left side (off-screen or not occluding objects)
- **State**: Objects completely visible, occluder not blocking view

### Final Frame
- **Objects**: Exactly the same as first frame (position, color, shape unchanged)
- **Occluder**: Moved out of frame (completely off-screen on right side)
- **State**: Objects revealed, occluder gone

## Prompt Design

Prompts are designed to:
- **Describe Scene**: Tell model what objects are in the scene
- **Specify Action**: Ask to move occluder from current position to right until it exits frame
- **Not Explicit**: Don't explicitly state objects should remain unchanged (tests inference)

**Example Prompts**:
- Easy: "A red ball is in the center of the scene. A gray panel is on the left side. Move the panel from its current position to the right until it completely exits the frame."
- Medium: "A red ball and a blue ball are in the scene. A gray panel is on the left side. Move the panel from its current position to the right until it completely exits the frame."
- Hard: "A red ball, a blue ball, and a green ball are in the scene. A gray panel is on the left side. Move the panel from its current position to the right until it completely exits the frame."

## Cognitive Abilities Tested

1. **Object Permanence**: Understanding that objects exist when occluded
2. **Spatial Memory**: Remembering object positions when occluded
3. **Consistency Reasoning**: Maintaining object properties across occlusion
4. **Visual Inference**: Inferring that objects should remain unchanged

## Evaluation Criteria

### Success Metrics
- ✅ Objects remain in exact same positions (pixel-level accuracy)
- ✅ Object colors unchanged
- ✅ Object shapes unchanged
- ✅ Object sizes unchanged
- ✅ Occluder correctly moved out of frame

### Scoring (1-5 scale)
- **5**: Perfect - Objects completely unchanged, occluder correctly moved
- **4**: Mostly correct - Minor deviations in object properties
- **3**: Partially correct - Some objects changed or moved
- **2**: Mostly incorrect - Significant changes to objects
- **1**: Completely wrong - Objects changed or disappeared

## Data Files per Question

```
data/questions/object_permanence_task/object_permanence_easy_0000/
├── first_frame.png      # Objects visible, occluder on left
├── final_frame.png      # Objects unchanged, occluder out of frame
├── prompt.txt           # Instructions for model
└── question_metadata.json
```

## Technical Details

- **Domain**: `object_permanence`
- **Module**: `vmevalkit.tasks.object_permanence_task`
- **Create Function**: `create_dataset()`
- **Task ID Format**: `object_permanence_{difficulty}_{id:04d}`

## Implementation Notes

- Objects are positioned in middle-right area to ensure occluder path covers them
- Occluder starts off-screen on left (x < 0) to not occlude objects initially
- Occluder moves to right side (x > canvas_width) to exit frame
- Objects must remain exactly the same between first and final frames

## Related Tasks

- **Object Subtraction**: Tests selective removal (opposite concept)
- **VPCT**: Tests physical reasoning with occlusion
- **Shape Sorter**: Tests spatial reasoning without occlusion

