# Perspective Taking Task

## Overview

The **Perspective Taking** task evaluates video models' ability to understand spatial relationships, transform viewpoints, and reason about what can be seen from different perspectives. This task tests the fundamental cognitive capability of taking another agent's point of view and generating the corresponding visual scene.

## Task Description

Given an initial scene showing an agent and objects from a random viewing angle, the model must:
1. **Understand spatial relationships** between the agent and objects in the scene
2. **Identify the agent's viewing direction** based on their position and orientation
3. **Transform the viewpoint** to show the scene from behind the agent
4. **Generate a new scene** that shows:
   - The agent's back, head, and posture silhouette
   - The environment that would appear in front of the agent
   - Proper spatial arrangement of visible objects

## Cognitive Skills Tested

- **Spatial Reasoning**: Understanding 3D positions and relationships
- **Perspective Transformation**: Converting between different viewpoints
- **Theory of Mind**: Reasoning about what another agent can see
- **Visual Scene Understanding**: Interpreting complex spatial scenes
- **Viewpoint Synthesis**: Generating coherent scenes from transformed perspectives

## Dataset Structure

Each perspective taking task consists of:

```
data/perspective_taking/{task_id}/
├── first_frame.png          # Initial scene from random angle
├── final_frame.png          # Target scene from behind agent
├── prompt.txt               # Task instructions
└── scene_metadata.json      # Scene configuration data
```

### Scene Metadata

The metadata includes detailed information about:
- **Objects**: Names, positions, scales, and visibility
- **Character**: Position, orientation, and viewing direction
- **Camera**: Configuration for both viewpoints
- **Visibility**: Which objects are visible to the character

## Task Examples

### Example 1: Basic Perspective Taking
- **Initial View**: Random angle showing agent and 3 objects
- **Agent Position**: Center of scene, facing forward
- **Task**: Generate view from behind agent showing objects in front
- **Key Challenge**: Correctly position objects relative to agent's viewing direction

### Example 2: Complex Spatial Arrangement
- **Initial View**: Oblique angle with multiple objects at varying distances
- **Agent Position**: Off-center, rotated
- **Task**: Transform to behind-agent view with proper depth ordering
- **Key Challenge**: Maintain correct spatial relationships during transformation

## Difficulty Levels

The tasks vary in complexity based on:
- **Number of objects**: 1-5 objects in scene
- **Spatial arrangement**: Simple vs. complex layouts
- **Agent orientation**: Cardinal directions vs. arbitrary angles
- **Occlusion**: Objects behind vs. in front of agent

## Evaluation Criteria

A successful video generation should:
1. **Show agent's back**: Clearly visible agent silhouette from behind
2. **Correct viewpoint**: Camera positioned behind agent, looking forward
3. **Proper object placement**: Objects visible to agent appear in correct positions
4. **Spatial consistency**: Distances and angles preserved from transformation
5. **Visual coherence**: Smooth transition maintaining scene identity

## Technical Details

### Data Format

Each task pair follows the VMEvalKit standard format:

```python
{
    'id': 'perspective_taking_0001',
    'domain': 'perspective_taking',
    'task_category': 'spatial_reasoning',
    'prompt': 'Rotate the current image around the center point...',
    'first_image_path': 'data/perspective_taking/1/first_frame.png',
    'final_image_path': 'data/perspective_taking/1/final_frame.png',
    'difficulty': 'medium',
    'perspective_taking_data': {
        'task_number': 1,
        'metadata': { /* scene configuration */ }
    }
}
```

### Loading the Dataset

```python
from vmevalkit.tasks.perspective_taking_task import create_dataset

# Load all perspective taking tasks
dataset = create_dataset()

# Load specific number of tasks
dataset = create_dataset(num_samples=10)

# Access task pairs
pairs = dataset['pairs']
for pair in pairs:
    print(f"Task {pair['id']}: {pair['prompt']}")
```

## Usage in VMEvalKit

### Generate Dataset

```bash
# Include perspective taking in dataset generation
python -m vmevalkit.runner.create_dataset --pairs-per-domain 25

# Read existing dataset
python -m vmevalkit.runner.create_dataset --read-only
```

### Run Inference

```bash
# Run inference on perspective taking tasks
python -m vmevalkit.runner.inference \
    --model luma-ray-2 \
    --domains perspective_taking \
    --experiment_name my_experiment
```

### Evaluate Results

```bash
# Evaluate with GPT-4O
python -m vmevalkit.runner.score \
    --model luma-ray-2 \
    --domains perspective_taking \
    --evaluator gpt4o
```

## Research Applications

This task is valuable for:
- **Spatial AI Research**: Understanding how models handle viewpoint transformations
- **Embodied AI**: Testing models for robot navigation and planning
- **Theory of Mind**: Evaluating perspective-taking capabilities
- **3D Scene Understanding**: Assessing spatial reasoning in video models
- **Human-AI Interaction**: Testing models' ability to understand human viewpoints

## Dataset Statistics

- **Total Tasks**: 25 perspective taking scenarios
- **Scene Complexity**: 1-5 objects per scene
- **Viewpoint Angles**: Random angles from 0-360 degrees
- **Object Types**: Various everyday items (hats, backpacks, etc.)
- **Image Resolution**: ~800x800 pixels per frame

## References

This task is inspired by research on:
- Theory of Mind in AI systems
- Spatial perspective taking in cognitive science
- 3D scene understanding and novel view synthesis
- Embodied cognition and spatial reasoning

## Future Extensions

Potential enhancements:
- **Dynamic scenes**: Objects moving during transformation
- **Partial occlusion**: More complex visibility reasoning
- **Multiple agents**: Reasoning about multiple viewpoints
- **Interactive scenarios**: Agent moving and changing viewpoint
- **Natural language**: Describing what the agent can see

## Contributing

To add more perspective taking tasks:
1. Create numbered folders in `data/perspective_taking/`
2. Include `first_frame.png`, `final_frame.png`, and `prompt.txt`
3. Optionally add `scene_metadata.json` with scene details
4. Run dataset creation to validate new tasks

---

*Last updated: November 2024*
*VMEvalKit Team*

