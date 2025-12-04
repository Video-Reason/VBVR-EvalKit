# Control Panel Animation Task

## Overview

The Control Panel Animation Task tests video generation models' ability to observe, reason, and generate animations for a control panel system with indicator lights and sliders. The key challenge is that models must **infer the mapping between slider positions and light colors** from the initial image, rather than being told the specific rules.

## Design Philosophy

This task is designed as an **observation and reasoning task**, not a simple instruction-following task:

- ❌ **Does NOT tell** the model the specific mapping (e.g., left=red, middle=green, right=blue)
- ✅ **Only provides** general rules (each position corresponds to a fixed color)
- ✅ **Requires** the model to observe the initial image and infer the mapping
- ✅ **Tests** visual understanding, logical reasoning, and animation generation

## Task Structure

### Scene Components

- **Indicator Lights**: Each light can display three discrete colors: red, green, or blue
- **Control Panels**: Each light has a unique control panel directly below it
- **Sliders**: Each control panel has a horizontal slider that can be positioned at three discrete positions: left, middle, or right
- **Mapping Rule**: All control panels share the same fixed mapping (left, middle, right → three colors), but the specific correspondence must be inferred

### Task Objective

Given an initial image showing lights in various colors with sliders at different positions, the model must:

1. **Observe** the initial state (light colors and slider positions)
2. **Infer** the mapping between slider positions and light colors
3. **Generate** an animation showing sliders moving horizontally to positions that make all lights display the target color

## Difficulty Levels

The task supports multiple difficulty levels based on the number of lights:

- **2_lights**: Two indicator lights (simpler, may need to infer the third position)
- **3_lights**: Three indicator lights (can fully infer the mapping)
- **4_lights**: Four indicator lights (may have duplicate colors, tests rule validation)
- **6_lights**: Six indicator lights (more complex state, requires stronger reasoning)

### Task Generation Strategy

For each scene configuration, **3 independent tasks** are generated, one for each target color:
- Task 1: Make all lights red
- Task 2: Make all lights green
- Task 3: Make all lights blue

Each task uses the same `first_frame` but has different `prompt` and `final_frame`.

## Usage

### Basic Usage

```python
from vmevalkit.tasks.control_panel_task import ControlPanelTaskGenerator

# Initialize generator
generator = ControlPanelTaskGenerator(
    canvas_size=(256, 256),
    temp_dir='./output'
)

# Generate a single task
task = generator.generate_single_task(
    task_id='control_panel_3lights_0001',
    difficulty='3_lights',
    seed=42,
    target_color='red'  # 'red', 'green', or 'blue'
)

# Access task components
print(task.prompt)              # The prompt for the video model
print(task.first_image_path)    # Path to initial state image
print(task.final_image_path)    # Path to target state image
print(task.num_lights)          # Number of lights
print(task.difficulty)          # Difficulty level
```

### Generate Dataset

```python
from vmevalkit.tasks.control_panel_task import create_dataset

# Generate dataset with custom distribution
dataset = create_dataset(
    num_samples=50,
    difficulty_distribution={
        '2_lights': 0.25,
        '3_lights': 0.25,
        '4_lights': 0.25,
        '6_lights': 0.25
    }
)

# Each scene generates 3 tasks (one per target color)
# So 50 samples = 150 tasks total
```

### Custom Difficulty Distribution

```python
# Generate more tasks for specific difficulty levels
dataset = create_dataset(
    num_samples=100,
    difficulty_distribution={
        '2_lights': 0.1,   # 10 scenes = 30 tasks
        '3_lights': 0.3,   # 30 scenes = 90 tasks
        '4_lights': 0.3,   # 30 scenes = 90 tasks
        '6_lights': 0.3    # 30 scenes = 90 tasks
    }
)
```

## Data Structure

### ControlPanelTaskPair

```python
@dataclass
class ControlPanelTaskPair:
    id: str                      # Unique task identifier
    prompt: str                  # Instructions for the video model
    first_image_path: str        # Path to initial state image
    final_image_path: str        # Path to target state image
    task_category: str           # "ControlPanel"
    control_panel_data: Dict     # Metadata (panel config, canvas size, target color)
    difficulty: str              # "2_lights", "3_lights", "4_lights", "6_lights"
    num_lights: int              # Number of lights
    created_at: str              # ISO timestamp
```

## Prompt Structure

The prompt follows a structured format:

1. **Scene Structure**: Description of the control panel system
2. **System Rules**: General rules (discrete colors, discrete positions, independence)
3. **Mapping Rules**: Emphasizes that the mapping must be inferred (does not reveal specific correspondence)
4. **Task Objective**: Make all lights display the target color
5. **Operation Steps**: Step-by-step instructions for observation, inference, and action
6. **Video Generation Requirements**: Detailed animation requirements and constraints

### Key Prompt Features

- **No explicit mapping**: Does not tell which position corresponds to which color
- **Observation requirement**: "must be inferred from the initial image"
- **Horizontal movement**: Emphasizes that sliders move horizontally
- **Discrete positions**: No intermediate positions allowed
- **No UI elements**: Video must not contain text, arrows, highlights, etc.
- **Fixed camera**: Camera view must remain completely fixed

## Visual Layout

### Layout Configurations

- **2 lights**: Single row (1×2)
- **3 lights**: Single row (1×3) with increased spacing
- **4 lights**: 2 rows × 2 columns (2×2 grid)
- **6 lights**: 2 rows × 3 columns (2×3 grid)

### Visual Elements

- **Lights**: Colored circles (red, green, blue) with rings
- **Control Slots**: Black rectangular slots below each light
- **Sliders**: Gray rectangular blocks that move horizontally
- **Position Markers**: Small white dots at empty positions (where slider is not)

## Testing Capabilities

This task tests:

1. **Visual Observation**: Can the model correctly identify light colors and slider positions?
2. **Logical Reasoning**: Can the model infer the mapping from observations?
3. **Rule Understanding**: Can the model understand that all panels share the same mapping?
4. **Strategy Planning**: Can the model determine which sliders to move and where?
5. **Animation Generation**: Can the model generate smooth, correct animations?

## Example Scenarios

### Scenario 1: 3 Lights (Full Inference)

- Initial: Light 1=red (left), Light 2=green (middle), Light 3=blue (right)
- Model can observe all three mappings and fully infer the rule
- Target: Make all lights red → Move all sliders to left

### Scenario 2: 2 Lights (Partial Inference)

- Initial: Light 1=red (left), Light 2=green (middle)
- Model observes 2 positions, must infer the third
- Target: Make all lights blue → Move all sliders to right (inferred position)

### Scenario 3: 4 Lights (Rule Validation)

- Initial: Light 1=red (left), Light 2=red (left), Light 3=green (middle), Light 4=blue (right)
- Model can validate the rule: same colors have same positions
- Target: Make all lights green → Move sliders 1, 2, 4 to middle

## File Structure

```
vmevalkit/tasks/control_panel_task/
├── __init__.py              # Module exports
├── PROMPTS.py               # Prompt templates
├── control_panel_reasoning.py  # Core logic (generator, renderer, task creation)
└── README.md                # This file
```

## Dependencies

- `matplotlib`: For rendering control panel scenes
- `numpy`: For numerical operations
- `pathlib`: For file path handling
- Standard library: `random`, `tempfile`, `dataclasses`, `datetime`

## Notes

- All control panels share the same mapping rule (left=red, middle=green, right=blue internally, but not revealed to the model)
- Each scene generates 3 independent tasks (one per target color)
- Tasks are designed to test reasoning ability, not just instruction following
- The prompt emphasizes observation and inference over explicit instructions

