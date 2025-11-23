# Clock Time Reasoning Task for VMEvalKit

## 📊 Overview

The Clock task evaluates video generation models' capacity for **temporal reasoning** and **time calculation**. This task tests whether models can:

1. **Understand clock representation** - Parse clock face with hour hand only
2. **Calculate time progression** - Add k hours (1-24) to initial time
3. **Generate time transitions** - Show clock hand movement through video
4. **Demonstrate temporal reasoning** - Handle 12-hour cycle correctly




## 🚀 Usage

### Generate Clock Tasks

Use the `create_questions.py` script to generate clock reasoning tasks:

```bash
# Generate 50 clock tasks (default)
python examples/create_questions.py --task clock

python examples/generate_videos.py --model svd --task clock
```


## 🎯 Task Description

### Input Components
- **First Frame**: A clock showing a random time with only the hour hand visible
- **Prompt**: Text instruction asking to show the clock after k hours (k from 1 to 24)
- **Format**: 600×600px PNG image at 150 DPI with clear clock face

### Expected Output
- **Video Sequence**: Animation showing the hour hand moving forward
- **Final Frame**: Clock showing the correct time after k hours
- **Time Calculation**: Proper handling of 12-hour cycle (e.g., 11:00 + 2 hours = 1:00)

### Core Features
- **Hour hand only**: No minute hand for simplicity
- **Random initial time**: Hour positions 0-11 (representing 12:00 to 11:00)
- **Variable hours to add**: k ranges from 1 to 24 hours
- **12-hour cycle**: Correctly wraps around (e.g., 10 + 5 = 3)

### Output Structure

Generated tasks are saved in:
```
data/questions/clock_task/
├── clock_0000/
│   ├── first_frame.png    # Initial clock time
│   ├── final_frame.png    # Clock after k hours
│   ├── prompt.txt         # Task instruction
│   └── question_metadata.json
├── clock_0001/
│   └── ...
└── ...
```

### Programmatic Usage

```python
from vmevalkit.tasks.clock_task import create_dataset, ClockTaskGenerator

# Generate dataset
dataset = create_dataset(num_samples=50)

# Access individual tasks
for task in dataset['pairs']:
    print(f"Task {task['id']}: {task['initial_hour']}:00 + {task['hours_to_add']}h = {task['final_hour']}:00")
    print(f"Prompt: {task['prompt']}")

# Generate single task
generator = ClockTaskGenerator()
task = generator.generate_single_task(task_id="custom_001")
```

## 📝 Task Format

Each clock task contains:
- **initial_hour**: Starting hour (0-11, where 0 = 12:00)
- **hours_to_add**: Number of hours to advance (1-24)
- **final_hour**: Resulting hour after addition (0-11)
- **prompt**: Instruction text with k value filled in

Example:
- Initial: 3:00 (hour = 3)
- Add: 5 hours
- Final: 8:00 (hour = 8)

Or with wrap-around:
- Initial: 10:00 (hour = 10)
- Add: 5 hours
- Final: 3:00 (hour = 3, since 10 + 5 = 15, 15 % 12 = 3)

## 🎨 Visual Design

The clock image features:
- **Circular clock face** with 12 hour markers
- **Hour numbers** (1-12) positioned around the circle
- **Hour hand only** (no minute hand) pointing to current hour
- **Center dot** for clock center
- **Clean white background** for maximum contrast

## 🔗 Related Resources

- [VMEvalKit Documentation](../../../README.md)
- [Adding Tasks Guide](../../../docs/ADDING_TASKS.md)
- Other reasoning tasks: Sudoku, Maze, Chess, Raven, Rotation
