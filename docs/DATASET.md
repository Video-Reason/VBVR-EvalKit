# Dataset Creation Framework for VMEvalKit

## Overview

This document provides a comprehensive guide for creating new reasoning datasets within the VMEvalKit framework. The framework is designed to evaluate video models' ability to demonstrate reasoning through visual problem-solving tasks that show transitions from initial problem states to final solution states.

## Core Design Philosophy

### Video Reasoning Evaluation
VMEvalKit evaluates video models on their ability to:
1. **Understand**: Parse initial problem state from visual input
2. **Reason**: Identify the solution or next steps
3. **Demonstrate**: Generate video showing the solution process
4. **Execute**: Accurately represent the transition from problem to solution

### Task Structure Pattern
All datasets follow the **First Frame → Final Frame** pattern:
- **First Frame**: Shows the initial problem state
- **Final Frame**: Shows the solved/target state  
- **Video Task**: Model must generate video showing the transition
- **Text Prompt**: Provides instructions for the reasoning task

---

## Fundamental Data Unit: Task Pair

**The basic unit of all VMEvalKit datasets is a "Task Pair"**, consisting of:

- **Initial state image**: Visual representation of the problem to solve
- **Final state image**: Visual representation of the solved/target state
- **Text prompt**: Natural language instructions for the video model
- **Metadata**: Task-specific information (difficulty, generation method, etc.)

This consistent structure enables unified evaluation across all reasoning domains.

---

## Dataset Framework Structure

### 1. Directory Organization

Every new dataset must follow this exact structure:

```
vmevalkit/tasks/{task_name}_task/
├── __init__.py                    # Module initialization
├── {task_name}_reasoning.py       # Main generation script
├── {TASK_NAME}.md                 # Documentation
└── [additional_modules.py]        # Optional helper modules

data/
├── questions/
│   ├── {task_name}_tasks/
│   │   └── {task_name}_tasks.json     # Dataset file
│   └── generated_{task_name}/
│       ├── {task_name}_0000_first.png # First frame images
│       ├── {task_name}_0000_final.png # Final frame images
│       └── ...                        # Additional image pairs
└── outputs/                           # Model-generated videos
```

### 2. Naming Conventions

- **Task folder**: `{task_name}_task` (e.g., `maze_task`, `chess_task`)
- **Main script**: `{task_name}_reasoning.py` 
- **Documentation**: `{TASK_NAME}.md` (uppercase)
- **Dataset file**: `{task_name}_tasks.json`
- **Images**: `{task_name}_{id:04d}_{first|final}.png`

---

## Required Dataset Format

### JSON Structure

Every dataset must follow this exact JSON structure:

```json
{
  "name": "{task_name}_tasks",
  "description": "Description of the reasoning tasks for video model evaluation (X pairs)",
  "pairs": [
    {
      "id": "{task_name}_0000",
      "prompt": "Task instruction for the video model",
      "first_image_path": "data/questions/generated_{task_name}/{task_name}_0000_first.png",
      "final_image_path": "data/questions/generated_{task_name}/{task_name}_0000_final.png", 
      "task_category": "TaskType",
      "domain": "{task_name}",
      "{task_name}_data": {
        "generation_method": "description_of_method",
        // Task-specific metadata
      },
      "difficulty": "easy|medium|hard",
      // Additional task-specific fields
      "created_at": "ISO timestamp"
    }
  ]
}
```

### Required Fields

1. **Top Level**:
   - `name`: Dataset identifier
   - `description`: Human-readable description with task count
   - `pairs`: Array of task pairs

2. **Task Pair Level**:
   - `id`: Unique identifier (`{task_name}_####`)
   - `prompt`: Instructions for video model
   - `first_image_path`: Path to initial state image
   - `final_image_path`: Path to solution state image
   - `task_category`: Category name for the task type
   - `domain`: One of `chess`, `maze`, `raven`, `rotation` (used for aggregation)
   - `{task_name}_data`: Task-specific metadata object
   - `difficulty`: Standardized difficulty level
   - `created_at`: ISO timestamp

### Image Requirements

- **Format**: PNG files (required for consistency)
- **Naming**: `{task_name}_{id:04d}_{first|final}.png`
- **Size**: Recommended 400x400 pixels (configurable)
- **Content**: Clear, unambiguous visual representation
- **Pairs**: Each task must have exactly one first and one final frame

---

## Implementation Template

### 1. Main Generation Script Template

Create `{task_name}_reasoning.py` following this structure:

```python
#!/usr/bin/env python3
"""
{Task Name} Reasoning Task for VMEvalKit

[Task description and purpose]

Author: VMEvalKit Team
"""

import os
import json
import random
from typing import List, Dict, Any
from datetime import datetime

class {TaskName}Generator:
    """Self-contained {task_name} task generator."""
    
    def __init__(self):
        self.generated_positions = []
        
    def generate_tasks(self, num_tasks: int = 50) -> List[Dict[str, Any]]:
        """Generate {task_name} tasks using built-in templates."""
        print(f"🎯 Generating {num_tasks} {task_name} tasks...")
        
        # Implementation: Generate your specific tasks here
        # Must return list of task dictionaries
        
        return self.generated_positions[:num_tasks]

def generate_task_images(task_data: Dict[str, Any], output_dir: str) -> tuple:
    """
    Generate first and final frame images for a task.
    
    Returns:
        (first_image_path, final_image_path)
    """
    # Implementation: Create PNG images showing initial and final states
    pass

def create_task_pair(task_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Create a task pair in VMEvalKit format."""
    
    # Generate images
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    first_image_path = f"data/questions/generated_{task_name}/{task_id}_first.png"
    final_image_path = f"data/questions/generated_{task_name}/{task_id}_final.png"
    
    # Create images
    os.makedirs(os.path.join(base_dir, f"data/questions/generated_{task_name}"), exist_ok=True)
    generate_task_images(task_data, base_dir)
    
    # Generate prompt
    prompt = generate_prompt(task_data)
    
    # Create task pair
    return {
        "id": task_id,
        "prompt": prompt,
        "first_image_path": first_image_path,
        "final_image_path": final_image_path,
        "task_category": "TaskType",
        f"{task_name}_data": {
            "generation_method": "method_description",
            # Add task-specific metadata
        },
        "difficulty": task_data.get("difficulty", "easy"),
        "created_at": datetime.now().isoformat()
    }

def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """Create complete dataset."""
    
    print(f"🎯 Creating {task_name} dataset with {num_samples} samples...")
    
    # Generate tasks
    generator = {TaskName}Generator()
    tasks = generator.generate_tasks(num_samples)
    
    # Create task pairs
    pairs = []
    for i, task_data in enumerate(tasks):
        task_id = f"{task_name}_{i:04d}"
        pair = create_task_pair(task_data, task_id)
        pairs.append(pair)
        print(f"✅ Created task {task_id}")
    
    # Create dataset
    dataset = {
        "name": f"{task_name}_tasks",
        "description": f"{Task_name} reasoning tasks for video model evaluation ({len(pairs)} pairs)",
        "pairs": pairs
    }
    
    # Save dataset
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    dataset_dir = os.path.join(base_dir, f"data/questions/{task_name}_tasks")
    os.makedirs(dataset_dir, exist_ok=True)
    output_path = os.path.join(dataset_dir, f"{task_name}_tasks.json")
    
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"✅ Saved dataset: {output_path}")
    return dataset

def main():
    """Generate dataset."""
    dataset = create_dataset(num_samples=50)
    print(f"🚀 {task_name} reasoning dataset ready!")

if __name__ == "__main__":
    main()
```

### 2. Module Initialization Template

Create `__init__.py`:

```python
"""
{Task Name} Task Module for VMEvalKit

[Brief description of task capabilities]
"""

from .{task_name}_reasoning import (
    {TaskName}Generator,
    create_dataset,
    # Add other exports
)

__all__ = [
    '{TaskName}Generator',
    'create_dataset',
    # List all exports
]

__version__ = "1.0.0"
```

### 3. Documentation Template

Create `{TASK_NAME}.md` following the structure of existing task documentation.

---

## Best Practices

### 1. Self-Contained Generation
- **No External Dependencies**: Tasks should generate without requiring external files
- **Reproducible**: Same parameters should produce consistent results
- **Configurable**: Allow customization of dataset size and parameters
- **Verified**: All generated tasks should be validated for correctness

### 2. Image Generation
- **PNG Format**: Required for consistency across all datasets
- **Clear Visuals**: Images must clearly show the problem and solution states
- **Consistent Size**: Use standard dimensions (recommend 400x400px)
- **Unambiguous**: No room for interpretation about what the task requires

### 3. Prompt Design
- **Clear Instructions**: Unambiguous task descriptions
- **Consistent Format**: Similar phrasing patterns across tasks
- **Appropriate Difficulty**: Match prompt complexity to task difficulty
- **Action-Oriented**: Focus on what the model should demonstrate

### 4. Metadata Richness
- **Generation Method**: Document how tasks were created
- **Difficulty Classification**: Consistent difficulty levels
- **Pattern Tags**: Classify task types for analysis
- **Solution Information**: Include correct answers for validation

### 5. Validation
- **Correctness**: Verify all tasks have valid solutions
- **Uniqueness**: Avoid duplicate tasks
- **Completeness**: Ensure all required fields are present
- **Format Compliance**: Match the exact JSON structure

---

## Integration with VMEvalKit

### 1. Runner Integration

To integrate with the main evaluation runner, tasks should be:
- **Discoverable**: Follow naming conventions
- **Loadable**: Provide standard loading interface
- **Evaluable**: Include validation methods

### 2. Evaluation Metrics

Define task-specific evaluation criteria:
- **Correctness**: Did the model find the right solution?
- **Completeness**: Was the full solution demonstrated?
- **Clarity**: Is the video demonstration clear?
- **Efficiency**: How quickly was the solution found?

### 3. Difficulty Scaling

Implement consistent difficulty levels:
- **Easy**: Basic pattern recognition, simple solutions
- **Medium**: Multi-step reasoning, moderate complexity
- **Hard**: Complex patterns, advanced reasoning required

---

## Examples and References

### Existing Implementations

Study these reference implementations:

1. **Maze Tasks** (`vmevalkit/tasks/maze_task/`):
   - Spatial reasoning and navigation
   - Path finding from start to goal
   - Multiple maze generation algorithms

2. **Chess Tasks** (`vmevalkit/tasks/chess_task/`):
   - Strategic reasoning and pattern recognition  
   - Mate-in-1 tactical problems
   - Template-based position generation

### Task Types Suited for This Framework

The framework works best for tasks that have:
- **Clear Problem State**: Visually representable initial conditions
- **Definite Solution**: Unambiguous correct final states
- **Demonstrable Process**: Reasonable to show via video
- **Reasoning Component**: Requires more than memorization

Good task types:
- **Puzzle Solving**: Sliding puzzles, logic puzzles, pattern completion
- **Game Reasoning**: Chess tactics, checkers, tic-tac-toe
- **Mathematical Visualization**: Geometric proofs, algebra steps
- **Spatial Reasoning**: 3D rotations, path planning, layout optimization
- **Sequential Logic**: Step-by-step processes, algorithmic thinking

---

## Quality Assurance Checklist

Before submitting a new dataset, verify:

### ✅ Structure Compliance
- [ ] Follows exact directory structure
- [ ] Uses correct naming conventions
- [ ] Includes all required files

### ✅ Format Compliance  
- [ ] JSON structure matches specification exactly
- [ ] All required fields present
- [ ] PNG image format used
- [ ] Consistent file paths

### ✅ Content Quality
- [ ] All tasks have valid solutions
- [ ] Images are clear and unambiguous
- [ ] Prompts are well-written and clear
- [ ] Appropriate difficulty distribution

### ✅ Technical Quality
- [ ] Self-contained generation works
- [ ] No external file dependencies
- [ ] Reproducible results
- [ ] Proper error handling

### ✅ Documentation
- [ ] Complete task documentation
- [ ] Usage examples provided
- [ ] Clear explanation of reasoning type
- [ ] Integration instructions

---

## Advanced Features

### 1. Dynamic Difficulty
Implement adaptive difficulty based on:
- Task complexity metrics
- Solution path length
- Required reasoning depth
- Pattern recognition difficulty

### 2. Multi-Modal Tasks
Extend to tasks involving:
- Text + Image → Video reasoning
- Audio + Visual → Video responses  
- Multi-step reasoning chains
- Interactive problem solving

### 3. Evaluation Extensions
Add sophisticated evaluation:
- Partial credit for incomplete solutions
- Style and efficiency scoring
- Creativity in solution approach
- Robustness across variations

---

## Conclusion

The VMEvalKit dataset framework provides a robust, extensible foundation for creating video reasoning evaluation tasks. By following these guidelines, you ensure:

1. **Consistency**: Your dataset integrates seamlessly with existing infrastructure
2. **Quality**: Tasks meet evaluation standards and provide meaningful assessment
3. **Maintainability**: Code follows established patterns and is easy to extend
4. **Reproducibility**: Others can understand, use, and build upon your work

The framework scales from simple pattern recognition to complex multi-step reasoning while maintaining consistent interfaces and evaluation standards. Focus on creating tasks that genuinely test the reasoning capabilities you want to evaluate, and the framework will handle the infrastructure details.

**Happy dataset creation!** 🎯
