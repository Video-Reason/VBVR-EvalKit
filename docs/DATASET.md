# VMEvalKit Dataset Creation Guide

## Core Concept: Task Pairs

All VMEvalKit datasets follow the **First Frame → Final Frame** pattern:
- **Task Pair**: Initial state image + Final state image + Text prompt + Metadata
- **Video Task**: Models generate video showing the reasoning transition
- **Consistent Structure**: Enables unified evaluation across all domains

## Quick Start: Adding a New Task

**The simplest way to add a new task is to register it in `vmevalkit/runner/create_dataset.py`:**

```python
DOMAIN_REGISTRY = {
    'your_task': {
        'emoji': '🎯',
        'name': 'YourTask',
        'description': 'Brief description of reasoning capability',
        'module': 'vmevalkit.tasks.your_task',
        'create_function': 'create_dataset',  # Standard function name
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    # ... existing tasks ...
}
```

That's it! The registry automatically handles dataset generation and integration.

## Directory Structure

```
vmevalkit/tasks/{task_name}_task/
├── __init__.py                       # Module exports
├── {task_name}_reasoning.py          # Main generation script
├── PROMPTS.py                        # Centralized prompts for prompt engineering experiments
├── {TASK_NAME}.md                    # Documentation
└── [helpers.py]                      # Optional modules

data/questions/
├── vmeval_dataset.json               # Master dataset (all domains)
├── {task_name}_task/                 # Per-domain folders
│   └── {task_name}_{id:04d}/        # Per-question folders
│       ├── first_frame.png          # Initial state
│       ├── final_frame.png          # Solution state
│       ├── prompt.txt                # Text instruction
│       └── question_metadata.json   # Task metadata
```

## JSON Format

### Master Dataset (`vmeval_dataset.json`)
```json
{
  "name": "vmeval_dataset",
  "description": "VMEvalKit video reasoning evaluation dataset (X task pairs)",
  "created_at": "ISO timestamp",
  "total_pairs": X,
  "generation_info": {
    "domains": {
      "chess": { "count": N, "description": "..." }
    }
  },
  "pairs": [/* Array of task pairs */]
}
```

### Task Pair Format
```json
{
  "id": "{task_name}_{id:04d}",
  "prompt": "Instruction for video model",
  "first_image_path": "{task_name}_task/{id}/first_frame.png",
  "final_image_path": "{task_name}_task/{id}/final_frame.png",
  "domain": "{task_name}",
  "task_category": "CategoryName",
  "difficulty": "easy|medium|hard",
  "{task_name}_data": { /* Task-specific metadata */ },
  "created_at": "ISO timestamp"
}
```

## Implementation Requirements

### 1. Required Function: `create_dataset()`

Every task module must implement this function:

```python
def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """Standard interface for dataset generation."""
    
    # 1. Generate task data
    generator = YourTaskGenerator()
    tasks = generator.generate_tasks(num_samples)
    
    # 2. Create task pairs with images
    pairs = []
    for i, task_data in enumerate(tasks):
        task_id = f"{task_name}_{i:04d}"
        
        # Create per-question folder
        question_dir = f"data/questions/{task_name}_task/{task_id}"
        os.makedirs(question_dir, exist_ok=True)
        
        # Generate and save images as PNG
        generate_images(task_data, question_dir)  # Creates first_frame.png, final_frame.png
        
        # Create metadata
        pair = {
            "id": task_id,
            "prompt": generate_prompt(task_data),
            "first_image_path": f"{task_name}_task/{task_id}/first_frame.png",
            "final_image_path": f"{task_name}_task/{task_id}/final_frame.png",
            "domain": task_name,
            "difficulty": task_data.get("difficulty", "medium"),
            # ... other fields ...
        }
        
        # Save metadata
        with open(f"{question_dir}/question_metadata.json", 'w') as f:
            json.dump(pair, f, indent=2)
            
        pairs.append(pair)
    
    # 3. Return dataset dictionary
    return {
        "name": f"{task_name}_tasks",
        "description": f"Description ({len(pairs)} pairs)",
        "pairs": pairs
    }
```

### 2. Image Requirements
- Format: PNG (required)
- Names: `first_frame.png`, `final_frame.png`
- Size: ~400x400 pixels recommended
- Content: Clear visual representation of problem/solution

### 3. Module Structure (`__init__.py`)
```python
from .{task_name}_reasoning import create_dataset, YourTaskGenerator

__all__ = ['create_dataset', 'YourTaskGenerator']
```

## Complete Example: Adding a Sudoku Task

1. **Register in `vmevalkit/runner/create_dataset.py`:**
```python
DOMAIN_REGISTRY = {
    'sudoku': {
        'emoji': '🔢',
        'name': 'Sudoku',
        'description': 'Logical deduction and constraint satisfaction',
        'module': 'vmevalkit.tasks.sudoku_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    }
}
```

2. **Create `vmevalkit/tasks/sudoku_task/sudoku_reasoning.py`:**
```python
def create_dataset(num_samples: int = 50):
    # Implementation following the template above
    pass
```

3. **Run generation:**
```bash
python vmevalkit/runner/create_dataset.py --pairs-per-domain 50
```

That's it! Your task is now integrated into VMEvalKit.

## Testing Your Task

```bash
# Generate dataset with your new task
python vmevalkit/runner/create_dataset.py --pairs-per-domain 10

# Verify generated files
ls data/questions/{your_task}_task/

# Run inference
python vmevalkit/runner/inference.py --model [model_name] --domain {your_task}
```

## Key Points

✅ **DO**:
- Follow the exact directory structure
- Implement `create_dataset(num_samples)` function
- Save images as PNG in per-question folders
- Include all required JSON fields
- Register in DOMAIN_REGISTRY for automatic integration

❌ **DON'T**:
- Use formats other than PNG for images
- Skip required metadata fields
- Create files outside the standard structure
- Forget to register in DOMAIN_REGISTRY

## Need Help?

Check existing implementations:
- `vmevalkit/tasks/chess_task/` - Chess reasoning
- `vmevalkit/tasks/maze_task/` - Spatial navigation  
- `vmevalkit/tasks/raven_task/` - Abstract patterns
- `vmevalkit/tasks/rotation_task/` - 3D mental rotation