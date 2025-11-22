# VMEvalKit Task Creation Guide

This guide explains how to add reasoning tasks to VMEvalKit. Tasks can be **locally generated** or **downloaded from external datasets**.

## 🎯 Core Concept

Every task consists of three components:
1. **First Frame**: Initial state image (problem)
2. **Final Frame**: Target state image (solution)  
3. **Text Prompt**: Instructions for the model

## 📦 Task Types

### Locally Generated
Programmatically created tasks (e.g., Chess, Maze, Sudoku)
- Full control over generation
- Unlimited variations
- Requires `PROMPTS.py`

### Downloaded Tasks
External datasets from HuggingFace (e.g., VideoThinkBench, MME-CoF)
- Pre-curated benchmarks
- Located in `vmevalkit/tasks/external/`
- Prompts included in dataset

## 🏗️ Architecture

### Directory Structure

```
vmevalkit/tasks/{task_name}_task/
├── __init__.py              # Export create_dataset
├── {task}_reasoning.py      # Main implementation
├── PROMPTS.py              # (Locally generated only)
└── {TASK}.md               # Documentation

data/questions/{task_name}_task/{task_id}/
├── first_frame.png
├── final_frame.png
├── prompt.txt
└── question_metadata.json
```

### Registry (`vmevalkit/runner/TASK_CATALOG.py`)

```python
DOMAIN_REGISTRY = {
    'your_task': {
        'name': 'Your Task',
        'description': 'Brief description',
        'module': 'vmevalkit.tasks.your_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    }
}
```

## 📋 Requirements

### All Tasks
- Module folder: `vmevalkit/tasks/{task_name}_task/`
- `__init__.py` exporting `create_dataset`
- Main script with `create_dataset(num_samples)` function
- `{TASK_NAME}.md` documentation
- Entry in `DOMAIN_REGISTRY`
- PNG images: `first_frame.png` and `final_frame.png` (~400x400px)

### Locally Generated Only
- `PROMPTS.py` with prompt templates
- Image generation logic

### Downloaded Only
- Download logic (e.g., HuggingFace)
- Format conversion to VMEvalKit structure

### Data Structure
```python
{
    "id": "{task_name}_{id:04d}",
    "prompt": "Instructions",
    "first_image_path": "...",
    "final_image_path": "...",
    "domain": "{task_name}",
    "task_category": "Category",
    "difficulty": "easy|medium|hard",
    "{task_name}_data": {...},
    "created_at": "ISO timestamp"
}
```

---

## 📝 Path A: Locally Generated Task

### Implementation Steps

**1. Register** (`TASK_CATALOG.py`):
```python
'your_task': {
    'name': 'YourTask',
    'description': 'Brief description',
    'module': 'vmevalkit.tasks.your_task',
    'create_function': 'create_dataset',
    'process_dataset': lambda dataset, num_samples: dataset['pairs']
}
```

**2. Create Module** (`__init__.py`):
```python
from .your_reasoning import create_dataset
__all__ = ['create_dataset']
```

**3. Create Prompts** (`PROMPTS.py`):
```python
PROMPTS = [
    "Clear instruction for the model.",
    "Solve this {difficulty} puzzle."  # With parameters
]
```

**4. Implement Generation** (`your_reasoning.py`):

```python
import tempfile
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
from .PROMPTS import PROMPTS

def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    pairs = []
    for i in range(num_samples):
        task_data = generate_single_task(i)
        pair = create_task_pair(task_data, f"your_task_{i:04d}")
        pairs.append(pair)
    return {"name": "your_task_tasks", "pairs": pairs}

def create_task_pair(task_data: Dict, task_id: str) -> Dict:
    temp_dir = tempfile.mkdtemp()
    first_path = Path(temp_dir) / f"{task_id}_first.png"
    final_path = Path(temp_dir) / f"{task_id}_final.png"
    
    # Generate images (task-specific)
    create_image(task_data["problem"], first_path)
    create_image(task_data["solution"], final_path)
    
    return {
        "id": task_id,
        "prompt": PROMPTS[0],
        "first_image_path": str(first_path),
        "final_image_path": str(final_path),
        "domain": "your_task",
        "task_category": "YourTask",
        "difficulty": task_data["difficulty"],
        "your_task_data": task_data,
        "created_at": datetime.now().isoformat()
    }

def create_image(data: Any, output_path: Path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    # Your visualization logic here
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

**5. Run**: `python vmevalkit/runner/create_dataset.py --pairs-per-domain 50`

---

## 📥 Path B: Downloaded Task

**1. Register** (`TASK_CATALOG.py`):
```python
'your_hf_task': {
    'name': 'Your HF Task',
    'description': 'External dataset',
    'module': 'vmevalkit.tasks.external.your_hf_task',
    'create_function': 'create_dataset',
    'process_dataset': lambda dataset, num_samples: dataset['pairs']
}
```

**2. Create Module** (`__init__.py`):
```python
from .your_hf_download import create_dataset
__all__ = ['create_dataset']
```

**3. Implement Download** (`your_hf_download.py`):

```python
from typing import Dict, Any
from datasets import load_dataset

def create_dataset(num_samples: int = None) -> Dict[str, Any]:
    dataset = load_dataset('your-org/your-dataset', 'subset', split='test')
    
    pairs = []
    for idx, item in enumerate(dataset):
        if not item.get('prompt') or not item.get('image'):
            continue
        
        pairs.append({
            'id': f"your_hf_task_{idx:04d}",
            'domain': 'your_hf_task',
            'prompt': item['prompt'],
            'first_image': item['image'],
            'solution_image': item.get('solution_image')
        })
    
    return {
        'name': 'your_hf_task',
        'pairs': pairs,
        'source': 'huggingface',
        'hf_dataset': 'your-org/your-dataset'
    }
```

**4. Run**: `python vmevalkit/runner/create_dataset.py --pairs-per-domain all`

---

## 📚 Reference: Complete Examples

See existing implementations:
- **Locally Generated**: `vmevalkit/tasks/sudoku_task/`
- **Downloaded**: `vmevalkit/tasks/external/videothinkbench_arc_agi_task/`

## 🎨 Image Generation Best Practices

```python
import matplotlib.pyplot as plt

def create_visualization(data, output_path):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Your drawing code
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()  # Important: free memory
```

**Standards**: figsize=(6,6), dpi=150, PNG format, ~400x400px, high contrast

## 🧪 Testing

```bash
# Generate test dataset
python vmevalkit/runner/create_dataset.py --pairs-per-domain 5

# Verify structure
ls data/questions/your_task_task/your_task_0000/

# Check images exist
open data/questions/your_task_task/your_task_0000/first_frame.png
```

**Verify**: All required fields present, images are PNG, prompts are clear

## 🔧 Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError` | Add to `DOMAIN_REGISTRY` in `TASK_CATALOG.py` |
| Images not generated | `pip install matplotlib` |
| Wrong format | Always use PNG, not JPEG |
| Import errors | Check `__init__.py` exports `create_dataset` |
| Images too large | Use `figsize=(6,6), dpi=150` |
| Temp files accumulating | Use `tempfile.mkdtemp()` |

## 📚 Advanced Patterns

**Difficulty Levels**: Vary parameters (grid_size, obstacles, etc.) based on difficulty
**Validation**: Check solutions are valid before including
**Caching**: Use pickle to cache expensive generations
**External Libraries**: Handle optional dependencies gracefully

## 🎯 Task Design Guidelines

**Good Tasks Have**:
- Clear visual distinction between problem/solution
- Unambiguous goals
- Verifiable solutions
- Multiple difficulty levels
- Focus on specific reasoning capabilities

**Task Categories**: Spatial reasoning, logical deduction, pattern recognition, strategic planning, mathematical reasoning

## ✅ Checklist

**All Tasks**:
- [ ] Module folder and `__init__.py` created
- [ ] `create_dataset()` function implemented
- [ ] Entry in `DOMAIN_REGISTRY`
- [ ] PNG images: `first_frame.png`, `final_frame.png`
- [ ] All required JSON fields present
- [ ] Documentation (.md file)

**Locally Generated**: 
- [ ] `PROMPTS.py` with templates
- [ ] Image generation logic

**Downloaded**: 
- [ ] Download/conversion logic

**Quality**:
- [ ] Clear images, unambiguous prompts
- [ ] Multiple difficulty levels
- [ ] Dataset generation runs successfully

---

## 🎓 Summary

Two paths for adding tasks:

**Locally Generated**: 
1. Register → 2. Create module → 3. Add PROMPTS.py → 4. Generate images → 5. Test

**Downloaded**: 
1. Register → 2. Create module → 3. Implement download → 4. Convert format → 5. Test

Both use the same `create_dataset()` interface. Mix and match as needed! 🚀