# Tin's Tasks - Quick Reference Guide

## 📦 Four New Tasks

| Task | What It Tests | Difficulty | Files Needed |
|------|---------------|------------|--------------|
| 🔵 **Counting Circles** | Visual counting of shapes | Easy-Medium | 3 files |
| 🔲 **Counting Squares** | Recursive/nested counting | Medium-Hard | 3 files |
| 🔤 **Letter Counting** | Text recognition + counting | Easy-Medium | 3 files |
| 🚇 **Subway Pathfinding** | Spatial navigation | Hard | 3 files |

---

## 🏗️ File Structure (Each Task)

```
vmevalkit/tasks/{task_name}_task/
├── __init__.py                    # Export create_dataset
├── {task_name}_reasoning.py       # Main implementation
├── PROMPTS.py                     # Prompt templates
└── {TASK_NAME}.md                 # Documentation
```

---

## 🔧 Implementation Checklist (Per Task)

### Step 1: Create Module Structure
```bash
mkdir -p vmevalkit/tasks/{task_name}_task
cd vmevalkit/tasks/{task_name}_task
```

### Step 2: Create Files
- [ ] `__init__.py` - Export function
- [ ] `{task_name}_reasoning.py` - Core logic
- [ ] `PROMPTS.py` - Prompt templates
- [ ] `{TASK_NAME}.md` - Documentation

### Step 3: Implement Core Function
```python
def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    # Generate task pairs
    # Return {"name": "...", "pairs": [...]}
```

### Step 4: Register Task
Add to `TASK_CATALOG.py`:
```python
'{task_name}': {
    'name': 'Task Name',
    'description': '...',
    'module': 'vmevalkit.tasks.{task_name}_task',
    'create_function': 'create_dataset',
    'process_dataset': lambda dataset, num_samples: dataset['pairs']
}
```

### Step 5: Add Evaluation Guidance
Update `gpt4o_eval.py` and `internvl.py` TASK_GUIDANCE

### Step 6: Test
```bash
python vmevalkit/runner/create_dataset.py --domains {task_name} --pairs-per-domain 5
```

---

## 📊 Implementation Order (Recommended)

1. **Counting Circles** (Easiest, good starting point)
2. **Letter Counting** (Similar structure to circles)
3. **Counting Squares** (More complex logic)
4. **Subway Pathfinding** (Most complex)

---

## 🎯 Key Differences from Original Code

| Aspect | Tin's Code | VMEvalKit |
|--------|------------|-----------|
| **Data Structure** | Plain dicts | Pydantic models |
| **Prompts** | In main file | Separate PROMPTS.py |
| **Output** | test_samples.json | Task pairs with paths |
| **Images** | Specific directories | tempfile.mkdtemp() |
| **Error Handling** | Try-catch blocks | No try-catch (per user rules) |

---

## 🚀 Quick Commands

```bash
# Generate single task
python vmevalkit/runner/create_dataset.py --domains counting_circles --pairs-per-domain 10

# Generate all Tin's tasks
python vmevalkit/runner/create_dataset.py \
  --domains counting_circles,counting_squares,letter_counting,subway_pathfinding \
  --pairs-per-domain 50

# Test evaluation
python vmevalkit/runner/evaluate.py --model gpt4o --tasks counting_circles --num-samples 5
```

---

## 📝 Code Templates

### Template: __init__.py
```python
from .{task_name}_reasoning import create_dataset
__all__ = ['create_dataset']
```

### Template: PROMPTS.py
```python
PROMPTS = [
    "Clear instruction for the model.",
    "Alternative instruction format.",
]
DEFAULT_PROMPT_INDEX = 0
```

### Template: Pydantic Model
```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from datetime import datetime

class TaskData(BaseModel):
    """Task-specific data structure."""
    # Add fields here
    
class TaskPair(BaseModel):
    id: str
    prompt: str
    first_image_path: str
    final_image_path: str
    domain: str
    task_category: str
    difficulty: str
    ground_truth: Any
    metadata: Dict[str, Any]
    created_at: datetime = Field(default_factory=datetime.now)
```

---

## ⚡ Common Pitfalls to Avoid

1. ❌ Using try-catch blocks (user rules forbid)
2. ❌ Forgetting to close matplotlib figures (`plt.close()`)
3. ❌ Using plain dicts instead of Pydantic models
4. ❌ Hardcoding paths (use `tempfile.mkdtemp()`)
5. ❌ Missing `__init__.py` or wrong exports
6. ❌ Forgetting to register in TASK_CATALOG.py
7. ❌ Not updating evaluation guidance files

---

## 📏 Image Standards

- **Format**: PNG only
- **Size**: figsize=(6,6), dpi=150
- **Dimensions**: ~400x400px
- **Contrast**: High, clear visuals
- **Tool**: matplotlib

---

## 🎓 Success Criteria

✅ All 4 tasks implemented  
✅ Registered in TASK_CATALOG.py  
✅ Evaluation guidance added  
✅ Documentation complete  
✅ Tests pass  
✅ Dataset generates successfully  
✅ Images meet quality standards  

---

For detailed implementation plan, see: `TIN_TASKS_INTEGRATION_PLAN.md`

