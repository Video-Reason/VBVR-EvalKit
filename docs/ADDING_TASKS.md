# VMEvalKit Task Creation Guide

This comprehensive guide explains how to add new reasoning tasks to VMEvalKit. All tasks follow a standardized **First Frame → Final Frame** pattern with text prompts, enabling unified evaluation across diverse reasoning domains.

## 🎯 Core Concept: Task Pairs

Every VMEvalKit task consists of a **task pair** with three essential components:

1. **First Frame**: Initial state image (the problem/puzzle)
2. **Final Frame**: Target state image (the solution)  
3. **Text Prompt**: Instructions telling the model what to do

This structure enables video models to demonstrate reasoning by generating transitions from problem to solution.

## 🏗️ Architecture Overview

### Task System Structure

```
vmevalkit/
├── runner/
│   └── create_dataset.py       # Dataset generation orchestrator with registry
├── tasks/
│   ├── chess_task/
│   │   ├── __init__.py        # Module exports
│   │   ├── chess_reasoning.py # Main task generation logic
│   │   ├── PROMPTS.py         # Centralized prompt templates
│   │   └── CHESS.md           # Task documentation
│   ├── maze_task/
│   │   ├── __init__.py
│   │   ├── maze_reasoning.py
│   │   ├── PROMPTS.py
│   │   └── MAZE.md
│   └── [your_task]/           # Your new task

data/questions/
├── vmeval_dataset.json         # Master dataset (all domains combined)
└── {task_name}_task/          # Per-task directories
    └── {task_name}_{id:04d}/  # Per-question folders
        ├── first_frame.png     # Initial state (required)
        ├── final_frame.png     # Solution state (required)
        ├── prompt.txt          # Text instruction (auto-generated)
        └── question_metadata.json # Task metadata (auto-generated)
```

### Domain Registry System

Tasks are registered in `vmevalkit/runner/create_dataset.py`:

```python
DOMAIN_REGISTRY = {
    'chess': {
        'emoji': '♟️',
        'name': 'Chess',
        'description': 'Strategic thinking and tactical pattern recognition',
        'module': 'vmevalkit.tasks.chess_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    # Your task gets added here!
}
```

## 📋 Task Requirements

### Required Components

✅ **Module Structure**: Create folder `vmevalkit/tasks/{task_name}_task/`  
✅ **Main Script**: `{task_name}_reasoning.py` with `create_dataset()` function  
✅ **Prompt Templates**: `PROMPTS.py` with standardized prompts  
✅ **Module Init**: `__init__.py` exporting required functions  
✅ **Documentation**: `{TASK_NAME}.md` describing the task  

### Image Requirements

✅ **Format**: PNG (required - no other formats)  
✅ **Names**: `first_frame.png` and `final_frame.png` (exact names)  
✅ **Size**: ~400x400 pixels recommended (consistent across frames)  
✅ **Quality**: High contrast, clear visual elements  
✅ **Content**: Unambiguous problem and solution states

### Data Structure Requirements

Each task pair must follow this exact format:

```python
{
    "id": "{task_name}_{id:04d}",           # e.g., "sudoku_0042"
    "prompt": "Instructions for the model",  # From PROMPTS.py
    "first_image_path": "{task_name}_task/{id}/first_frame.png",
    "final_image_path": "{task_name}_task/{id}/final_frame.png",
    "domain": "{task_name}",                # Task domain name
    "task_category": "CategoryName",        # Display category
    "difficulty": "easy|medium|hard",       # Difficulty level
    "{task_name}_data": {...},             # Task-specific metadata
    "created_at": "ISO timestamp"           # Auto-generated
}
```

## 🚀 Quick Start: Minimal Implementation

The simplest way to add a new task:

### Step 1: Register Your Task

Add to `vmevalkit/runner/create_dataset.py`:

```python
DOMAIN_REGISTRY = {
    # ... existing tasks ...
    'your_task': {
        'emoji': '🎯',
        'name': 'YourTask',
        'description': 'Brief description of reasoning capability',
        'module': 'vmevalkit.tasks.your_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    }
}
```

### Step 2: Create Task Module

Create `vmevalkit/tasks/your_task/__init__.py`:

```python
from .your_reasoning import create_dataset, YourTaskGenerator

__all__ = ['create_dataset', 'YourTaskGenerator']
```

### Step 3: Create PROMPTS.py

Create `vmevalkit/tasks/your_task/PROMPTS.py`:

```python
"""Centralized prompts for your task."""

PROMPTS = [
    "Clear instruction telling the model what to do with the task.",
    # Add variations if needed
]

# Or with parameters
PROMPTS = [
    "Solve this {difficulty} puzzle by finding the pattern.",
]
```

### Step 4: Implement Task Generation

Create `vmevalkit/tasks/your_task/your_reasoning.py`:

```python
"""
Your Task Reasoning for VMEvalKit.

Brief description of what your task tests.
"""

import json
import random
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Import centralized prompts
from .PROMPTS import PROMPTS


def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """
    Standard interface for dataset generation.
    
    Args:
        num_samples: Number of task pairs to generate
        
    Returns:
        Dictionary with 'pairs' key containing task pairs
    """
    pairs = []
    
    for i in range(num_samples):
        # Generate task data
        task_data = generate_single_task(i)
        
        # Create task pair
        pair = create_task_pair(task_data, f"your_task_{i:04d}")
        pairs.append(pair)
    
    return {
        "name": "your_task_tasks",
        "description": f"Your task dataset ({len(pairs)} pairs)",
        "pairs": pairs
    }


def generate_single_task(index: int) -> Dict[str, Any]:
    """Generate data for a single task."""
    
    # Your task generation logic here
    # This is task-specific
    
    return {
        "problem": "...",     # Problem specification
        "solution": "...",    # Solution specification
        "difficulty": random.choice(["easy", "medium", "hard"]),
        # Add task-specific data
    }


def create_task_pair(task_data: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Create a task pair with images and metadata."""
    
    # Use temporary directory for images
    temp_dir = tempfile.mkdtemp()
    
    # Generate images
    first_image_path = Path(temp_dir) / f"{task_id}_first.png"
    final_image_path = Path(temp_dir) / f"{task_id}_final.png"
    
    # Create your images (task-specific visualization)
    create_problem_image(task_data["problem"], first_image_path)
    create_solution_image(task_data["solution"], final_image_path)
    
    # Generate prompt from template
    prompt = PROMPTS[0]  # Or select/format as needed
    
    # Create standardized task pair
    return {
        "id": task_id,
        "prompt": prompt,
        "first_image_path": str(first_image_path),
        "final_image_path": str(final_image_path),
        "domain": "your_task",
        "task_category": "YourTask",
        "difficulty": task_data["difficulty"],
        "your_task_data": {
            # Include all task-specific metadata
            "problem": task_data["problem"],
            "solution": task_data["solution"],
            # ... other fields
        },
        "created_at": datetime.now().isoformat()
    }


def create_problem_image(problem_data: Any, output_path: Path):
    """Create visualization of the problem."""
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Your visualization code here
    # Draw the problem state
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_solution_image(solution_data: Any, output_path: Path):
    """Create visualization of the solution."""
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Your visualization code here
    # Draw the solution state
    
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
```

### Step 5: Run Dataset Generation

```bash
python vmevalkit/runner/create_dataset.py --pairs-per-domain 50
```

That's it! Your task is now integrated into VMEvalKit.

## 📝 Complete Implementation Example: Sudoku Task

Here's a real, complete implementation from the codebase:

### File: `vmevalkit/tasks/sudoku_task/PROMPTS.py`

```python
"""Centralized prompts for Sudoku tasks."""

PROMPTS = [
    "Complete this 3x3 Sudoku puzzle. Fill in the empty cells so each row, column contains the numbers 1, 2, and 3 exactly once.",
    
    "Solve this {difficulty} 3x3 Sudoku puzzle by filling the empty squares.",
    
    "This is a 3x3 Sudoku puzzle. Each row and column must contain the digits 1-3 exactly once. Complete the puzzle.",
]

# Additional prompt templates for variety
SUDOKU_INSTRUCTIONS = {
    "concise": "Complete the 3x3 Sudoku.",
    
    "detailed": "Fill in the empty cells of this 3x3 Sudoku grid so that each row and column contains the numbers 1, 2, and 3 exactly once.",
    
    "step_by_step": "Analyze this 3x3 Sudoku puzzle. Identify the missing numbers and complete the grid following Sudoku rules."
}
```

### File: `vmevalkit/tasks/sudoku_task/sudoku_reasoning.py`

```python
"""
Sudoku Reasoning Task for VMEvalKit.

Simple 3x3 Sudoku puzzle generation for video model evaluation.
"""

import json
import random
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from .PROMPTS import PROMPTS


class Simple3x3SudokuGenerator:
    """Generator for 3x3 Sudoku puzzles."""
    
    def generate_solved_sudoku(self) -> List[int]:
        """Generate a complete valid 3x3 Sudoku solution."""
        
        # Pre-computed valid 3x3 Latin squares
        solutions = [
            [1, 2, 3, 2, 3, 1, 3, 1, 2],
            [1, 2, 3, 3, 1, 2, 2, 3, 1],
            [1, 3, 2, 2, 1, 3, 3, 2, 1],
            [2, 1, 3, 1, 3, 2, 3, 2, 1],
            [2, 3, 1, 1, 2, 3, 3, 1, 2],
            [3, 1, 2, 1, 2, 3, 2, 3, 1],
        ]
        
        return random.choice(solutions)
    
    def create_puzzle(
        self, 
        solution: List[int], 
        difficulty: int = 1
    ) -> List[Optional[int]]:
        """Remove numbers from solution to create puzzle."""
        
        # Difficulty determines how many numbers to remove
        cells_to_remove = {
            0: 2,  # Easy: remove 2 numbers
            1: 3,  # Medium: remove 3 numbers  
            2: 4,  # Hard: remove 4 numbers
        }.get(difficulty, 3)
        
        puzzle = solution.copy()
        indices = list(range(9))
        random.shuffle(indices)
        
        for i in indices[:cells_to_remove]:
            puzzle[i] = None
            
        return puzzle
    
    def create_board_image(self, sudoku_array: List, filepath: Path):
        """Create a visual representation of the Sudoku board."""
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Draw grid
        for i in range(4):
            lw = 2 if i % 3 == 0 else 1
            ax.axhline(i, color='black', linewidth=lw)
            ax.axvline(i, color='black', linewidth=lw)
        
        # Add numbers
        for i in range(3):
            for j in range(3):
                value = sudoku_array[i * 3 + j]
                if value is not None:
                    ax.text(j + 0.5, 2.5 - i, str(value),
                           ha='center', va='center',
                           fontsize=24, fontweight='bold')
        
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_aspect('equal')
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()


def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """
    Create Sudoku dataset with specified number of samples.
    
    Args:
        num_samples: Number of puzzles to generate
    
    Returns:
        Dataset dictionary with 'pairs' key
    """
    generator = Simple3x3SudokuGenerator()
    pairs = []
    
    # Generate puzzles with mixed difficulty
    difficulties = [0, 1, 2]  # easy, medium, hard
    difficulty_names = ["easy", "medium", "hard"]
    
    for i in range(num_samples):
        # Use temporary directory for images
        temp_dir = tempfile.mkdtemp()
        task_id = f"sudoku_{i:04d}"
        
        # Generate puzzle
        solution = generator.generate_solved_sudoku()
        difficulty = random.choice(difficulties)
        puzzle = generator.create_puzzle(solution, difficulty)
        
        # Create images
        puzzle_path = Path(temp_dir) / f"{task_id}_first.png"
        solution_path = Path(temp_dir) / f"{task_id}_final.png"
        
        generator.create_board_image(puzzle, puzzle_path)
        generator.create_board_image(solution, solution_path)
        
        # Select and format prompt
        prompt = PROMPTS[0]  # Use main standardized prompt
        
        # Create task pair
        pair = {
            "id": task_id,
            "prompt": prompt,
            "first_image_path": str(puzzle_path),
            "final_image_path": str(solution_path),
            "domain": "sudoku",
            "task_category": "Sudoku",
            "difficulty": difficulty_names[difficulty],
            "sudoku_data": {
                "puzzle": puzzle,
                "solution": solution,
                "num_given": sum(1 for x in puzzle if x is not None),
                "difficulty_level": difficulty
            },
            "created_at": datetime.now().isoformat()
        }
            
        pairs.append(pair)
    
    return {
        "name": "sudoku_tasks",
        "description": f"3x3 Sudoku reasoning tasks ({len(pairs)} pairs)",
        "pairs": pairs,
        "metadata": {
            "generator": "Simple3x3SudokuGenerator",
            "version": "1.0"
        }
    }
```

## 🎨 Image Generation Best Practices

### Using Matplotlib for Visualizations

Most tasks use matplotlib for consistent, high-quality visualizations:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch

def create_task_visualization(data, output_path):
    """Template for creating task visualizations."""
    
    # Standard figure setup
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    
    # Set coordinate system
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # Draw task elements
    # Example: Draw a grid
    for i in range(11):
        ax.axhline(i, color='gray', linewidth=0.5, alpha=0.5)
        ax.axvline(i, color='gray', linewidth=0.5, alpha=0.5)
    
    # Add shapes
    circle = Circle((5, 5), 1, color='blue', alpha=0.7)
    ax.add_patch(circle)
    
    rect = Rectangle((2, 2), 3, 2, color='red', alpha=0.5)
    ax.add_patch(rect)
    
    # Add text
    ax.text(5, 8, "Task Title", fontsize=16, fontweight='bold',
            ha='center', va='center')
    
    # Style settings
    ax.set_facecolor('#f0f0f0')  # Light gray background
    ax.grid(True, alpha=0.3)
    ax.axis('off')  # Hide axes for cleaner look
    
    # Save with consistent settings
    plt.tight_layout()
    plt.savefig(output_path, 
                dpi=150,  # High quality
                bbox_inches='tight',  # Crop whitespace
                facecolor='white',  # White background
                edgecolor='none')
    plt.close()  # Important: close figure to free memory
```

### Visual Consistency Guidelines

```python
# Color scheme for task elements
COLORS = {
    'player': '#2ECC40',      # Green for current position
    'goal': '#FF4136',        # Red for targets
    'obstacle': '#111111',    # Black for walls/blocks
    'path': '#FFFFFF',        # White for valid moves
    'highlight': '#FFDC00',   # Yellow for emphasis
    'grid': '#DDDDDD',        # Light gray for grid lines
}

# Standard sizes
FIGURE_SIZE = (6, 6)         # Consistent figure size
DPI = 150                     # High quality rendering
GRID_SIZE = 10               # Standard coordinate system
FONT_SIZE = 14               # Readable text size
LINE_WIDTH = 2               # Visible lines

# Marker styles
MARKERS = {
    'circle': Circle,
    'square': Rectangle,
    'arrow': patches.FancyArrow,
    'star': patches.RegularPolygon,  # With appropriate parameters
}
```

## 🧪 Testing Your Task

### Unit Test Template

```python
# test_your_task.py
import pytest
from pathlib import Path
from vmevalkit.tasks.your_task import create_dataset


def test_dataset_creation():
    """Test basic dataset creation."""
    
    dataset = create_dataset(num_samples=5)
    
    # Check structure
    assert "pairs" in dataset
    assert len(dataset["pairs"]) == 5
    
    # Check each pair
    for pair in dataset["pairs"]:
        # Verify required fields
        assert "id" in pair
        assert "prompt" in pair
        assert "first_image_path" in pair
        assert "final_image_path" in pair
        assert "domain" in pair
        assert "difficulty" in pair
        
        # Check images exist
        assert Path(pair["first_image_path"]).exists()
        assert Path(pair["final_image_path"]).exists()
        
        # Check image format
        assert pair["first_image_path"].endswith(".png")
        assert pair["final_image_path"].endswith(".png")


def test_difficulty_distribution():
    """Test that all difficulties are generated."""
    
    dataset = create_dataset(num_samples=30)
    difficulties = [p["difficulty"] for p in dataset["pairs"]]
    
    assert "easy" in difficulties
    assert "medium" in difficulties
    assert "hard" in difficulties


def test_prompt_generation():
    """Test prompt templates."""
    
    from vmevalkit.tasks.your_task.PROMPTS import PROMPTS
    
    assert len(PROMPTS) > 0
    assert all(isinstance(p, str) for p in PROMPTS)
    
    # Test prompt formatting if using parameters
    prompt = PROMPTS[0].format(difficulty="easy")
    assert "easy" in prompt
```

### Integration Test

```python
# integration_test.py
from vmevalkit.runner.create_dataset import main

def test_full_integration():
    """Test complete dataset generation pipeline."""
    
    # Run generation
    args = ["--pairs-per-domain", "10", "--output", "test_output"]
    main(args)
    
    # Verify output files
    from pathlib import Path
    import json
    
    dataset_file = Path("data/questions/vmeval_dataset.json")
    assert dataset_file.exists()
    
    with open(dataset_file) as f:
        dataset = json.load(f)
    
    # Check your task is included
    your_tasks = [p for p in dataset["pairs"] if p["domain"] == "your_task"]
    assert len(your_tasks) > 0
    
    # Check directory structure
    task_dir = Path("data/questions/your_task_task")
    assert task_dir.exists()
    
    # Check individual question folders
    for task in your_tasks:
        question_dir = task_dir / task["id"]
        assert question_dir.exists()
        assert (question_dir / "first_frame.png").exists()
        assert (question_dir / "final_frame.png").exists()
        assert (question_dir / "question_metadata.json").exists()
```

### Manual Testing Checklist

✅ **Generate Small Dataset**
```bash
python vmevalkit/runner/create_dataset.py --pairs-per-domain 5
```

✅ **Verify File Structure**
```bash
ls -la data/questions/your_task_task/
ls -la data/questions/your_task_task/your_task_0000/
```

✅ **Check Image Quality**
```bash
# Open images to verify visual quality
open data/questions/your_task_task/your_task_0000/first_frame.png
open data/questions/your_task_task/your_task_0000/final_frame.png
```

✅ **Validate JSON Structure**
```bash
python -c "
import json
with open('data/questions/vmeval_dataset.json') as f:
    data = json.load(f)
    your_tasks = [p for p in data['pairs'] if p['domain'] == 'your_task']
    print(f'Found {len(your_tasks)} your_task pairs')
    print('Sample:', json.dumps(your_tasks[0], indent=2))
"
```

✅ **Test with Inference**
```python
from vmevalkit.runner.inference import InferenceRunner

runner = InferenceRunner()
result = runner.run(
    model_name="luma-ray-2",
    image_path="data/questions/your_task_task/your_task_0000/first_frame.png",
    text_prompt="Your task prompt here"
)
print(f"Video generated: {result.get('video_path')}")
```

## 🔧 Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError` | Task not properly registered | Add to `DOMAIN_REGISTRY` |
| Images not generated | Missing matplotlib | `pip install matplotlib` |
| Wrong image format | Using JPEG/JPG instead of PNG | Always use `.png` extension |
| Missing `create_dataset` | Function not implemented | Ensure function exists and is exported |
| Import errors | Wrong module structure | Check `__init__.py` exports |
| Images too large | High DPI or large figure size | Use `figsize=(6,6), dpi=150` |
| Temp files accumulating | Not using temp directory | Use `tempfile.mkdtemp()` |
| Inconsistent IDs | Manual ID generation | Use `f"{task}_{i:04d}"` format |
| Missing metadata | Incomplete task pairs | Include all required fields |

## 📚 Advanced Patterns

### Using External Libraries

If your task needs specialized libraries:

```python
# At the top of your_reasoning.py
try:
    import specialized_library
    HAS_SPECIALIZED = True
except ImportError:
    HAS_SPECIALIZED = False
    print("Warning: specialized_library not available. Install with: pip install specialized_library")

def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    if not HAS_SPECIALIZED:
        raise ImportError("This task requires specialized_library. Please install it first.")
    
    # Your implementation
```

### Caching Generated Tasks

For expensive generation:

```python
import pickle
from pathlib import Path

CACHE_DIR = Path("data/cache/your_task")
CACHE_DIR.mkdir(exist_ok=True, parents=True)

def create_dataset(num_samples: int = 50, use_cache: bool = True) -> Dict[str, Any]:
    cache_file = CACHE_DIR / f"dataset_{num_samples}.pkl"
    
    if use_cache and cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Generate dataset
    dataset = generate_dataset_internal(num_samples)
    
    # Cache for reuse
    with open(cache_file, 'wb') as f:
        pickle.dump(dataset, f)
    
    return dataset
```

### Multiple Difficulty Strategies

```python
def generate_by_difficulty(difficulty: str) -> Dict[str, Any]:
    """Generate task with specific difficulty."""
    
    strategies = {
        "easy": {
            "grid_size": 3,
            "num_obstacles": 2,
            "solution_length": 3
        },
        "medium": {
            "grid_size": 5,
            "num_obstacles": 5,
            "solution_length": 7
        },
        "hard": {
            "grid_size": 7,
            "num_obstacles": 10,
            "solution_length": 12
        }
    }
    
    params = strategies[difficulty]
    return generate_with_params(**params)
```

### Validating Solutions

```python
def validate_task_pair(task_data: Dict[str, Any]) -> bool:
    """Ensure task has valid solution."""
    
    # Check problem is solvable
    if not is_solvable(task_data["problem"]):
        return False
    
    # Verify solution is correct
    if not verify_solution(task_data["problem"], task_data["solution"]):
        return False
    
    # Check solution is unique (optional)
    if require_unique and count_solutions(task_data["problem"]) > 1:
        return False
    
    return True

def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    pairs = []
    attempts = 0
    max_attempts = num_samples * 10
    
    while len(pairs) < num_samples and attempts < max_attempts:
        attempts += 1
        
        task_data = generate_single_task()
        
        if validate_task_pair(task_data):
            pair = create_task_pair(task_data, f"task_{len(pairs):04d}")
            pairs.append(pair)
    
    if len(pairs) < num_samples:
        print(f"Warning: Only generated {len(pairs)} valid tasks out of {num_samples} requested")
    
    return {"pairs": pairs}
```

## 🎯 Task Design Guidelines

### Good Task Characteristics

✅ **Clear Visual Representation**: Problem and solution are visually distinct  
✅ **Unambiguous Goal**: The objective is clear from the prompt and images  
✅ **Verifiable Solution**: Can determine if the solution is correct  
✅ **Progressive Difficulty**: Easy/medium/hard variants possible  
✅ **Reasoning Focus**: Tests specific cognitive abilities

### Task Categories to Consider

1. **Spatial Reasoning**: Navigation, rotation, spatial transformations
2. **Logical Deduction**: Puzzles, constraints, rule-following
3. **Pattern Recognition**: Sequences, matrices, abstract patterns  
4. **Strategic Planning**: Games, optimization, resource management
5. **Mathematical Reasoning**: Arithmetic, geometry, algebra
6. **Visual Understanding**: Object manipulation, scene understanding

### Example Task Ideas

```python
TASK_IDEAS = {
    "tangram": "Arrange shapes to form a target silhouette",
    "tower_of_hanoi": "Move disks between pegs following rules",
    "sliding_puzzle": "Rearrange tiles to form target image",
    "connect_dots": "Draw lines to connect numbered dots in order",
    "pathfinding": "Find optimal path avoiding obstacles",
    "balance_scale": "Balance objects on a scale",
    "gear_rotation": "Predict gear movement directions",
    "water_pouring": "Transfer liquids between containers",
    "circuit_completion": "Complete electrical circuit",
    "domino_chain": "Arrange dominoes to create chain reaction"
}
```

## 📋 Complete Task Checklist

Before submitting your task, ensure:

### Code Structure
- [ ] Created `vmevalkit/tasks/{task_name}_task/` directory
- [ ] Implemented `{task_name}_reasoning.py` with `create_dataset()`
- [ ] Created `PROMPTS.py` with prompt templates
- [ ] Added `__init__.py` with proper exports
- [ ] Created `{TASK_NAME}.md` documentation
- [ ] Added entry to `DOMAIN_REGISTRY`

### Functionality
- [ ] `create_dataset(num_samples)` works correctly
- [ ] Images saved as PNG format
- [ ] Images named `first_frame.png` and `final_frame.png`
- [ ] All required JSON fields included
- [ ] Temporary directories used for image generation
- [ ] Task IDs follow `{task}_{id:04d}` format

### Quality
- [ ] Images are clear and consistent
- [ ] Prompts are unambiguous
- [ ] Multiple difficulty levels supported
- [ ] Solutions are verifiable
- [ ] Code is well-documented
- [ ] Tests pass successfully

### Integration
- [ ] Dataset generation script runs without errors
- [ ] Files created in correct directory structure
- [ ] Task appears in master dataset
- [ ] Can run inference on generated tasks

## 🚀 Advanced Integration Features

### Custom Evaluation Metrics

```python
def evaluate_solution(generated_video_path: str, task_pair: Dict) -> Dict[str, Any]:
    """Custom evaluation logic for your task."""
    
    # Extract final frame from generated video
    final_frame = extract_last_frame(generated_video_path)
    
    # Compare with ground truth
    solution_image = load_image(task_pair["final_image_path"])
    
    # Calculate similarity metrics
    similarity = calculate_similarity(final_frame, solution_image)
    
    # Task-specific validation
    is_valid = validate_your_task_solution(final_frame, task_pair["your_task_data"])
    
    return {
        "similarity_score": similarity,
        "is_valid_solution": is_valid,
        "task_specific_metrics": {
            # Add your metrics
        }
    }
```

### Batch Generation Optimization

```python
def create_dataset(num_samples: int = 50, batch_size: int = 10) -> Dict[str, Any]:
    """Generate dataset in batches for better performance."""
    
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    
    pairs = []
    
    def generate_batch(start_idx: int, size: int) -> List[Dict]:
        batch_pairs = []
        for i in range(size):
            task_id = f"your_task_{start_idx + i:04d}"
            task_data = generate_single_task(start_idx + i)
            pair = create_task_pair(task_data, task_id)
            batch_pairs.append(pair)
        return batch_pairs
    
    # Parallel generation
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(0, num_samples, batch_size):
            size = min(batch_size, num_samples - i)
            future = executor.submit(generate_batch, i, size)
            futures.append(future)
        
        for future in futures:
            batch_pairs = future.result()
            pairs.extend(batch_pairs)
    
    return {"pairs": pairs}
```

## 🎓 Summary

Adding a task to VMEvalKit is straightforward:

1. **Register** in `DOMAIN_REGISTRY`
2. **Create** module with `create_dataset()` function
3. **Generate** PNG images for first/final frames
4. **Include** all required metadata fields
5. **Test** thoroughly

The system handles all the integration complexity - you just need to focus on generating interesting reasoning challenges!

---

Ready to add your reasoning task? Follow this guide and contribute to advancing video model evaluation! 🚀