# Tin's Tasks Integration Plan

**Date**: November 29, 2025  
**Source Repository**: https://github.com/tin-xai/simple_task_video_reasoning.git  
**Target**: VMEvalKit  

---

## 🎯 Executive Summary

Integrate 4 visual reasoning tasks from Tin's repository into VMEvalKit, adding **counting** and **pathfinding** capabilities to the benchmark suite. These tasks test fundamental visual reasoning skills where AI models have demonstrated failure cases.

---

## 📊 Task Overview

| Task Name | Type | Reasoning Skill | Difficulty | Priority |
|-----------|------|-----------------|------------|----------|
| **Counting Circles** | Local Gen | Visual counting, attention | Easy-Medium | High |
| **Counting Squares** | Local Gen | Nested counting, recursion | Medium-Hard | High |
| **Letter Counting** | Local Gen | Text recognition, counting | Easy-Medium | Medium |
| **Subway Pathfinding** | Local Gen | Spatial navigation, tracking | Hard | High |

---

## 🗂️ Proposed VMEvalKit Structure

```
vmevalkit/tasks/
├── counting_circles_task/
│   ├── __init__.py
│   ├── counting_circles_reasoning.py
│   ├── PROMPTS.py
│   └── COUNTING_CIRCLES.md
│
├── counting_squares_task/
│   ├── __init__.py
│   ├── counting_squares_reasoning.py
│   ├── PROMPTS.py
│   └── COUNTING_SQUARES.md
│
├── letter_counting_task/
│   ├── __init__.py
│   ├── letter_counting_reasoning.py
│   ├── PROMPTS.py
│   └── LETTER_COUNTING.md
│
└── subway_pathfinding_task/
    ├── __init__.py
    ├── subway_pathfinding_reasoning.py
    ├── PROMPTS.py
    └── SUBWAY_PATHFINDING.md
```

---

## 📝 Detailed Implementation Plan

### **Task 1: Counting Circles** 
**Priority**: HIGH  
**Source**: `CountingCircles/create_circles.py` + `create_pentagons.py`

#### Reasoning Tested
- Visual object counting
- Attention to overlapping objects
- Shape discrimination (circles vs pentagons)

#### Implementation Steps
1. **Create Module**: `vmevalkit/tasks/counting_circles_task/`
2. **Adapt Generation Logic**:
   - Port circle/pentagon drawing functions from Tin's code
   - Maintain Olympic-logo-like arrangements
   - Support configurable: DPI, radius, thickness, colors, shape type
3. **PROMPTS.py**:
   ```python
   PROMPTS = [
       "Count the total number of {shape}s in the image. Show the counting process.",
       "How many {shape}s are in this image? Demonstrate step-by-step counting.",
   ]
   ```
4. **Difficulty Levels**:
   - Easy: 3-5 circles, single color, no overlap
   - Medium: 5-7 circles, multiple colors, slight overlap
   - Hard: 7-9 circles, complex arrangements, overlapping
5. **Output Format**:
   - First frame: Shapes only
   - Final frame: Shapes + count overlay
   - Metadata: shape type, count, positions, colors
6. **Register in TASK_CATALOG.py**:
   ```python
   'counting_circles': {
       'name': 'Counting Circles',
       'description': 'Visual counting of geometric shapes (circles/pentagons)',
       'module': 'vmevalkit.tasks.counting_circles_task',
       'create_function': 'create_dataset',
       'process_dataset': lambda dataset, num_samples: dataset['pairs']
   }
   ```

#### Evaluation Criteria
- **Exact Count Match**: Must output exact number
- **Process Demonstration**: Should show counting method (e.g., highlighting each)
- **Score**: Binary (correct=1, incorrect=0)

---

### **Task 2: Counting Squares**
**Priority**: HIGH  
**Source**: `CountingSquares/create_squares.py`

#### Reasoning Tested
- Recursive/nested structure understanding
- Multi-level counting (including all inner squares)
- Spatial relationship reasoning

#### Implementation Steps
1. **Create Module**: `vmevalkit/tasks/counting_squares_task/`
2. **Adapt Generation Logic**:
   - Port recursive square generation with random offsets
   - Support depth levels (2-5 nested squares)
   - Vary line thickness, initial size, reduction factor
3. **PROMPTS.py**:
   ```python
   PROMPTS = [
       "Count all nested squares including inner squares. Show the counting process.",
       "How many squares are in this nested structure? Count all layers.",
   ]
   ```
4. **Difficulty Levels**:
   - Easy: 2-3 depth (3-4 total squares)
   - Medium: 3-4 depth (4-5 total squares)
   - Hard: 4-5 depth (5-6 total squares)
5. **Output Format**:
   - First frame: Nested squares only
   - Final frame: Squares + count overlay
   - Metadata: depth, all square positions/sizes
6. **Register in TASK_CATALOG.py**:
   ```python
   'counting_squares': {
       'name': 'Counting Nested Squares',
       'description': 'Count all squares in recursively nested structures',
       'module': 'vmevalkit.tasks.counting_squares_task',
       'create_function': 'create_dataset',
       'process_dataset': lambda dataset, num_samples: dataset['pairs']
   }
   ```

#### Evaluation Criteria
- **Exact Count Match**: Must count ALL nested squares (not just outermost)
- **Recursive Understanding**: Tests if model understands nested structures
- **Score**: Binary with partial credit for off-by-one errors

---

### **Task 3: Letter Counting**
**Priority**: MEDIUM  
**Source**: `FindingWords/create_strings.py`

#### Reasoning Tested
- Text/character recognition
- Pattern matching and counting
- Attention to specific features

#### Implementation Steps
1. **Create Module**: `vmevalkit/tasks/letter_counting_task/`
2. **Adapt Generation Logic**:
   - Port word display and letter highlighting
   - Use challenging words (STRAWBERRY, MISSISSIPPI, etc.)
   - Support multiple DPI settings
3. **PROMPTS.py**:
   ```python
   PROMPTS = [
       "Count the number of '{letter}' in the word '{word}'. Show each occurrence.",
       "How many times does the letter '{letter}' appear in '{word}'?",
   ]
   ```
4. **Difficulty Levels**:
   - Easy: Short words (5-7 letters), single occurrence
   - Medium: Medium words (8-12 letters), 2-3 occurrences
   - Hard: Long words (13+ letters), repeated letters (MISSISSIPPI)
5. **Output Format**:
   - First frame: Word displayed
   - Final frame: Word + circles around target letters + count
   - Metadata: word, target letter, ground truth count
6. **Register in TASK_CATALOG.py**:
   ```python
   'letter_counting': {
       'name': 'Letter Counting',
       'description': 'Count occurrences of specific letters in words',
       'module': 'vmevalkit.tasks.letter_counting_task',
       'create_function': 'create_dataset',
       'process_dataset': lambda dataset, num_samples: dataset['pairs']
   }
   ```

#### Evaluation Criteria
- **Exact Count Match**: Must output exact number of letter occurrences
- **Case Insensitivity**: Should handle uppercase/lowercase correctly
- **Score**: Binary (correct=1, incorrect=0)

---

### **Task 4: Subway Pathfinding**
**Priority**: HIGH  
**Source**: `PathFinding/create_subway.py`

#### Reasoning Tested
- Spatial navigation and path tracking
- Multi-step sequential reasoning
- Color/path discrimination in complex networks

#### Implementation Steps
1. **Create Module**: `vmevalkit/tasks/subway_pathfinding_task/`
2. **Adapt Generation Logic**:
   - Port subway network generation (4 stations: A, B, C, D)
   - Generate multiple colored paths between stations
   - Support varying line thickness and image size
3. **PROMPTS.py**:
   ```python
   PROMPTS = [
       "An agent travels from station {source} to station {dest}. Show the path taken.",
       "Trace the route from station {source} to station {dest} using the {color} line.",
       "Which path connects station {source} to station {dest}?",
   ]
   ```
4. **Difficulty Levels**:
   - Easy: 1-2 path segments, clear color distinction
   - Medium: 2-3 path segments, multiple colors
   - Hard: 3+ path segments, overlapping paths
5. **Output Format**:
   - First frame: Complete subway map
   - Final frame: Map + agent icon at destination + highlighted path
   - Metadata: source, destination, path color, coordinates
6. **Register in TASK_CATALOG.py**:
   ```python
   'subway_pathfinding': {
       'name': 'Subway Pathfinding',
       'description': 'Navigate and track paths through subway networks',
       'module': 'vmevalkit.tasks.subway_pathfinding_task',
       'create_function': 'create_dataset',
       'process_dataset': lambda dataset, num_samples: dataset['pairs']
   }
   ```

#### Evaluation Criteria
- **Correct Destination**: Agent reaches correct station
- **Correct Path**: Uses appropriate colored line/path
- **Path Continuity**: Path segments are connected
- **Score**: Multi-level (destination=0.5, correct path=0.5)

---

## 🔧 Technical Implementation Details

### Code Adaptation Strategy

#### 1. **Image Generation Functions**
Refactor Tin's matplotlib code to match VMEvalKit standards:

```python
def create_image(data: Any, output_path: Path, add_solution: bool = False):
    """
    Standard image generation following VMEvalKit conventions.
    
    Args:
        data: Task-specific data structure
        output_path: Path to save PNG image
        add_solution: If True, add answer overlay (for final frame)
    """
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Task-specific drawing logic here
    # ...
    
    if add_solution:
        # Add count/answer overlay
        pass
    
    plt.savefig(output_path, bbox_inches='tight', dpi=150, pad_inches=0)
    plt.close(fig)
```

#### 2. **Data Structure**
Use Pydantic models (as per user rules):

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class CountingCirclesData(BaseModel):
    """Data for counting circles task."""
    shape_type: str = Field(description="'circle' or 'pentagon'")
    num_shapes: int = Field(ge=1, le=10)
    centers: List[List[float]]
    radius: float = Field(gt=0)
    colors: List[str]
    dpi: int = Field(ge=100, le=300)
    thickness: float = Field(gt=0)
    
class TaskPair(BaseModel):
    """Standard task pair for VMEvalKit."""
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

#### 3. **Create Dataset Function Template**

```python
def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    """
    Generate {task_name} dataset.
    
    Args:
        num_samples: Number of task pairs to generate
        
    Returns:
        Dataset dictionary with 'pairs' list
    """
    pairs = []
    
    for i in range(num_samples):
        # Determine difficulty based on sample index
        difficulty = determine_difficulty(i, num_samples)
        
        # Generate task-specific data
        task_data = generate_task_data(difficulty)
        
        # Create temporary files for images
        temp_dir = tempfile.mkdtemp()
        task_id = f"{TASK_NAME}_{i:04d}"
        first_path = Path(temp_dir) / f"{task_id}_first.png"
        final_path = Path(temp_dir) / f"{task_id}_final.png"
        
        # Generate images
        create_image(task_data, first_path, add_solution=False)
        create_image(task_data, final_path, add_solution=True)
        
        # Create task pair
        pair = TaskPair(
            id=task_id,
            prompt=format_prompt(task_data),
            first_image_path=str(first_path),
            final_image_path=str(final_path),
            domain=TASK_NAME,
            task_category=TASK_CATEGORY,
            difficulty=difficulty,
            ground_truth=task_data.ground_truth,
            metadata=task_data.model_dump()
        )
        
        pairs.append(pair.model_dump())
    
    return {
        "name": f"{TASK_NAME}_tasks",
        "pairs": pairs,
        "source": "tin_tasks",
        "total_samples": len(pairs)
    }
```

---

## 📋 Evaluation Guidance

### Update Required Files

**File**: `vmevalkit/eval/gpt4o_eval.py`  
**File**: `vmevalkit/eval/internvl.py`

Add to `TASK_GUIDANCE` dictionary:

```python
TASK_GUIDANCE = {
    # ... existing tasks ...
    
    'counting_circles': {
        'description': 'Count geometric shapes (circles or pentagons)',
        'scoring': '''
        Award 1 point if:
        - The exact count is provided in the final frame
        - The count matches the ground truth
        
        Award 0 points if:
        - Count is incorrect
        - No count is shown
        - Count is ambiguous
        
        Partial credit (0.5 points) if off by one.
        ''',
        'key_aspects': ['exact count', 'all shapes identified', 'clear visualization']
    },
    
    'counting_squares': {
        'description': 'Count all nested squares including inner squares',
        'scoring': '''
        Award 1 point if:
        - All nested squares are counted (not just outermost)
        - The exact total count matches ground truth
        
        Award 0.5 points if:
        - Off by one square (common mistake)
        
        Award 0 points if:
        - Only counts outermost squares
        - Count is significantly incorrect (off by 2+)
        ''',
        'key_aspects': ['recursive counting', 'all layers identified', 'total count accuracy']
    },
    
    'letter_counting': {
        'description': 'Count occurrences of a specific letter in a word',
        'scoring': '''
        Award 1 point if:
        - The exact count of target letter is correct
        - All occurrences are identified
        
        Award 0 points if:
        - Count is incorrect
        - Case sensitivity error (if applicable)
        - Misidentified letters
        ''',
        'key_aspects': ['exact count', 'case handling', 'all occurrences marked']
    },
    
    'subway_pathfinding': {
        'description': 'Navigate from source to destination through subway network',
        'scoring': '''
        Award 1 point if:
        - Agent reaches correct destination station
        - Path taken is valid and continuous
        - Correct colored line is followed
        
        Award 0.5 points if:
        - Correct destination but suboptimal path
        
        Award 0 points if:
        - Wrong destination
        - Discontinuous path
        - Invalid route (not following lines)
        ''',
        'key_aspects': ['correct destination', 'valid path', 'path continuity', 'color matching']
    }
}
```

---

## 🎯 Implementation Timeline

### Phase 1: Foundation (Week 1)
- [ ] Set up module directories
- [ ] Create __init__.py files
- [ ] Write PROMPTS.py for each task
- [ ] Define Pydantic data models

### Phase 2: Core Implementation (Week 2-3)
- [ ] Implement Counting Circles task
- [ ] Implement Counting Squares task
- [ ] Implement Letter Counting task
- [ ] Implement Subway Pathfinding task

### Phase 3: Integration (Week 4)
- [ ] Register all tasks in TASK_CATALOG.py
- [ ] Update evaluation guidance files
- [ ] Create documentation (.md files)
- [ ] Write unit tests

### Phase 4: Testing & Validation (Week 5)
- [ ] Generate test datasets (5-10 samples each)
- [ ] Verify image quality and formats
- [ ] Validate metadata structure
- [ ] Test with sample video models
- [ ] Review failure cases

### Phase 5: Production Deployment (Week 6)
- [ ] Generate full datasets (50-100 samples each)
- [ ] Performance optimization
- [ ] Documentation finalization
- [ ] Integration testing with full VMEvalKit pipeline

---

## ✅ Quality Checklist

### Per-Task Requirements
- [ ] Module folder created in `vmevalkit/tasks/`
- [ ] `__init__.py` exports `create_dataset`
- [ ] `PROMPTS.py` with at least 2 prompt variations
- [ ] Main reasoning file with complete implementation
- [ ] Documentation markdown file
- [ ] Entry in `DOMAIN_REGISTRY`
- [ ] Evaluation guidance added to eval files
- [ ] Images are PNG format, ~400x400px
- [ ] Uses Pydantic models (no plain dicts)
- [ ] No try-catch blocks (per user rules)
- [ ] Proper typing with type hints
- [ ] Tests run successfully

### Dataset Quality
- [ ] Clear visual distinction between first/final frames
- [ ] Unambiguous ground truth
- [ ] Multiple difficulty levels
- [ ] High contrast, readable images
- [ ] Consistent image dimensions
- [ ] Complete metadata
- [ ] Valid JSON structure

---

## 🚀 Quick Start Commands

```bash
# 1. Generate datasets for all Tin's tasks
python vmevalkit/runner/create_dataset.py \
    --domains counting_circles,counting_squares,letter_counting,subway_pathfinding \
    --pairs-per-domain 50

# 2. Verify generated data
ls -R data/questions/counting_circles_task/
ls -R data/questions/counting_squares_task/
ls -R data/questions/letter_counting_task/
ls -R data/questions/subway_pathfinding_task/

# 3. Test with sample model
python vmevalkit/runner/evaluate.py \
    --model gpt4o \
    --tasks counting_circles \
    --num-samples 5

# 4. Run full evaluation
python vmevalkit/runner/evaluate.py \
    --model gpt4o \
    --tasks counting_circles,counting_squares,letter_counting,subway_pathfinding \
    --num-samples 50
```

---

## 📊 Expected Outcomes

### Dataset Statistics (per task)
- **Counting Circles**: 50-100 samples
  - Easy: 30%, Medium: 40%, Hard: 30%
  - Mix of circles and pentagons
  - Expected model accuracy: 60-80%

- **Counting Squares**: 50-100 samples
  - Easy: 30%, Medium: 40%, Hard: 30%
  - Depths 2-5
  - Expected model accuracy: 40-60% (harder due to recursion)

- **Letter Counting**: 100-150 samples
  - Easy: 30%, Medium: 40%, Hard: 30%
  - 25+ unique words
  - Expected model accuracy: 70-85%

- **Subway Pathfinding**: 50-100 samples
  - Easy: 30%, Medium: 40%, Hard: 30%
  - All station combinations
  - Expected model accuracy: 50-70%

### Impact
- Adds **4 new reasoning domains** to VMEvalKit
- Covers **counting** and **pathfinding** capabilities
- Documents **known failure cases** for AI models
- Provides **benchmarks** for visual reasoning improvements

---

## 🔍 Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Image generation performance | Medium | Use efficient matplotlib settings, cache where possible |
| Evaluation ambiguity | High | Clear scoring rubrics, ground truth validation |
| Task too easy/hard | Medium | Multiple difficulty levels, pilot testing |
| Integration conflicts | Low | Follow VMEvalKit conventions strictly |
| Memory usage | Low | Use tempfile, close plt figures properly |

---

## 📚 References

- **Source Repository**: https://github.com/tin-xai/simple_task_video_reasoning
- **VMEvalKit Docs**: `/home/hokindeng/VMEvalKit/docs/ADDING_TASKS.md`
- **Task Catalog**: `/home/hokindeng/VMEvalKit/vmevalkit/runner/TASK_CATALOG.py`
- **Example Tasks**: 
  - Local: `vmevalkit/tasks/sudoku_task/`
  - External: `vmevalkit/tasks/external/videothinkbench_arc_agi_task/`

---

## 🎓 Next Steps

1. **Review this plan** with team/stakeholders
2. **Prioritize tasks** based on evaluation needs
3. **Begin Phase 1** implementation
4. **Iterate** based on testing feedback
5. **Deploy** to production

---

**Plan Version**: 1.0  
**Last Updated**: November 29, 2025  
**Status**: Ready for Implementation

