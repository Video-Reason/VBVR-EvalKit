# Raven's Progressive Matrices Task Documentation

## Overview

The Raven's Progressive Matrices (RPM) task evaluates video generation models' ability to demonstrate abstract reasoning and pattern recognition by completing visual matrix puzzles. This task tests fundamental cognitive capabilities including pattern detection, logical inference, and abstract rule application - core components of visual intelligence.

### Implementation Philosophy
The Raven task implementation follows VMEvalKit's core design principles:
- **Procedural Generation**: Uses the `RPMPuzzleGenerator` class with seeded randomization for reproducible puzzle creation
- **Modular Architecture**: Separates puzzle generation (`rpm_generator.py`), dataset creation (`raven_reasoning.py`), and prompts (`PROMPTS.py`)
- **Standardized Integration**: Implements `create_dataset()` interface for seamless VMEvalKit runner integration
- **Dynamic Pattern Creation**: Five distinct pattern types with algorithmic generation ensuring unique, solvable puzzles

## Task Description

### Core Challenge
Models must:
1. **Pattern Recognition**: Identify the underlying rule governing the matrix
2. **Abstract Reasoning**: Apply the discovered rule to determine the missing element
3. **Visual Completion**: Generate a video showing the reasoning from incomplete to complete matrix
4. **Logical Consistency**: Ensure the solution follows the established pattern

### Visual Format
- **3x3 Matrix Grid**: Nine cells arranged in a square grid (450x450 pixels total)
- **Missing Element**: Bottom-right cell (position [2,2]) is always the unknown
- **Pattern Types**: Shapes, numbers, rotations, colors, and combinations
- **Clean Visualization**: High-contrast black and white design with clear geometric shapes

### Implementation Architecture
The task implementation consists of three core components:

1. **`RPMPuzzleGenerator` Class**: Core pattern generation engine
   - Manages tile rendering (150x150 pixels per cell)
   - Implements pattern generation algorithms
   - Handles shape drawing with PIL/ImageDraw
   - Provides seeded randomization for reproducibility

2. **Pattern Generation Pipeline**:
   - Random pattern type selection from 5 categories
   - Matrix data structure generation following selected rule
   - Cell rendering with shape positioning logic
   - Question mark placeholder for missing cell

3. **Dataset Creation Workflow**:
   - Temporary file generation in system temp directory
   - Standardized naming convention (`raven_XXXX_first.png`, `raven_XXXX_final.png`)
   - Metadata generation with rule descriptions
   - Integration with VMEvalKit's folder structure

## Data Structure

### Internal Matrix Representation
The `RPMPuzzleGenerator` uses a cell-based data structure for matrix representation:

```python
# Each cell in the 3x3 matrix contains:
cell_data = {
    "shapes": List[str],        # Shape types to draw (e.g., ["circle", "square"])
    "positions": List[str],     # Position of each shape ("center", "left", "right")
    "colors": List[str],        # Color for each shape ("black", "gray", "darkgray")
    "fills": List[Optional[str]] # Fill color (None, "lightgray", "white")
}

# Full matrix structure
matrix = [
    [cell_0_0, cell_0_1, cell_0_2],
    [cell_1_0, cell_1_1, cell_1_2],
    [cell_2_0, cell_2_1, cell_2_2]  # cell_2_2 is the missing element
]
```

### RavenTaskPair
Each generated task follows this exact structure:

```python
{
    "id": "raven_0014",                     # Format: raven_{4-digit-index}
    "prompt": "This is Raven's Progressive Matrices like task. Complete the missing pattern in this 3x3 matrix.",
    "first_image_path": "raven_task/raven_0014/first_frame.png",  # Relative path in questions folder
    "final_image_path": "raven_task/raven_0014/final_frame.png",  # Complete solution path
    "domain": "raven",                      # Fixed domain identifier
    "task_category": "AbstractReasoning",   # Fixed category
    "difficulty": "hard",                    # Determined by rule_type mapping
    "raven_data": {                         # Task-specific metadata
        "rule": "Combination pattern",      # Human-readable rule description
        "rule_type": "combination",         # Algorithmic pattern type
        "matrix_size": "3x3",               # Fixed for current implementation
        "seed": 2039                        # Seed = 2025 + task_index
    },
    "created_at": "2025-10-15T22:14:40.537472Z"  # ISO timestamp with timezone
}
```

### RavenDataset
The dataset structure returned by `create_dataset()`:

```python
{
    "name": "raven_tasks",
    "description": "Raven Progressive Matrices visual reasoning tasks (50 pairs)",
    "created_at": "2025-10-15T22:14:40.537472Z",
    "total_pairs": 50,
    "pairs": [RavenTaskPair, ...],
    "generation_info": {
        "generator": "RPMPuzzleGenerator",
        "tile_size": 150,                  # Pixels per matrix cell
        "matrix_size": "3x3",
        "difficulty_distribution": {
            "easy": 20,                     # shape_progression, color_pattern
            "medium": 20,                   # number_progression, rotation
            "hard": 10                      # combination patterns
        }
    }
}
```

### Difficulty Mapping
The implementation uses a hardcoded mapping for difficulty assignment:

```python
difficulty_map = {
    "shape_progression": "easy",    # Simple sequential patterns
    "color_pattern": "easy",        # Fill/color variations
    "number_progression": "medium",  # Requires counting
    "rotation": "medium",           # Spatial transformation
    "combination": "hard"           # Multiple simultaneous rules
}
```

## Pattern Types

### 1. Shape Progression (`_generate_shape_progression`)
- **Algorithm**: Randomly samples 3 shapes, applies them consistently by row or column
- **Implementation**:
  ```python
  shapes = self.rng.sample(self.shapes, 3)  # e.g., ["circle", "square", "triangle"]
  direction = self.rng.choice(["row", "column"])
  # If row: each row has same sequence [shape1, shape2, shape3]
  # If column: each column has same shape throughout
  ```
- **Example Matrix** (row progression):
  ```
  [Circle] [Square] [Triangle]
  [Circle] [Square] [Triangle]
  [Circle] [Square] [?]
  ```
- **Difficulty**: Easy
- **Shape Pool**: `["circle", "square", "triangle", "diamond", "cross", "star"]`

### 2. Number Progression (`_generate_number_progression`)
- **Algorithm**: Uses modulo arithmetic to determine shape count based on position
- **Implementation**:
  ```python
  num_shapes = ((i + j) % 3) + 1  # Results in 1, 2, or 3 shapes
  # Positions: 1 shape = "center", 2 shapes = ["left", "right"], 3 shapes = ["left", "center", "right"]
  ```
- **Example Matrix**:
  ```
  [1 shape] [2 shapes] [3 shapes]
  [2 shapes] [3 shapes] [1 shape]
  [3 shapes] [1 shape] [?]       # Should be 2 shapes
  ```
- **Difficulty**: Medium (requires counting)
- **Visual Layout**: Shapes distributed horizontally within cell

### 3. Rotation Pattern (`_generate_rotation_pattern`)
- **Algorithm**: Cycles through 3 randomly selected shapes using position-based indexing
- **Implementation**:
  ```python
  shapes_set = self.rng.sample(self.shapes, 3)
  idx = (i + j) % 3  # Rotation index based on position
  shape = shapes_set[idx]
  ```
- **Example Matrix** (with shapes A, B, C):
  ```
  [A] [B] [C]
  [B] [C] [A]
  [C] [A] [?]  # Should be B
  ```
- **Difficulty**: Medium (spatial transformation understanding)
- **Note**: Despite the name, this implements shape cycling, not geometric rotation

### 4. Color/Fill Pattern (`_generate_color_pattern`)
- **Algorithm**: Cycles through fill patterns based on cell position
- **Implementation**:
  ```python
  fills = self.rng.sample(self.fills, 3)  # e.g., [None, "lightgray", "white"]
  fill_idx = (i + j) % 3
  # Same shape, different fills
  ```
- **Example Matrix**:
  ```
  [Empty] [Light] [White]
  [Light] [White] [Empty]
  [White] [Empty] [?]     # Should be Light fill
  ```
- **Difficulty**: Easy
- **Fill Options**: `[None, "lightgray", "white"]`

### 5. Combination Pattern (`_generate_combination_pattern`)
- **Algorithm**: Complex rule combining shape selection and quantity based on position
- **Implementation**:
  ```python
  # Even positions (i+j)%2==0: shape[0], odd positions: shape[1]
  # Corner cells get additional shapes
  if i != 1 and j != 1:  # Corners
      cell_shapes.append(shapes[(i + j) % 2])
  ```
- **Example Matrix**:
  ```
  [2 shapes] [1 shape] [2 shapes]
  [1 shape]  [1 shape] [1 shape]
  [2 shapes] [1 shape] [?]        # Should be 2 shapes (corner position)
  ```
- **Difficulty**: Hard (multiple simultaneous rules)
- **Challenge**: Requires identifying both shape alternation AND position-based quantity rules

## Visual Elements

### Shape Drawing Implementation
The `create_shape()` function in `rpm_generator.py` implements precise geometric rendering:

```python
def create_shape(draw, shape_type, bbox, color="black", fill=None):
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center point
    radius = min(x2 - x1, y2 - y1) // 3       # Shape size
```

#### Shape Specifications
- **Circle**: `draw.ellipse()` with radius = tile_size/3
- **Square**: `draw.rectangle()` with equal sides
- **Triangle**: Upward-pointing with 3 vertices
- **Diamond**: 4-point polygon rotated 45°
- **Cross**: Two perpendicular lines (width=3)
- **Star**: 8-point star with alternating radii (r, r/2)

### Visual Properties
- **Color Palette**: 
  - Outlines: `["black", "gray", "darkgray"]`
  - Fills: `[None, "lightgray", "white"]`
- **Line Width**: 
  - Shape outlines: 2 pixels
  - Cross lines: 3 pixels
  - Cell borders: 1 pixel
- **Positioning System**:
  - `"center"`: bbox = (tile/4, tile/4, 3*tile/4, 3*tile/4)
  - `"left"`: bbox = (tile/8, tile/4, 3*tile/8, 3*tile/4)
  - `"right"`: bbox = (5*tile/8, tile/4, 7*tile/8, 3*tile/4)

### Matrix Rendering
- **Cell Size**: 150x150 pixels (configurable via `tile_size`)
- **Matrix Size**: 450x450 pixels (3x3 grid)
- **Missing Cell**: Shows "?" in gray at position [2,2]
- **Background**: White (#FFFFFF)
- **Cell Borders**: Black 1px outline for each cell

## Prompts

### Standardized Prompt
The Raven task uses a single standardized prompt:

**"This is Raven's Progressive Matrices like task. Complete the missing pattern in this 3x3 matrix."**

This prompt:
- Clearly identifies the task type
- Specifies the goal (complete the pattern)
- References the familiar RPM framework
- Indicates matrix size for clarity

## Difficulty Levels

### Easy
- **Pattern Types**: Shape progression, color patterns
- **Characteristics**: Single rule, clear progression
- **Cognitive Load**: Low - pattern immediately apparent
- **Example**: Simple left-to-right shape sequence

### Medium
- **Pattern Types**: Number progression, rotation
- **Characteristics**: Requires counting or spatial transformation
- **Cognitive Load**: Moderate - needs systematic analysis
- **Example**: 90° rotation pattern across rows

### Hard
- **Pattern Types**: Combination patterns
- **Characteristics**: Multiple simultaneous rules
- **Cognitive Load**: High - requires identifying and coordinating multiple patterns
- **Example**: Shape changes while number increases and rotation occurs

## Generation Process

### Complete Workflow Implementation

#### 1. Dataset Creation Entry Point (`create_dataset()`)
```python
def create_dataset(num_samples: int = 50) -> Dict[str, Any]:
    generator = RPMPuzzleGenerator(tile_size=150)
    pairs = []
    
    for i in range(num_samples):
        task_id = f"raven_{i:04d}"
        seed = 2025 + i  # Deterministic seeding
        generator.rng.seed(seed)
```

#### 2. Pattern Generation Pipeline
```python
# In RPMPuzzleGenerator.generate_pattern_matrix()
rule_type = self.rng.choice([
    "shape_progression",
    "number_progression",
    "rotation",
    "color_pattern",
    "combination"
])

# Dispatch to specific generator method
if rule_type == "shape_progression":
    return self._generate_shape_progression()
# ... etc
```

#### 3. Matrix Rendering Process
```python
# Two-stage rendering for incomplete and complete matrices
def render_matrix(matrix, hide_last=True):
    for i in range(3):
        for j in range(3):
            if hide_last and i == 2 and j == 2:
                # Draw question mark for missing cell
                draw.text((tile_size//2-10, tile_size//2-10), "?", fill="gray")
            else:
                cell_img = render_cell(matrix[i][j])
```

#### 4. File Generation and Storage
```python
# Temporary file creation
import tempfile
temp_dir = tempfile.mkdtemp()

# Save images with standardized names
first_frame.save(f"{temp_dir}/{task_id}_first.png")
final_frame.save(f"{temp_dir}/{task_id}_final.png")

# Files moved to final location by create_dataset.py runner
```

#### 5. Metadata Assembly
```python
pair = {
    "id": task_id,
    "prompt": PROMPTS[0],  # From PROMPTS.py
    "first_image_path": first_frame_path,
    "final_image_path": final_frame_path,
    "domain": "raven",
    "task_category": "AbstractReasoning",
    "difficulty": difficulty_map.get(rule_type, "medium"),
    "raven_data": {
        "rule": rule,  # Human-readable description
        "rule_type": rule_type,
        "matrix_size": "3x3",
        "seed": seed
    },
    "created_at": datetime.now().isoformat()
}
```

### Quality Assurance

#### Pattern Validation
- **Deterministic Generation**: Fixed seed ensures reproducibility
- **Rule Consistency**: Each pattern follows mathematical formula
- **Unique Solutions**: Position-based algorithms guarantee single answer

#### Visual Quality Control
- **Consistent Sizing**: All cells exactly 150x150 pixels
- **Clear Boundaries**: 1px black borders separate cells
- **High Contrast**: Black shapes on white background
- **Missing Cell Indicator**: Gray "?" clearly marks unknown

#### Integration Validation
- **Path Consistency**: Temporary files ensure atomic operations
- **Metadata Completeness**: All required fields populated
- **Difficulty Distribution**: Automatic categorization based on rule type
- **Timestamp Accuracy**: ISO format with timezone

## Usage Examples

### 1. Standard Dataset Generation
```python
from vmevalkit.tasks.raven_task import create_dataset

# Generate default 50 tasks
dataset = create_dataset(num_samples=50)
print(f"✅ Created {dataset['total_pairs']} RPM tasks")
print(f"📊 Difficulty distribution: {dataset['generation_info']['difficulty_distribution']}")

# Access individual tasks
for pair in dataset['pairs'][:3]:
    print(f"ID: {pair['id']}")
    print(f"  Rule: {pair['raven_data']['rule']}")
    print(f"  Type: {pair['raven_data']['rule_type']}")
    print(f"  Difficulty: {pair['difficulty']}")
    print(f"  Seed: {pair['raven_data']['seed']}")
```

### 2. VMEvalKit Integration
```python
# Via runner/create_dataset.py
from vmevalkit.runner.create_dataset import generate_domain_to_folders

# Generate directly to folder structure
pairs = generate_domain_to_folders(
    domain_name="raven",
    num_samples=100,
    output_base=Path("data/questions"),
    random_seed=42
)

# Files created in: data/questions/raven_task/raven_XXXX/
# - first_frame.png
# - final_frame.png
# - prompt.txt
# - question_metadata.json
```

### 3. Custom Pattern Generation
```python
from vmevalkit.tasks.raven_task.rpm_generator import RPMPuzzleGenerator
import random

# Create custom generator
generator = RPMPuzzleGenerator(tile_size=200)  # Larger cells
generator.rng = random.Random(12345)  # Custom seed

# Generate specific pattern
matrix, rule = generator.generate_pattern_matrix()
print(f"Generated: {rule}")

# Render both frames
incomplete = generator.render_matrix(matrix, hide_last=True)
complete = generator.render_matrix(matrix, hide_last=False)

# Save custom puzzles
incomplete.save("custom_puzzle_incomplete.png")
complete.save("custom_puzzle_complete.png")
```

### 4. Pattern Type Analysis
```python
# Analyze pattern distribution in generated dataset
from collections import Counter

dataset = create_dataset(num_samples=100)
pattern_types = [p['raven_data']['rule_type'] for p in dataset['pairs']]
distribution = Counter(pattern_types)

print("Pattern Type Distribution:")
for pattern, count in distribution.items():
    percentage = (count / len(pattern_types)) * 100
    print(f"  {pattern:20} : {count:3d} ({percentage:.1f}%)")
```

### 5. Batch Processing with Progress
```python
from vmevalkit.tasks.raven_task import create_dataset
from pathlib import Path
import json

# Generate large dataset with progress tracking
num_samples = 500
dataset = create_dataset(num_samples=num_samples)

# Save dataset metadata
output_dir = Path("data/raven_tasks")
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "dataset_metadata.json", 'w') as f:
    json.dump({
        'total_pairs': dataset['total_pairs'],
        'generation_info': dataset['generation_info'],
        'created_at': dataset['created_at']
    }, f, indent=2)

print(f"Dataset saved with {dataset['total_pairs']} puzzles")
```

### 6. Visualization Helper (if implemented)
```python
from vmevalkit.tasks.raven_task import visualize_solution_process

# Note: This function requires task already exists in data/questions/
viz_path = visualize_solution_process(
    task_id="raven_0001",
    output_dir="output/raven_solutions"
)
# Creates combined image showing problem and solution side-by-side
```

## Implementation Details

### Module Architecture

#### Core Classes

##### `RPMPuzzleGenerator`
The main puzzle generation engine with the following key attributes:
```python
class RPMPuzzleGenerator:
    def __init__(self, tile_size: int = 192, seed: Optional[int] = None):
        self.tile_size = tile_size          # Size of each matrix cell
        self.rng = random.Random(seed)     # Seeded random generator
        self.shapes = ["circle", "square", "triangle", "diamond", "cross", "star"]
        self.colors = ["black", "gray", "darkgray"]
        self.fills = [None, "lightgray", "white"]
```

**Key Methods**:
- `generate_pattern_matrix()`: Main entry point, returns (matrix, rule_description)
- `_generate_[pattern]_pattern()`: Pattern-specific generation methods
- `render_cell(cell_data)`: Converts cell data to PIL Image
- `render_matrix(matrix, hide_last)`: Creates full matrix visualization

#### Module Functions

##### Shape Drawing (`create_shape()`)
```python
def create_shape(draw: ImageDraw.Draw, shape_type: str, 
                bbox: Tuple[int, int, int, int], 
                color: str = "black", fill: Optional[str] = None)
```
- Standalone function for geometric shape rendering
- Uses PIL's ImageDraw primitives
- Calculates shape dimensions based on bounding box

##### Dataset Creation (`create_dataset()`)
```python
def create_dataset(num_samples: int = 50) -> Dict[str, Any]
```
- VMEvalKit integration point
- Manages temporary file creation
- Assembles metadata for each task pair
- Returns standardized dataset dictionary

### Design Patterns

#### 1. **Strategy Pattern for Pattern Generation**
Each pattern type has its own generation method (`_generate_*_pattern()`), allowing easy addition of new patterns without modifying core logic.

#### 2. **Factory Pattern for Shape Creation**
The `create_shape()` function acts as a factory, dispatching to appropriate drawing code based on `shape_type` parameter.

#### 3. **Builder Pattern for Dataset Assembly**
The dataset is built incrementally with each component (images, metadata, prompts) assembled separately before final aggregation.

### File I/O Strategy

#### Temporary File Handling
```python
import tempfile
temp_dir = tempfile.mkdtemp()
# Generate files in temp directory
# Runner moves files to final location
```
- Ensures atomic operations
- Prevents partial writes
- Enables parallel generation

#### Path Management
- Relative paths used in metadata
- Temporary paths during generation
- Final paths follow VMEvalKit convention: `data/questions/raven_task/raven_XXXX/`

### Integration Points

#### 1. **VMEvalKit Runner Integration**
```python
# In runner/create_dataset.py
DOMAIN_REGISTRY = {
    'raven': {
        'module': 'vmevalkit.tasks.raven_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    }
}
```

#### 2. **Evaluation Integration**
- GPT-4O evaluation: Compares generated final frame with ground truth
- Human evaluation: Presents puzzle for manual assessment
- Task guidance: "Verify that the pattern completion in the final frame matches the expected pattern"

### Error Handling and Validation

#### Seed Management
```python
seed = 2025 + i  # Deterministic seed based on index
generator.rng.seed(seed)
```
- Ensures reproducibility across runs
- Allows debugging specific puzzles
- Enables result verification

#### Pattern Validation
- Mathematical formulas guarantee valid patterns
- Position-based algorithms ensure unique solutions
- No ambiguous patterns possible with current implementation

## Integration with VMEvalKit

### End-to-End Workflow

1. Dataset generation
   - Entry point: `vmevalkit.tasks.raven_task.create_dataset(num_samples)`
   - Used by runner: `vmevalkit.runner.create_dataset.generate_domain_to_folders("raven", ...)`
   - Flow:
     - Generate matrices and temp images (`*_first.png`, `*_final.png`)
     - Runner copies to `data/questions/raven_task/<id>/`
     - Writes `prompt.txt` and `question_metadata.json`
     - Updates image paths to be relative to `data/questions`

2. Inference (video generation)
   - API: `vmevalkit.runner.inference.run_inference(...)` or `InferenceRunner`
   - Output layout (per task and run):
     ```
     data/outputs/<experiment>/<model>/raven_task/<task_id>/<run_timestamp>/
     ├── question/
     │   ├── first_frame.png
     │   ├── final_frame.png
     │   └── prompt.txt
     └── video/
         └── <generated>.mp4
     ```

3. Automatic evaluation
   - Module: `vmevalkit.eval.gpt4o_eval.GPT4OEvaluator`
   - Task guidance for Raven: "Verify that the pattern completion in the final frame matches the expected pattern."
   - Compares model's video final frame with ground truth `question/final_frame.png`

4. Human evaluation (optional)
   - Module: `vmevalkit.eval.human_eval.HumanEvaluator`
   - Loads the same folder layout and presents side-by-side comparison

### Runner Integration Details

- Domain registry entry (runner):
  ```
  DOMAIN_REGISTRY["raven"] = {
      'module': 'vmevalkit.tasks.raven_task',
      'create_function': 'create_dataset',
      'process_dataset': lambda dataset, n: dataset['pairs']
  }
  ```
- Function used: `generate_domain_to_folders("raven", num_samples, output_base, random_seed)`
- Copies temp files from generator into standardized per-question folders

### File and Metadata Contract

- Required files per question:
  - `first_frame.png` (incomplete matrix)
  - `final_frame.png` (ground truth)
  - `prompt.txt` (standardized instruction)
  - `question_metadata.json` (includes `raven_data.rule_type` and `difficulty`)
- Paths in metadata are relative to `data/questions/`

### Reproducibility

- Seeds: `seed = 2025 + index` stored under `raven_data.seed`
- `RPMPuzzleGenerator.rng` reseeded per task to ensure deterministic regeneration

## Evaluation Criteria

### Pattern Recognition
- **Correct Rule Identification**: Does the model identify the underlying pattern?
- **Rule Application**: Can the model apply the rule to find the missing element?
- **Consistency**: Is the solution consistent with all matrix elements?

### Visual Reasoning
- **Spatial Understanding**: Accurate interpretation of 2D arrangements
- **Abstract Thinking**: Going beyond surface features to find rules
- **Systematic Analysis**: Checking rows, columns, and diagonals

### Solution Quality
- **Correctness**: The completed pattern follows the established rule
- **Uniqueness**: Solution is the only valid completion
- **Visual Clarity**: Generated video clearly shows the reasoning

## Expected Model Behavior

```
INPUT:  Incomplete 3x3 matrix + "This is Raven's Progressive Matrices like task. 
        Complete the missing pattern in this 3x3 matrix."

ANALYSIS: Model should:
          - Examine rows for horizontal patterns
          - Check columns for vertical patterns  
          - Consider diagonal relationships
          - Identify the governing rule

OUTPUT: Video showing:
        - Initial incomplete matrix (8 cells filled)
        - Reasoning process (optional intermediate frames)
        - Final complete matrix (all 9 cells filled correctly)

VALIDATION: ✅ Pattern rule maintained  ✅ Logical consistency  ✅ Visual clarity
```

## File Organization

```
vmevalkit/tasks/raven_task/
├── __init__.py                  # Module exports
├── raven_reasoning.py          # Main task generation logic
├── rpm_generator.py            # RPM puzzle generator
└── RAVEN.md                    # This documentation

data/
├── raven_tasks/
│   └── raven_tasks.json       # Dataset metadata
└── generated_raven/
    ├── raven_0000_first.png   # Incomplete matrices
    ├── raven_0000_final.png   # Complete solutions
    └── ...                     # Additional pairs
```

## Research Applications

### Cognitive Assessment
- **Abstract Reasoning**: Fundamental pattern recognition abilities
- **Logical Inference**: Rule discovery and application
- **Visual Intelligence**: Non-verbal reasoning capabilities

### Model Capabilities
- **Systematic Thinking**: Methodical pattern analysis
- **Generalization**: Applying learned rules to new instances
- **Multi-step Reasoning**: Building from pattern to solution

### Benchmarking Value
- **Standardized Format**: Consistent with classical RPM tests
- **Difficulty Scaling**: Progressive complexity levels
- **Clear Metrics**: Objective success/failure criteria

## Technical Specifications

### Dependencies

#### Required Packages
```python
# Core dependencies (from imports)
from PIL import Image, ImageDraw, ImageFont  # Pillow>=8.3.0
import numpy as np                           # numpy>=1.21.0
import random                                # Python standard library
import json                                  # Python standard library
import os                                    # Python standard library
import tempfile                              # Python standard library
from datetime import datetime                # Python standard library
from typing import Dict, Any, List, Tuple, Optional  # Python 3.7+
```

#### Version Requirements
```toml
# Minimum versions for core functionality
python = ">=3.7"       # Type hints support
pillow = ">=8.3.0"     # PIL.Image, PIL.ImageDraw
numpy = ">=1.21.0"     # np.pi, np.cos, np.sin for star drawing
```

### Performance Characteristics

#### Generation Metrics
- **Single Puzzle Generation**: ~50-100ms
- **Dataset Creation (50 tasks)**: ~3-5 seconds
- **Memory Footprint**:
  - Per puzzle: ~2MB (two 450x450 RGB images)
  - Generator instance: ~1MB
  - Peak during batch generation: ~100MB for 50 tasks

#### Image Specifications
- **Matrix Dimensions**: 450x450 pixels (3×150px cells)
- **Color Depth**: RGB (3 channels, 8-bit)
- **File Size**: 
  - PNG format: ~30-50KB per image
  - Total per task: ~60-100KB (first + final frame)
- **Compression**: PNG lossless compression

#### Computational Complexity
```python
# Pattern generation complexity
O(1) for each cell (9 cells total)
O(n) for dataset of n puzzles

# Rendering complexity
O(s²) where s = tile_size (pixel operations)
O(9s²) for complete matrix
```

### System Requirements

#### Minimum Requirements
- **CPU**: Any x86_64 or ARM64 processor
- **RAM**: 512MB available
- **Disk Space**: 10MB per 100 puzzles
- **OS**: Linux, macOS, Windows (Python 3.7+)

#### Recommended for Large-Scale Generation
- **CPU**: Multi-core for parallel generation
- **RAM**: 4GB+ for datasets > 1000 puzzles
- **Disk**: SSD for faster I/O operations
- **GPU**: Not required (CPU-only operations)

## Related Work

Inspired by:
- **Raven's Progressive Matrices (1938)**: Original non-verbal intelligence test
- **Abstract Reasoning Corpus (ARC)**: Modern AI reasoning benchmark
- **PGM Dataset**: Procedurally generated matrices for machine learning

## Future Extensions

### Enhanced Patterns
- **Analogy Patterns**: A:B::C:? relationships
- **Distributed Rules**: Patterns spanning multiple cells
- **Hidden Patterns**: Rules requiring deeper analysis

### Increased Complexity
- **Larger Matrices**: 4x4 or 5x5 grids
- **Multiple Missing Cells**: More than one unknown
- **Conditional Rules**: If-then pattern logic

### Interactive Features
- **Hint System**: Progressive clues for difficult patterns
- **Multiple Choice**: Selecting from candidate solutions
- **Explanation Generation**: Models describe discovered rules

## Conclusion

The Raven's Progressive Matrices task provides a rigorous evaluation of abstract reasoning capabilities in video generation models. By requiring pattern recognition, rule application, and visual completion, this task tests fundamental aspects of visual intelligence that are crucial for advanced AI systems.

The standardized format, clear evaluation criteria, and scalable difficulty make this an essential component of the VMEvalKit reasoning evaluation suite, complementing spatial (maze, rotation) and strategic (chess, sudoku) reasoning tasks.
