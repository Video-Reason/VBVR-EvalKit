# 3D Mental Rotation Task Documentation

## Overview

The 3D Mental Rotation Task evaluates video generation models' ability to demonstrate spatial reasoning and 3D visualization by generating videos that show how 3D voxel structures appear when the camera rotates horizontally around them. **Current implementation uses 8-9 voxel structures with tilted camera views (20-40° elevation) and horizontal-only rotations with exactly 180° azimuth changes for clear opposite viewpoint perspectives.** This task is part of VMEvalKit's reasoning evaluation suite and tests fundamental spatial cognition capabilities.

## Table of Contents
- [Task Description](#task-description)
- [Data Structure](#data-structure)
- [Prompts](#prompts)
- [Generation Algorithm](#generation-algorithm)
- [Implementation Details](#implementation-details)
- [Usage Examples](#usage-examples)
- [Evaluation Metrics](#evaluation-metrics)
- [Technical Specifications](#technical-specifications)
- [Research Applications](#research-applications)
- [Future Extensions](#future-extensions)

## Task Description

### Core Challenge
Models must:
1. **Parse 3D Structure**: Understand the 3D configuration from a tilted 2D projection
2. **Horizontal Rotation**: Generate smooth camera rotation around the fixed object
3. **Perspective Consistency**: Maintain consistent tilted viewing angle throughout
4. **Generate Transition**: Create smooth video showing 180° horizontal camera movement

### Visual Elements
- **3D Voxel Structures**: Snake-like configurations made of 8-9 connected cubes
- **Tilted Views**: Consistent 20-40° elevation for clear 3D perspective
- **Horizontal Rotations**: Camera moves horizontally around the fixed sculpture (180° change)
- **Consistent Rendering**: High-quality 3D visualization with proper lighting

## Data Structure

### RotationTaskPair
Each task consists of a pair of images and a text prompt:

```python
{
    "id": str,                    # Unique identifier (e.g., "rotation_0001")
    "prompt": str,                # Instructions for the video model
    "first_image_path": str,      # Path to initial viewpoint image
    "final_image_path": str,      # Path to rotated viewpoint image
    "task_category": str,         # "3D Mental Rotation"
    "rotation_data": {            # Task-specific metadata
        "generation_method": str,           # "3D voxel snake with viewpoint rotation"
        "num_voxels": int,                 # Number of cubes in structure (8-9)
        "first_view_elev": float,          # Initial elevation angle (20-40°)
        "first_view_azim": float,          # Initial azimuth angle (0-359°)
        "final_view_elev": float,          # Final elevation angle (same as initial)
        "final_view_azim": float,          # Final azimuth angle (initial + 180°)
        "angle_difference": float,          # Always 180° for current implementation
        "structural_complexity": str        # "snake_like_3d_voxels"
    },
    "difficulty": str,            # "easy", "medium", or "hard" (based on structure)
    "created_at": str            # ISO timestamp
}
```

### RotationDataset
A collection of rotation task pairs with metadata:

```python
{
    "name": "rotation_tasks",
    "description": "3D mental rotation tasks with 8-9 voxels, tilted views (20-40° elevation) and 180° horizontal rotations for video model evaluation (N pairs)",
    "pairs": [RotationTaskPair, ...]
}
```

## Prompts

### Current Prompt Template
The prompts emphasize horizontal camera rotation with maintained tilt and explicit 180° rotation:

#### Template Structure
```python
"A {num_voxels}-block sculpture sits fixed on a table. "
"First frame: Your camera is tilted at {elev1}° elevation, viewing from {azim1}° azimuth. "
"Final frame: Your camera remains at {elev2}° elevation, but rotates horizontally to {azim2}° azimuth. This is a 180-degree rotation "
"Create a smooth video showing the camera's horizontal rotation around the sculpture, and try to maintain the tilted viewing angle throughout."
```

#### Example Prompts

##### Example 1 (8 blocks)
"A 8-block sculpture sits fixed on a table. First frame: Your camera is tilted at 30° elevation, viewing from 45° azimuth. Final frame: Your camera remains at 30° elevation, but rotates horizontally to 225° azimuth. This is a 180-degree rotation Create a smooth video showing the camera's horizontal rotation around the sculpture, and try to maintain the tilted viewing angle throughout."

##### Example 2 (9 blocks)
"A 9-block sculpture sits fixed on a table. First frame: Your camera is tilted at 25° elevation, viewing from 120° azimuth. Final frame: Your camera remains at 25° elevation, but rotates horizontally to 300° azimuth. This is a 180-degree rotation Create a smooth video showing the camera's horizontal rotation around the sculpture, and try to maintain the tilted viewing angle throughout."

### Key Prompt Features
1. **Tilted Perspective**: Camera maintains consistent 20-40° elevation for 3D depth
2. **Horizontal Movement**: Camera rotates horizontally around the fixed sculpture
3. **Fixed Object**: The sculpture stays fixed - only the camera moves around it
4. **Explicit Rotation Amount**: Clearly states "180-degree rotation"
5. **No Ambiguity**: Clear that only azimuth changes while elevation stays constant

## Generation Algorithm

### 1. 3D Voxel Structure Creation
The system generates snake-like 3D structures using a sophisticated recursive algorithm:

```python
def _generate_snake(
    N: int = 8-9,              # Number of voxels (limited to 8-9 for easier difficulty)
    Lmin: int = 1,              # Minimum segment length
    Lmax: int = 3,              # Maximum segment length
    p_branch: float = 0.2,      # Branching probability (20%)
    max_deg: int = 3,           # Maximum neighbors per voxel
    tries: int = 1000           # Maximum generation attempts
)
```

#### Algorithm Steps:
1. **Initialization**: Start with a single voxel at origin (0,0,0)
2. **Growth Direction**: Choose initial random direction from 6 cardinal directions
3. **Segment Creation**: 
   - Grow straight segments of length Lmin to Lmax
   - Ensure no self-intersection
   - Respect max_deg neighbor constraint
4. **Optional Branching**: 
   - With probability p_branch, add a branch perpendicular to current direction
   - Branch from the start of the current segment
5. **Direction Change**: 
   - After each segment, choose new orthogonal direction
   - Prefer unused axes to ensure 3D spread
6. **Validation**:
   - Ensure all three axes (x, y, z) are utilized
   - Check structure is not rotationally symmetric
   - Verify exact voxel count matches target

**Key Features**:
- **Connected Structure**: All voxels form a connected graph (no floating pieces)
- **3D Spread**: Algorithm enforces usage of all three spatial dimensions
- **Controlled Complexity**: Branching probability and segment length constraints
- **Non-symmetric**: Rejects rotationally symmetric configurations to ensure distinct views

### 2. Viewpoint Selection
Generates tilted views with 180° horizontal rotation for maximum perspective difference:

```python
def generate_horizontal_rotation():
    # Fixed tilted elevation for both views (ensures horizontal rotation)
    tilted_elevation = random.randint(20, 40)  # degrees
    
    # Random initial azimuth
    azim1 = random.randint(0, 359)  # degrees
    
    # Final azimuth is exactly 180° away (opposite side)
    azim2 = (azim1 + 180) % 360
    
    return (tilted_elevation, azim1), (tilted_elevation, azim2)
```

**Constraints**:
- **Tilted Views**: 20-40° elevation provides optimal 3D perspective
- **Horizontal Only**: Elevation remains constant between views
- **Opposite Views**: Exactly 180° azimuth change shows front vs. back perspectives

### 3. Difficulty Assessment
Automated difficulty classification based on structural complexity:

```python
def _assess_difficulty(voxels: List[Voxel], angle_diff: float) -> str:
    # Base complexity (all tasks start with low complexity due to 8-9 voxels)
    complexity_score = 2
    
    # Analyze spatial distribution
    axes_used = (len(set(v[0] for v in voxels)) + 
                 len(set(v[1] for v in voxels)) + 
                 len(set(v[2] for v in voxels)))
    
    if axes_used > 6:  # Wide spatial spread
        complexity_score += 2
    elif axes_used > 4:
        complexity_score += 1
    
    # Classification thresholds (adjusted for 8-9 voxel range)
    if complexity_score <= 4:
        return "easy"
    elif complexity_score <= 6:
        return "medium"  
    else:
        return "hard"
```

**Note**: Due to the limited voxel count (8-9), most tasks fall into the "easy" category. The 180° rotation is considered standard difficulty and doesn't add complexity.

### 4. Rotational Equivalence Detection
Ensures generated structures are unique and not rotational duplicates:

```python
def _are_rotationally_equivalent(A: List[Voxel], B: List[Voxel]) -> bool:
    """Check if structure A can be rotated to match structure B"""
    # Generate all 24 orientation-preserving 3D rotations
    # Compare canonicalized versions after each rotation
    # Return True if any rotation matches
```

This uses group theory to generate all 24 rotation matrices that preserve orientation in 3D space (the rotation group SO(3) restricted to 90° rotations).

## Implementation Details

### Rendering System
High-quality 3D visualization using matplotlib:

```python
def _render_voxel_image(voxels: List[Voxel], elev: float, azim: float, output_path: str):
    """
    Create publication-quality 3D renderings with:
    - Perspective projection (not orthographic)
    - Consistent lighting and shading
    - Anti-aliased edges (0.8pt black borders)
    - White background for clarity
    - Light blue cube faces (RGB: 0.7, 0.7, 0.9)
    """
    # Figure size: 8x8 inches at 150 DPI
    # Face alpha: 0.8 for slight transparency
    # Edge width: 0.8 for clear definition
```

### Cube Rendering Details
Each voxel is rendered as a unit cube with:
- **Vertices**: 8 corners defining the cube
- **Faces**: 6 faces (bottom, top, front, back, left, right)
- **Color**: Light blue (0.7, 0.7, 0.9) with 0.8 alpha
- **Edges**: Black borders with 0.8pt width
- **Shading**: Matplotlib's automatic 3D shading

### Image Processing Pipeline
Standardized image format for VMEvalKit compatibility:

1. **High-Resolution Rendering**: 
   - Initial render at 1200x1200 pixels (150 DPI × 8 inches)
   - Matplotlib figure size: 8×8 inches
   
2. **Cropping**: 
   - Center crop to square aspect ratio
   - Remove any whitespace margins
   
3. **Resizing**: 
   - Downsample to 400×400 pixels (VMEvalKit standard)
   - Uses Lanczos resampling for quality
   
4. **Color Space**: 
   - Convert to RGB (remove alpha channel)
   - Ensure consistent color representation
   
5. **Format**: 
   - Save as PNG for lossless compression
   - File size: ~100KB per image

### Axis Scaling Algorithm
Ensures proper 3D visualization:

```python
def _set_axes_equal(ax, points: np.ndarray, padding: float = 0.2):
    """Force equal scaling on all axes with padding"""
    # Find maximum range across all dimensions
    # Center the view on the structure's centroid
    # Add 20% padding for visual clarity
```

### Quality Assurance
Multiple validation steps ensure task quality:

- **Structural Validation**: 
  - Connected graph verification
  - Exact voxel count (8 or 9)
  - All three axes utilized
  
- **Uniqueness Check**: 
  - No rotationally equivalent structures
  - Flip test to ensure asymmetry
  
- **Rendering Validation**:
  - Image dimensions correct (400×400)
  - RGB color space
  - File accessibility
  
- **Prompt Consistency**:
  - Elevation values match (horizontal rotation)
  - Azimuth difference is exactly 180°

## Usage Examples

### Basic Dataset Generation
```python
from vmemalkit.tasks.rotation_task import create_dataset

# Generate 50 mental rotation tasks (8-9 voxels, 180° horizontal rotations)
dataset = create_dataset(num_samples=50)
print(f"Created {len(dataset['pairs'])} rotation tasks")

# Dataset structure
print(f"Name: {dataset['name']}")
print(f"Description: {dataset['description']}")
print(f"Number of pairs: {len(dataset['pairs'])}")
```

### Using the RotationGenerator Class
```python
from vmemalkit.tasks.rotation_task import RotationGenerator
from collections import Counter

# Initialize generator with fixed random seed
generator = RotationGenerator()

# Generate task data (without images)
tasks = generator.generate_tasks(num_tasks=100)

# Analyze generated tasks
difficulties = Counter(task['difficulty'] for task in tasks)
voxel_counts = Counter(task['num_voxels'] for task in tasks)

print(f"Difficulty distribution: {dict(difficulties)}")
print(f"Voxel count distribution: {dict(voxel_counts)}")
print(f"All rotations are 180°: {all(task['angle_difference'] == 180 for task in tasks)}")
```

### Individual Task Creation
```python
from vmemalkit.tasks.rotation_task import create_task_pair, generate_prompt

# Define task data manually
task_data = {
    "voxels": [(0,0,0), (1,0,0), (1,1,0), (1,1,1), 
               (2,1,0), (2,1,1), (2,2,1), (3,2,1)],  # 8 voxels
    "first_view": (30, 45),   # 30° elevation, 45° azimuth
    "final_view": (30, 225),  # Same elevation, 180° rotation
    "angle_difference": 180.0,
    "difficulty": "easy",
    "num_voxels": 8
}

# Create task pair with images and prompt
pair = create_task_pair(task_data, "rotation_custom")

# Generate prompt separately if needed
prompt = generate_prompt(task_data)
print(f"Generated prompt: {prompt}")
```

### Accessing Task Metadata
```python
# Load existing dataset
import json
with open('data/rotation_tasks/rotation_tasks.json', 'r') as f:
    dataset = json.load(f)

# Analyze first task
task = dataset['pairs'][0]
print(f"Task ID: {task['id']}")
print(f"Number of voxels: {task['rotation_data']['num_voxels']}")
print(f"First view: elev={task['rotation_data']['first_view_elev']}°, "
      f"azim={task['rotation_data']['first_view_azim']}°")
print(f"Final view: elev={task['rotation_data']['final_view_elev']}°, "
      f"azim={task['rotation_data']['final_view_azim']}°")
print(f"Rotation amount: {task['rotation_data']['angle_difference']}°")
print(f"Difficulty: {task['difficulty']}")
```

### Batch Processing with Multiprocessing
```python
from multiprocessing import Pool
from functools import partial

def process_single_task(task_id, task_data):
    """Process a single task (for parallel execution)"""
    return create_task_pair(task_data, f"rotation_{task_id:04d}")

# Generate task data
generator = RotationGenerator()
tasks = generator.generate_tasks(100)

# Process in parallel
with Pool() as pool:
    process_func = partial(process_single_task)
    pairs = pool.starmap(process_func, enumerate(tasks))

print(f"Processed {len(pairs)} tasks in parallel")
```

## Evaluation Metrics

### Spatial Accuracy Metrics
- **Geometric Correctness**: 
  - Voxel positions maintained throughout rotation
  - Structure integrity preserved
  - No floating or disconnected pieces
  
- **Rotation Fidelity**: 
  - Accurate 180° azimuth transformation
  - Correct perspective changes (front→back view)
  - Proper occlusion handling
  
- **Perspective Consistency**: 
  - Elevation angle maintained (±2° tolerance)
  - No vertical drift during rotation
  - Consistent camera distance

### Motion Quality Metrics
- **Smoothness**: 
  - Continuous rotation without jumps
  - Frame interpolation quality
  - Rotation speed consistency
  
- **Temporal Consistency**: 
  - Structure stability across frames
  - No flickering or artifacts
  - Coherent lighting throughout
  
- **Horizontal Movement**: 
  - Pure azimuthal rotation
  - No elevation changes
  - Circular camera path

### Reasoning Demonstration Metrics
- **Problem Understanding**: 
  - Correct interpretation of tilted initial view
  - Recognition of 3D structure from 2D projection
  
- **Solution Planning**: 
  - Logical rotation sequence
  - Appropriate intermediate viewpoints
  
- **Goal Achievement**: 
  - Accurate final viewpoint (180° rotation)
  - Maintained tilt angle
  - Complete rotation arc

## Technical Specifications

### Dependencies
```python
# Core requirements (must be installed)
numpy>=1.21.0          # Array operations and numerical computations
matplotlib>=3.5.0      # 3D rendering and visualization
pillow>=8.3.0         # Image processing and format conversion

# Optional for development
tqdm>=4.62.0          # Progress bars for batch processing
```

### Performance Characteristics
- **Generation Speed**: 
  - Single task: ~2-5 seconds (CPU)
  - Includes structure generation, rendering, and image processing
  
- **Memory Usage**: 
  - Peak: ~50MB during rendering
  - Steady state: ~10MB per task
  
- **Image Quality**: 
  - Resolution: 400×400 RGB
  - File size: ~100KB per image
  - Format: PNG (lossless)
  
- **Success Rate**: 
  - Structure generation: >95% within 1000 attempts
  - Rendering: 100% (deterministic)

### File Organization
```
vmemalkit/tasks/rotation_task/
├── __init__.py                 # Module exports and version
├── rotation_reasoning.py       # Main generation logic
├── PROMPTS.py                 # Centralized prompt templates
└── ROTATION.md                # This documentation

data/
├── rotation_tasks/
│   └── rotation_tasks.json    # Dataset metadata file
└── generated_rotation/
    ├── rotation_0000/
    │   ├── first_frame.png    # Initial viewpoint
    │   ├── final_frame.png    # Final viewpoint (180° rotated)
    │   ├── prompt.txt         # Task prompt
    │   └── question_metadata.json  # Task-specific metadata
    └── ...                    # Additional task folders
```

### Module Architecture

#### Core Components
1. **RotationGenerator**: Main class for task generation
   - Snake structure generation algorithm
   - Viewpoint selection logic
   - Difficulty assessment

2. **Rendering Functions**:
   - `_render_voxel_image()`: Main rendering pipeline
   - `_plot_cubes()`: 3D cube visualization
   - `_process_and_save_image()`: Image post-processing

3. **Utility Functions**:
   - `_are_rotationally_equivalent()`: Uniqueness checking
   - `_shift_to_origin()`: Coordinate normalization
   - `_set_axes_equal()`: Visualization scaling

4. **API Functions**:
   - `create_dataset()`: High-level dataset creation
   - `create_task_pair()`: Individual task generation
   - `generate_prompt()`: Prompt formatting

## Research Applications

### Spatial Cognition Assessment
- **Mental Rotation Ability**: 
  - 180° horizontal rotation provides maximum perspective change
  - Tests ability to mentally transform 3D objects
  - Evaluates understanding of opposite viewpoints
  
- **3D Visualization**: 
  - Understanding complex 3D structures from 2D projections
  - Depth perception from monocular cues
  - Shape completion from partial views
  
- **Perspective Taking**: 
  - Camera-centric vs. object-centric reference frames
  - Maintaining consistent viewing angle during movement

### Model Capabilities Testing
- **Geometric Understanding**: 
  - 3D spatial relationships between voxels
  - Connectivity and adjacency reasoning
  - Volumetric shape representation
  
- **Temporal Reasoning**: 
  - Planning smooth rotation trajectories
  - Interpolating intermediate viewpoints
  - Maintaining temporal coherence
  
- **Visual Consistency**: 
  - Object permanence during rotation
  - Consistent lighting and shading
  - Proper occlusion ordering

### Benchmark Applications
- **Cross-Model Comparison**: 
  - Standardized 400×400 format
  - Consistent difficulty levels
  - Reproducible generation (seeded random)
  
- **Difficulty Scaling**: 
  - Structural complexity variations
  - Future extensibility to more voxels
  - Rotation angle adjustments
  
- **Performance Analytics**: 
  - Success rate tracking
  - Error categorization
  - Comparative analysis tools

## Future Extensions

### Enhanced Structures
- **Variable Voxel Count**: 
  - Extend beyond 8-9 voxels for harder tasks
  - Support 4-20 voxel range
  - Adaptive difficulty based on performance
  
- **Complex Shapes**: 
  - Multi-branch structures
  - Hollow configurations
  - Interlocking components
  
- **Textured Voxels**: 
  - Different colors per voxel
  - Surface patterns for disambiguation
  - Material properties (metallic, glass, etc.)

### Advanced Rotations
- **Variable Angles**: 
  - Support 90°, 270° rotations
  - Arbitrary angle specifications
  - Multi-axis rotations (not just horizontal)
  
- **Complex Trajectories**: 
  - Spiral camera paths
  - Zoom during rotation
  - Focal length changes
  
- **Partial Rotations**: 
  - Stop at intermediate angles
  - Reverse rotations
  - Oscillating movements

### Interactive Elements
- **User-Guided Generation**: 
  - Specify exact voxel configurations
  - Choose rotation parameters
  - Difficulty preferences
  
- **Real-Time Feedback**: 
  - Progressive evaluation
  - Intermediate frame checking
  - Error visualization
  
- **Adaptive Testing**: 
  - Performance-based difficulty adjustment
  - Personalized challenge levels
  - Learning curve tracking

### Integration Enhancements
- **Multi-Modal Tasks**: 
  - Add text descriptions
  - Audio narration of rotation
  - Haptic feedback simulation
  
- **Composite Scenes**: 
  - Multiple objects
  - Background environments
  - Relative motion tasks

---

## Conclusion

The 3D Mental Rotation Task provides a robust evaluation framework for spatial reasoning capabilities in video generation models. By combining sophisticated 3D voxel structure generation with tilted camera views and 180° horizontal rotations, this task creates clear, unambiguous challenges that test fundamental spatial cognition abilities while providing maximum perspective change.

The task's implementation strengths include:
- **Reproducible Generation**: Seeded random generation ensures consistency
- **Objective Evaluation**: Clear success/failure criteria with 180° rotation
- **Quality Rendering**: High-quality 3D visualization with proper perspective
- **Standard Format**: Full VMEvalKit compatibility

Current limitations that inform future development:
- Limited to 8-9 voxels for manageable difficulty
- Fixed 180° rotation angle
- Single-object scenes only
- Uniform voxel appearance

This makes it an excellent foundational tool for benchmarking and advancing video model spatial reasoning capabilities, with clear paths for future enhancements.

**Happy spatial reasoning evaluation!** 🎯🔄

---

*Documentation Version: 2.0.0*  
*Last Updated: November 2024*  
*Module Version: 1.0.0*