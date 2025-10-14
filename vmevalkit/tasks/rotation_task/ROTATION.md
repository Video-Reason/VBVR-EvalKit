# 3D Mental Rotation Task Documentation

## Overview

The 3D Mental Rotation Task evaluates video generation models' ability to demonstrate spatial reasoning and 3D visualization by generating videos that show how 3D voxel structures appear when rotated from one viewpoint to another. **Current implementation uses 8-15 voxel structures with 30-90° or axis-aligned 90° rotations for challenging spatial reasoning tasks.** This task is part of VMEvalKit's reasoning evaluation suite and tests fundamental spatial cognition capabilities.

## Task Description

### Core Challenge
Models must:
1. **Parse 3D Structure**: Understand the 3D configuration from a 2D projection
2. **Mental Rotation**: Mentally rotate the structure in 3D space
3. **Viewpoint Transform**: Predict appearance from a different viewing angle
4. **Generate Transition**: Create smooth video showing the rotation process

### Visual Elements
- **3D Voxel Structures**: Snake-like configurations made of connected cubes
- **Viewing Angles**: Different elevation and azimuth combinations
- **Rotation Transitions**: Smooth transformations between viewpoints
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
        "num_voxels": int,                 # Number of cubes in structure
        "first_view_elev": float,          # Initial elevation angle (degrees)
        "first_view_azim": float,          # Initial azimuth angle (degrees)
        "final_view_elev": float,          # Final elevation angle (degrees)
        "final_view_azim": float,          # Final azimuth angle (degrees)
        "angle_difference": float,          # Total angular difference (degrees)
        "structural_complexity": str        # "snake_like_3d_voxels"
    },
    "difficulty": str,            # "easy", "medium", or "hard"
    "created_at": str            # ISO timestamp
}
```

### RotationDataset
A collection of rotation task pairs with metadata:

```python
{
    "name": "rotation_tasks",
    "description": "3D mental rotation reasoning tasks for video model evaluation (N pairs)",
    "pairs": [RotationTaskPair, ...]
}
```

## Prompts

### Simple and Clear Prompt Structure
The prompts specify exact camera viewing angles to eliminate ambiguity:

#### Template
- "A {num_voxels}-block sculpture sits fixed on a table."
- "First frame: Your camera is at viewing angle (elevation: {elev1}°, azimuth: {azim1}°)."
- "Final frame: Your camera is at viewing angle (elevation: {elev2}°, azimuth: {azim2}°)."
- "Create a smooth video transition as the camera moves between these viewing angles, showing how the sculpture appears from different perspectives."

#### Example Prompts

##### Example 1
"A 4-block sculpture sits fixed on a table. First frame: Your camera is at viewing angle (elevation: 45°, azimuth: 0°). Final frame: Your camera is at viewing angle (elevation: 45°, azimuth: 90°). Create a smooth video transition as the camera moves between these viewing angles, showing how the sculpture appears from different perspectives."

##### Example 2
"A 5-block sculpture sits fixed on a table. First frame: Your camera is at viewing angle (elevation: 30°, azimuth: 45°). Final frame: Your camera is at viewing angle (elevation: 60°, azimuth: 45°). Create a smooth video transition as the camera moves between these viewing angles, showing how the sculpture appears from different perspectives."

### Key Features
1. **Clear Camera Angles**: Explicitly states these are camera viewing angles, not object positions
2. **Fixed Object**: The sculpture stays fixed - only the camera moves around it
3. **Exact Coordinates**: Precise elevation and azimuth angles for camera positions
4. **Simple Task**: "Create a smooth video transition" between two camera angles
5. **No Ambiguity**: Clear that the camera moves, not the sculpture

## Generation Algorithm

### 1. 3D Voxel Structure Creation
The system generates snake-like 3D structures using a sophisticated algorithm:

```python
def generate_snake(N, Lmin, Lmax, p_branch, max_deg, tries):
    """
    Create a 3D voxel snake with:
    - N: Number of voxels (8-15 range for challenging spatial reasoning)
    - Lmin, Lmax: Segment length bounds (2-4)
    - p_branch: Branching probability (0.35)
    - max_deg: Maximum neighbors per voxel (4)
    - tries: Maximum generation attempts (1000)
    """
```

**Key Features**:
- **Connected Structure**: All voxels form a connected graph
- **3D Spread**: Structures span all three spatial dimensions
- **Controlled Complexity**: Branching and segment length constraints
- **Unique Shapes**: Avoids rotationally symmetric configurations

### 2. Viewpoint Selection
Strategic camera positioning ensures meaningful rotation tasks:

```python
def sample_view(elev_range, azim_sectors):
    """
    Generate viewing angles with:
    - elev_range: Elevation sectors (15-75°, 105-165°, etc.)
    - azim_sectors: Azimuth sectors (15-75°, 105-165°, etc.)
    """
```

**Constraints**:
- **Minimum Angular Separation**: 30° minimum between views
- **Diverse Perspectives**: Covers different spatial orientations
- **Meaningful Rotations**: Avoids trivial or ambiguous transitions

### 3. Difficulty Assessment
Automated difficulty classification based on multiple factors:

```python
def assess_difficulty(voxels, angle_diff):
    """
    Factors:
    - Structure complexity (number and spread of voxels)
    - Rotation magnitude (angle difference)
    - Spatial distribution across axes
    """
```

**Difficulty Levels**:
- **Easy**: Smaller structures (8-9 voxels), moderate rotations (≤4 complexity score)
- **Medium**: Medium structures (10-12 voxels) or large rotations (5-7 complexity score)  
- **Hard**: Large structures (13-15 voxels) with challenging rotations (≥8 complexity score)

## Implementation Details

### Rendering System
High-quality 3D visualization using matplotlib:

```python
def render_voxel_image(voxels, elev, azim, output_path):
    """
    Create publication-quality 3D renderings with:
    - Proper perspective projection
    - Consistent lighting and shading
    - Anti-aliased edges
    - White background for clarity
    """
```

### Image Processing Pipeline
Standardized image format for VMEvalKit:

1. **High-Resolution Rendering**: 1200x1200 initial render
2. **Cropping**: Square aspect ratio maintenance
3. **Resizing**: Standard 400x400 VMEvalKit format
4. **Color Space**: RGB for compatibility
5. **Format**: PNG for lossless quality

### Quality Assurance
Multiple validation steps ensure task quality:

- **Structural Validation**: Connected, non-degenerate shapes
- **Rotation Significance**: Minimum 30° angular separation
- **Uniqueness**: No rotationally equivalent structures
- **Complexity Distribution**: Balanced difficulty levels

## Usage Examples

### Basic Dataset Generation
```python
from vmevalkit.tasks.rotation_task import create_dataset

# Generate 50 mental rotation tasks (8-15 voxels, 30-90° or 90° rotations)
dataset = create_dataset(num_samples=50)
print(f"Created {len(dataset['pairs'])} rotation tasks")
```

### Custom Generation Parameters
```python
from vmevalkit.tasks.rotation_task import RotationGenerator

generator = RotationGenerator()
tasks = generator.generate_tasks(num_tasks=100)

# Analyze difficulty distribution
difficulties = [task['difficulty'] for task in tasks]
print(f"Distribution: {Counter(difficulties)}")
```

### Individual Task Creation
```python
from vmevalkit.tasks.rotation_task import create_task_pair

task_data = {
    "voxels": [(0,0,0), (1,0,0), (1,1,0), (1,1,1), (2,1,0), (2,1,1), (2,2,1), (3,2,1), (3,2,2), (3,3,2)],
    "first_view": (30, 45),
    "final_view": (60, 135),
    "angle_difference": 75.5,
    "difficulty": "medium",
    "num_voxels": 10
}

pair = create_task_pair(task_data, "rotation_test")
```

## Evaluation Metrics

### Spatial Accuracy
- **Geometric Correctness**: Proper 3D structure representation
- **Rotation Fidelity**: Accurate angular transformations
- **Perspective Consistency**: Correct viewpoint changes

### Motion Quality  
- **Smoothness**: Fluid rotation transitions
- **Temporal Consistency**: Coherent frame sequences
- **Axis Alignment**: Proper rotation axes

### Reasoning Demonstration
- **Problem Understanding**: Correct initial state interpretation  
- **Solution Planning**: Logical rotation sequence
- **Goal Achievement**: Accurate final viewpoint matching

## Technical Specifications

### Dependencies
```python
# Core requirements
numpy>=1.21.0          # Numerical computations
matplotlib>=3.5.0      # 3D rendering
pillow>=8.3.0         # Image processing

# Optional for development
tqdm>=4.62.0          # Progress bars
```

### Performance Characteristics
- **Generation Speed**: ~2-5 seconds per task (CPU)
- **Memory Usage**: ~50MB peak during rendering
- **Image Quality**: 400x400 RGB, ~100KB per image
- **Success Rate**: >95% valid structure generation

### File Organization
```
vmevalkit/tasks/rotation_task/
├── __init__.py                 # Module exports
├── rotation_reasoning.py       # Main generation logic
└── ROTATION.md                # This documentation

data/
├── rotation_tasks/
│   └── rotation_tasks.json    # Dataset file
└── generated_rotation/
    ├── rotation_0000_first.png # Initial viewpoints
    ├── rotation_0000_final.png # Final viewpoints
    └── ...                     # Additional pairs
```

## Research Applications

### Spatial Cognition Assessment
- **Mental Rotation Ability**: Core spatial reasoning skill
- **3D Visualization**: Understanding 3D structure from 2D views
- **Perspective Taking**: Viewpoint transformation capabilities

### Model Capabilities Testing
- **Geometric Understanding**: 3D spatial relationships
- **Temporal Reasoning**: Smooth transition generation
- **Visual Consistency**: Maintaining object identity across views

### Benchmark Applications
- **Cross-Model Comparison**: Standardized evaluation format
- **Difficulty Scaling**: Progressive challenge levels
- **Performance Analytics**: Detailed success metrics

## Future Extensions

### Enhanced Structures
- **Multi-Object Scenes**: Complex arrangements
- **Articulated Objects**: Moving parts during rotation
- **Textured Surfaces**: Rich visual detail

### Advanced Rotations
- **Multi-Axis Rotations**: Complex 3D transformations
- **Partial Occlusions**: Hidden surface reasoning
- **Animation Sequences**: Extended rotation videos

### Interactive Elements
- **User-Guided Rotation**: Specific angle requests
- **Real-Time Generation**: Dynamic task creation
- **Adaptive Difficulty**: Performance-based scaling

---

## Conclusion

The 3D Mental Rotation Task provides a robust evaluation framework for spatial reasoning capabilities in video generation models. By combining sophisticated 3D structure generation with carefully controlled viewpoint transformations, this task creates meaningful challenges that test fundamental spatial cognition abilities.

The task's strength lies in its:
- **Objective Evaluation**: Clear success/failure criteria
- **Scalable Difficulty**: Progressive challenge levels  
- **Rich Metadata**: Detailed task characterization
- **Standard Format**: VMEvalKit compatibility

This makes it an excellent tool for benchmarking and advancing video model spatial reasoning capabilities.

**Happy spatial reasoning evaluation!** 🎯🔄
