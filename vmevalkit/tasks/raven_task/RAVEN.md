# RAVEN Progressive Matrix Reasoning Task Documentation

## Overview

The RAVEN Progressive Matrix Reasoning Task evaluates video generation models' ability to demonstrate **abstract visual reasoning** and **pattern completion** by generating videos that show the logical process of completing Raven's Progressive Matrices (RPM). This task tests analogical reasoning, relational understanding, and rule-based pattern recognition capabilities.

Based on the **CVPR 2019 RAVEN dataset** for **R**elational and **A**nalogical **V**isual r**E**aso**N**ing.

## Task Structure

### Core Concept
- **First Frame**: Shows 8 panels of a 3×3 Progressive Matrix with the 9th panel missing (marked with "?")  
- **Final Frame**: Shows the complete 3×3 matrix with the correct 9th panel filled in
- **Video Task**: Model must generate video showing the reasoning process to determine the missing panel
- **Text Prompt**: Provides configuration-specific instructions and reasoning hints

## Configuration Types

The task supports **7 different figural configurations** from the original RAVEN dataset:

### 1. Center Configuration
- **Description**: Single centered element with systematic transformations
- **Pattern**: Elements arranged around central focus
- **Example Prompt**: "Complete this center-focused pattern matrix"

### 2. 2×2Grid Configuration  
- **Description**: Four-element grid patterns with systematic relationships
- **Pattern**: 2×2 arrangements with rule-based variations
- **Example Prompt**: "Complete this 2×2 grid pattern matrix"

### 3. 3×3Grid Configuration
- **Description**: Nine-element grid patterns with complex relationships
- **Pattern**: 3×3 arrangements with multi-level rule applications  
- **Example Prompt**: "Complete this 3×3 grid pattern matrix"

### 4. Left-Right Configuration
- **Description**: Horizontal relationships between left and right elements
- **Pattern**: Systematic transformations across horizontal axis
- **Example Prompt**: "Complete this left-right pattern matrix"

### 5. Up-Down Configuration
- **Description**: Vertical relationships between upper and lower elements  
- **Pattern**: Systematic transformations across vertical axis
- **Example Prompt**: "Complete this up-down pattern matrix"

### 6. Out-InCenter Configuration
- **Description**: Relationship between outer elements and inner center
- **Pattern**: Outside-inside transformations with central focus
- **Example Prompt**: "Complete this outside-inside center pattern matrix"

### 7. Out-InGrid Configuration
- **Description**: Relationship between outer grid and inner elements
- **Pattern**: Complex outside-inside transformations with grid structure
- **Example Prompt**: "Complete this outside-inside grid pattern matrix"

## Rule Types

**Note: Current implementation focuses on simplified rules (Constant and Progression) for reliable generation. More complex rules (Arithmetic, Distribute_Three) are documented below but not actively used in the current simplified version.**

### Abstract Reasoning Rules

#### 1. Constant Rules (Easy)
- **Description**: Certain attributes remain unchanged across panels
- **Application**: Number/Position, Type, Size, or Color stays fixed
- **Reasoning**: Identify invariant elements

#### 2. Progression Rules (Medium)  
- **Description**: Systematic step-by-step changes across panels
- **Variations**:
  - **Number/Position**: +1, +2, -1, -2 changes
  - **Type**: Shape progression (triangle→square→pentagon→hexagon→circle)
  - **Size**: Systematic size increases/decreases  
  - **Color**: Systematic color shifts
- **Reasoning**: Detect sequential patterns

#### 3. Arithmetic Rules (Hard)
- **Description**: Mathematical relationships between panels
- **Operations**:
  - **Addition/Subtraction**: Panel₃ = Panel₁ ± Panel₂
  - **Set Operations**: Union/Difference of elements
- **Reasoning**: Apply mathematical logic to visual elements

#### 4. Distribute_Three Rules (Medium)
- **Description**: Three distinct values distributed systematically  
- **Application**: Three numbers, positions, types, sizes, or colors across rows
- **Reasoning**: Recognize systematic distribution patterns

## Data Structure

### RavenTaskPair
Each task consists of a pair of Progressive Matrix images and reasoning prompt:

```python
@dataclass  
class RavenTaskPair:
    id: str                          # "raven_0001"
    prompt: str                      # Configuration-specific instruction
    first_image_path: str           # Incomplete matrix (8 panels + empty)
    final_image_path: str           # Complete matrix (all 9 panels)
    task_category: str              # Configuration type
    raven_data: Dict[str, Any]      # Generation and rule metadata
    difficulty: str                 # "easy", "medium", "hard"
    rule_types: List[str]           # Applied rules
    configuration_type: str         # Configuration display name
    created_at: str                 # Timestamp
```

### RAVEN Data Structure
The `raven_data` field contains detailed task information:

```python
raven_data = {
    "generation_method": "RAVEN Progressive Matrix Generator",
    "configuration": "center_single",           # Internal configuration name
    "rule_groups": {                           # Detailed rule applications
        "component_0": [                       # Rules for component 0
            {
                "name": "Progression", 
                "attr": "Size",
                "value": 2                     # Step size
            }
        ]
    },
    "primary_rules": ["Progression", "Constant"],  # Main rule types used
    "matrix_size": "160x160",                      # Panel dimensions
    "pattern_type": "Progressive Matrix"
}
```

## Visual Representation

### First Frame (Incomplete Matrix)
```
┌─────┬─────┬─────┐
│ P₁  │ P₂  │ P₃  │  
├─────┼─────┼─────┤
│ P₄  │ P₅  │ P₆  │  
├─────┼─────┼─────┤
│ P₇  │ P₈  │  ?  │  ← Missing panel
└─────┴─────┴─────┘
```

### Final Frame (Complete Matrix)  
```
┌─────┬─────┬─────┐
│ P₁  │ P₂  │ P₃  │  
├─────┼─────┼─────┤
│ P₄  │ P₅  │ P₆  │  
├─────┼─────┼─────┤
│ P₇  │ P₈  │ P₉  │  ← Completed panel
└─────┴─────┴─────┘
```

## Difficulty Classification

### Easy Tasks
- **Rules**: Primarily Constant rules
- **Complexity**: Single rule application
- **Example**: All shapes remain triangles, only color changes
- **Reasoning**: Basic pattern recognition

### Medium Tasks  
- **Rules**: Progression, Distribute_Three, or multiple rules
- **Complexity**: Multi-step reasoning required
- **Example**: Size increases +2 steps, type progresses through shapes
- **Reasoning**: Sequential logic and distribution patterns

### Hard Tasks
- **Rules**: Arithmetic rules or complex combinations
- **Complexity**: Mathematical reasoning required  
- **Example**: Element₉ = Element₁ + Element₂ (set union)
- **Reasoning**: Abstract mathematical operations on visual elements

## Evaluation Criteria

### Core Reasoning Assessment
1. **Pattern Recognition**: Can the model identify the underlying rule?
2. **Logical Completion**: Is the 9th panel correctly determined?
3. **Process Demonstration**: Does the video show logical reasoning steps?
4. **Rule Consistency**: Are the identified rules applied correctly?

### Video Quality Metrics
1. **Clarity**: Is the reasoning process visually clear?
2. **Completeness**: Does the video show the full reasoning sequence?
3. **Accuracy**: Is the final panel completion correct?
4. **Coherence**: Is the reasoning logically consistent throughout?

## Usage Examples

### Basic Generation
```python
from vmevalkit.tasks.raven_task import RavenGenerator, create_dataset

# Generate 50 RAVEN tasks
dataset = create_dataset(num_samples=50)
print(f"Generated {len(dataset['pairs'])} Progressive Matrix tasks")

# Access individual tasks
task = dataset['pairs'][0]
print(f"Task: {task['prompt']}")
print(f"Category: {task['task_category']}")  
print(f"Rules: {task['rule_types']}")
```

### Custom Configuration
```python  
generator = RavenGenerator()

# Generate specific configuration
task_data = generator.generate_single_task("center_single", difficulty="hard")
print(f"Generated {task_data['config_display']} task with rules: {task_data['rules']['primary_rules']}")
```

## Integration with VMEvalKit

### Runner Compatibility
The RAVEN task integrates seamlessly with VMEvalKit's evaluation pipeline:

```python
# Load RAVEN dataset
from vmevalkit.runner.inference import run_inference

# Run evaluation on RAVEN tasks
results = run_inference(
    model_name="your_model",
    dataset_path="data/questions/raven_tasks/raven_tasks.json",
    task_type="raven_reasoning"
)
```

### Performance Baselines
Based on the original RAVEN paper benchmarks:

| Model Type | Accuracy | Notes |
|------------|----------|-------|
| **Human Performance** | **84.41%** | Average across all configurations |
| ResNet+DRT | 59.56% | Best automated performance |
| CNN+DRT | 39.42% | Standard CNN approach |
| Random Baseline | 12.5% | Random selection from 8 choices |

## Technical Implementation

### Matrix Generation
- Uses the original RAVEN attributed stochastic image grammar
- Generates systematic rule-based transformations
- Creates 160×160 pixel panels with clear geometric shapes
- Supports all original attribute types: Number, Position, Type, Size, Color

### Image Processing  
- **Format**: PNG images at 100 DPI
- **Layout**: 3×3 grid layout with clear panel borders
- **Incomplete**: 9th panel shows "?" placeholder  
- **Complete**: All panels filled with appropriate elements

### Rule Validation
- Validates all generated matrices for rule consistency
- Ensures unique solutions for each Progressive Matrix
- Filters out ambiguous or invalid task configurations
- Maintains difficulty balance across rule types

## Research Applications

### Abstract Reasoning Research
- **Analogical Thinking**: A:B :: C:? relationships in visual domain
- **Rule Learning**: Can models learn abstract transformation rules?
- **Transfer Learning**: Do learned patterns transfer across configurations?
- **Systematic Generalization**: Performance across unseen rule combinations

### Video Reasoning Evaluation
- **Process Visualization**: How do models show their reasoning?
- **Step-by-Step Logic**: Can models break down complex reasoning?
- **Visual Communication**: How clearly can models demonstrate abstract thinking?
- **Temporal Consistency**: Do reasoning steps follow logically in sequence?

## Limitations and Considerations

### Scope Constraints
- **2D Patterns Only**: Limited to 2D geometric transformations
- **Fixed Grid**: Always 3×3 matrix structure  
- **Geometric Shapes**: Focuses on basic geometric primitives
- **Rule-Based**: May not capture all forms of visual reasoning

### Evaluation Challenges
- **Subjectivity**: Some reasoning processes may have multiple valid demonstrations
- **Complexity**: Hard to evaluate partial reasoning steps automatically
- **Ambiguity**: Distinguishing between lucky guesses and genuine understanding

## Future Extensions

### Enhanced Complexity
- **Temporal Matrices**: Progressive matrices that change over time
- **3D Reasoning**: Spatial reasoning with 3D transformations
- **Multi-Modal**: Combining visual patterns with textual rules
- **Interactive**: Progressive matrices that respond to user input

### Advanced Evaluation
- **Process Scoring**: Evaluate reasoning steps, not just final answers  
- **Creativity Metrics**: Assess novel approaches to pattern completion
- **Robustness Testing**: Performance under visual noise or partial occlusion
- **Explanability**: Natural language explanation of reasoning process

## Conclusion

The RAVEN Progressive Matrix task brings **abstract visual reasoning** evaluation to video models, complementing VMEvalKit's spatial (maze) and strategic (chess) reasoning tasks. It tests fundamental cognitive capabilities:

- **Pattern Recognition**: Identifying systematic transformations
- **Analogical Reasoning**: Understanding A:B :: C:? relationships  
- **Rule Application**: Applying abstract logical rules to visual data
- **Process Demonstration**: Showing reasoning steps through video

This creates a comprehensive evaluation framework for **higher-order reasoning capabilities** in video generation models, moving beyond basic visual quality to test genuine **abstract intelligence**.

The task is particularly valuable for:
- 🧠 **Cognitive AI Research**: Testing abstract reasoning capabilities
- 📊 **Model Benchmarking**: Standardized progressive matrix evaluation  
- 🔬 **Reasoning Analysis**: Understanding how models approach pattern completion
- 🎯 **Intelligence Assessment**: Measuring genuine vs. superficial pattern matching

**RAVEN tasks push video models beyond visual generation to demonstrate true reasoning intelligence.**
