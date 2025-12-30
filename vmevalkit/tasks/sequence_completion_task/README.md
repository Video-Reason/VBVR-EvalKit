# Sequence Completion Reasoning Task for VMEvalKit

## 📊 Overview

The Sequence Completion task evaluates video generation models' capacity for **pattern recognition** and **logical extrapolation**. This task tests whether models can:

1. **Recognize mathematical patterns** - Understand arithmetic, geometric, power, and Fibonacci sequences
2. **Identify cyclic patterns** - Recognize repeating cycles in shapes, colors, and positions
3. **Handle mixed attributes** - Process sequences with combined attributes (color+shape, color+position, shape+position)
4. **Extrapolate sequences** - Predict the next element based on observed patterns
5. **Demonstrate reasoning** - Show understanding through video generation

The sequence completion tasks test pattern recognition, mathematical reasoning, and logical extrapolation capabilities in video models.

## 🚀 Usage

### Generate Sequence Completion Tasks

Use the `create_questions.py` script to generate sequence completion reasoning tasks:

```bash
# Generate all sequence completion tasks (default: all 1242+ tasks)
python examples/create_questions.py --task sequence_completion

# Generate specific number of tasks (random sampling)
python examples/create_questions.py --task sequence_completion --pairs-per-domain 100

python examples/generate_videos.py --model svd --task sequence_completion
```

## 🎯 Task Description

### Input Components
- **First Frame**: A sequence of elements with the last element missing (shown as "?")
- **Prompt**: Text instruction asking to complete the sequence by identifying the pattern
- **Format**: PNG image showing sequence elements (numbers, shapes, colors, positions, or mixed)

### Expected Output
- **Video Sequence**: Animation showing the sequence being completed
- **Final Frame**: The complete sequence with the correct next element
- **Reasoning**: Proper understanding of the pattern and correct extrapolation

### Core Features
- **8 task types**: Different pattern types (arithmetic, geometric, power, Fibonacci, cycles, mixed)
- **Multiple sequence lengths**: 5-8 elements depending on type
- **Visual elements**: Numbers, shapes (○, □, △, ◇, star), colors (red, blue, green, yellow, orange), positions (top, bottom, left, right, center, etc.)
- **Total 1242+ tasks**: Comprehensive coverage of pattern types

## 📋 Task Types

### Type 1: Arithmetic Sequence
Tests understanding of linear number sequences with constant difference.

**Pattern**: Each element increases/decreases by a fixed step
**Example**: [1, 2, 3, 4, ?] → Answer: 5
**Parameters**: start[1-15], step[-5,-2,-1,1,2,5], length[5,6,7]
**Total**: 180 tasks

### Type 2: Geometric Sequence
Tests understanding of exponential number sequences with constant ratio.

**Pattern**: Each element is multiplied by a fixed ratio
**Example**: [1, 2, 4, 8, ?] → Answer: 16
**Parameters**: start[1-30], ratio[2,3,4], length[5,6,7] (values ≤ 1000)
**Total**: 95 tasks

### Type 3: Power Sequence
Tests understanding of square number sequences.

**Pattern**: Each element is the square of consecutive integers
**Example**: [1, 4, 9, 16, ?] → Answer: 25
**Parameters**: base[1-10], power[2], length[5,6] (values ≤ 100)
**Total**: 20 tasks

### Type 4: Fibonacci Sequence
Tests understanding of additive sequences where each element is the sum of previous two.

**Pattern**: Each element is the sum of the two preceding elements
**Example**: [1, 1, 2, 3, 5, ?] → Answer: 8
**Parameters**: first[1-9], second[1-9], length[6,7]
**Total**: 162 tasks

### Type 5: Shape Cycle
Tests understanding of repeating shape patterns.

**Pattern**: Shapes repeat in a fixed cycle
**Examples**: 
- Cycle [○, □, △]: [○, □, △, ○, □, ?] → Answer: △
- Cycle [◇, star]: [◇, star, ◇, star, ?] → Answer: ◇
**Parameters**: cycle_length[3,4,5], shapes[○,□,△,◇,star], length[5-8]
**Total**: 141 tasks

### Type 6: Color Cycle
Tests understanding of repeating color patterns.

**Pattern**: Colors repeat in a fixed cycle
**Examples**: 
- Cycle [red, blue, green]: [red, blue, green, red, blue, ?] → Answer: green
- Cycle [yellow, orange]: [yellow, orange, yellow, orange, ?] → Answer: yellow
**Parameters**: cycle_length[3,4], colors[red,blue,green,yellow,orange], length[5-8]
**Total**: 110 tasks

### Type 7: Direction Cycle
Tests understanding of repeating direction patterns.

**Pattern**: Directions repeat in a fixed cycle
**Examples**: 
- Cycle [top, bottom, left]: [top, bottom, left, top, bottom, ?] → Answer: left
- Cycle [left, right]: [left, right, left, right, ?] → Answer: left
**Parameters**: cycle_length[3,4,5], directions[various combinations], length[5-8]
**Total**: 54 tasks

### Type 8: Mixed Sequence
Tests understanding of sequences with combined attributes.

**Pattern**: Elements combine color and shape attributes and repeat in a fixed cycle
**Examples**: 
- Color+Shape cycle [red○, blue□, green△]: [red○, blue□, green△, red○, blue□, ?] → Answer: green△
- Color+Shape cycle [yellow◇, orange○]: [yellow◇, orange○, yellow◇, orange○, ?] → Answer: yellow◇
**Parameters**: mixed_type[color_shape], length[6,7,8]
**Total**: 144 tasks (48 cycles × 3 lengths)

## 📊 Task Statistics

| Type | Description | Count |
|------|-------------|-------|
| Type 1 | Arithmetic Sequence | 180 |
| Type 2 | Geometric Sequence | 95 |
| Type 3 | Power Sequence | 20 |
| Type 4 | Fibonacci Sequence | 162 |
| Type 5 | Shape Cycle | 141 |
| Type 6 | Color Cycle | 110 |
| Type 7 | Direction Cycle | 54 |
| Type 8 | Mixed Sequence | 144 |
| **Total** | | **906+** |

## 🎨 Visual Design

### Element Rendering
- **Numbers**: Displayed in rounded boxes with bold text
- **Shapes**: Rendered as geometric shapes (circles, squares, triangles, diamonds, stars)
- **Colors**: Displayed as colored circles with Chinese text labels
- **Positions**: Shown as text in colored boxes
- **Mixed Elements**: Combined visual representation (e.g., colored shapes)

### Image Format
- **Size**: ~800×200px (adjustable via figsize parameter)
- **DPI**: 150
- **Format**: PNG with transparent background
- **Layout**: Horizontal sequence with equal spacing

## 🔍 Evaluation

The task is evaluated by comparing the final frame with the expected completion:
- **Correct**: The next element matches the expected answer
- **Pattern Recognition**: Model correctly identified the sequence pattern
- **Extrapolation**: Model correctly predicted the next element

## 📝 Notes

- All sequences are designed to have clear, unambiguous patterns
- The missing element (shown as "?") is always the last element in the sequence
- Sequences are designed to test reasoning, not memorization
- Mixed sequences test multi-attribute pattern recognition

