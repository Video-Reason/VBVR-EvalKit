# Light Sequence Reasoning Task for VMEvalKit

## 📊 Overview

The Light Sequence task evaluates video generation models' capacity for **spatial reasoning** and **mathematical pattern recognition**. This task tests whether models can:

1. **Understand spatial positioning** - Identify specific light positions (1st, 2nd, nth from left)
2. **Recognize mathematical patterns** - Understand odd/even position concepts
3. **Process spatial ranges** - Handle left/right half specifications
4. **Handle multiple positions** - Process multiple simultaneous position requirements
5. **Follow complex instructions** - Combine spatial and mathematical reasoning

The light sequence tasks test spatial perception, mathematical reasoning, and instruction following capabilities in video models.

## 🚀 Usage

### Generate Light Sequence Tasks

Use the `create_questions.py` script to generate light sequence reasoning tasks:

```bash
# Generate all 768 light sequence tasks (default)
python examples/create_questions.py --task light_sequence

# Generate specific number of tasks (random sampling)
python examples/create_questions.py --task light_sequence --pairs-per-domain 100

python examples/generate_videos.py --model svd --task light_sequence
```

## 🎯 Task Description

### Input Components
- **First Frame**: A row of lights (4, 6, 8, or 10 lights) in a random on/off state
- **Prompt**: Text instruction asking to manipulate lights according to spatial/mathematical rules
- **Format**: 885×885px PNG image at 150 DPI with clear light representation

### Expected Output
- **Video Sequence**: Animation showing lights turning on/off
- **Final Frame**: Lights in the correct target state according to the instruction
- **Reasoning**: Proper understanding of spatial positions, patterns, and ranges

### Core Features
- **Four light counts**: 4, 6, 8, and 10 lights
- **14 initial states per count**: Fixed random initial configurations (excluding all-on/all-off states)
- **6 task types**: Different reasoning challenges
- **2 questions per type**: Variations within each type
- **Total 672 tasks**: 4 counts × 14 states × 6 types × 2 questions

## 📋 Task Types

### Type 1: Single Point Localization
Tests basic spatial positioning and counting from left to right.

**Questions:**
- Question 1: "Ensure the 1st light is on, and all other lights are off."
- Question 2: "Ensure the Nth light is on, and all other lights are off." (N = last position)

### Type 2: Multiple Point Localization
Tests ability to handle multiple simultaneous position requirements.

**Questions:**
- Question 1: Specific positions (varies by light count)
- Question 2: Different set of specific positions

### Type 3: Mathematical Pattern
Tests understanding of odd/even position concepts.

**Questions:**
- Question 1: "Ensure all lights at odd positions (counting from left to right) are on, and all other lights are off."
- Question 2: "Ensure all lights at even positions (counting from left to right) are on, and all other lights are off."

### Type 4: Spatial Range
Tests understanding of spatial divisions (left/right half).

**Questions:**
- Question 1: "Ensure the left half of the lights are on, and all other lights are off."
- Question 2: "Ensure the right half of the lights are on, and all other lights are off."

### Type 5: Continuous Sequence
Tests understanding of continuous position ranges.

**Questions:**
- Question 1: "Ensure the lights from Xth to Yth (counting from left to right) are on, and all other lights are off."
- Question 2: Different continuous range

### Type 6: Relative Position
Tests understanding of relative positioning (leftmost/rightmost).

**Questions:**
- Question 1: "Ensure the leftmost 2 lights are on, and all other lights are off."
- Question 2: "Ensure the rightmost 2 lights are on, and all other lights are off."

## 📊 Complete Task Definition

### 4 Lights - 14 Initial States

(Removed: `0000` - all off, `1111` - all on)

1. `1000` - 1st on
2. `0001` - 4th on
3. `0110` - 2nd, 3rd on
4. `1001` - 1st, 4th on
5. `0101` - 2nd, 4th on
6. `1010` - 1st, 3rd on
7. `1100` - 1st, 2nd on
8. `0011` - 3rd, 4th on
9. `1110` - 1st, 2nd, 3rd on
10. `0111` - 2nd, 3rd, 4th on
11. `1101` - 1st, 2nd, 4th on
12. `1011` - 1st, 3rd, 4th on
13. `0100` - 2nd on
14. `0010` - 3rd on

### 6 Lights - 14 Initial States

(Removed: `011100` - index 0, `101000` - index 3)

1. `100101` - 1st, 4th, 6th on
2. `010011` - 2nd, 5th, 6th on
3. `000110` - 4th, 5th on
4. `110010` - 1st, 2nd, 5th on
5. `001101` - 3rd, 4th, 6th on
6. `111001` - 1st, 2nd, 3rd, 6th on
7. `010100` - 2nd, 4th on
8. `101101` - 1st, 3rd, 4th, 6th on
9. `011010` - 2nd, 3rd, 5th on
10. `100011` - 1st, 5th, 6th on
11. `110100` - 1st, 2nd, 4th on
12. `001011` - 3rd, 5th, 6th on
13. `101110` - 1st, 3rd, 4th, 5th on
14. `010001` - 2nd, 6th on

### 8 Lights - 14 Initial States

(Removed: `11010101` - index 3, `01010011` - index 8)

1. `01101001` - 2nd, 3rd, 5th, 8th on
2. `10110010` - 1st, 3rd, 4th, 7th on
3. `01001100` - 2nd, 5th, 6th on
4. `00111011` - 3rd, 4th, 5th, 7th, 8th on
5. `10010110` - 1st, 4th, 6th, 7th on
6. `01100101` - 2nd, 3rd, 6th, 8th on
7. `10101100` - 1st, 3rd, 5th, 6th on
8. `11100010` - 1st, 2nd, 3rd, 7th on
9. `00011101` - 4th, 5th, 6th, 8th on
10. `11001011` - 1st, 2nd, 5th, 7th, 8th on
11. `01110100` - 2nd, 3rd, 4th, 6th on
12. `10100111` - 1st, 3rd, 6th, 7th, 8th on
13. `01011010` - 2nd, 4th, 5th, 7th on
14. `11001110` - 1st, 2nd, 5th, 6th, 7th on

### 10 Lights - 14 Initial States

(Removed: `0101101010` - index 2, `1010101101` - index 7)

1. `0110100110` - 2nd, 3rd, 5th, 8th, 9th on
2. `1011010011` - 1st, 3rd, 4th, 6th, 9th, 10th on
3. `1100110101` - 1st, 2nd, 5th, 6th, 8th, 10th on
4. `0011011100` - 3rd, 4th, 6th, 7th, 8th on
5. `1001010110` - 1st, 4th, 6th, 8th, 9th on
6. `0110011011` - 2nd, 3rd, 6th, 7th, 9th, 10th on
7. `0100110011` - 2nd, 5th, 6th, 9th, 10th on
8. `1110001010` - 1st, 2nd, 3rd, 7th, 9th on
9. `0001110101` - 4th, 5th, 6th, 8th, 10th on
10. `1101000111` - 1st, 2nd, 4th, 8th, 9th, 10th on
11. `0111100010` - 2nd, 3rd, 4th, 5th, 9th on
12. `1010011010` - 1st, 3rd, 6th, 7th, 9th on
13. `0101011011` - 2nd, 4th, 6th, 7th, 9th, 10th on
14. `1100101100` - 1st, 2nd, 5th, 7th, 8th on

## 📝 Type Definitions by Light Count

### 4 Lights

**Type 1: Single Point**
- Q1: Position 1
- Q2: Position 4

**Type 2: Multiple Points**
- Q1: Positions 1, 3
- Q2: Positions 2, 4

**Type 3: Mathematical Pattern**
- Q1: Odd positions (1, 3)
- Q2: Even positions (2, 4)

**Type 4: Spatial Range**
- Q1: Left half (positions 1, 2)
- Q2: Right half (positions 3, 4)

**Type 5: Continuous Sequence**
- Q1: Positions 1-3
- Q2: Positions 2-4

**Type 6: Relative Position**
- Q1: Leftmost 2 (positions 1, 2)
- Q2: Rightmost 2 (positions 3, 4)

### 6 Lights

**Type 1: Single Point**
- Q1: Position 1
- Q2: Position 6

**Type 2: Multiple Points**
- Q1: Positions 1, 3, 5
- Q2: Positions 2, 4, 6

**Type 3: Mathematical Pattern**
- Q1: Odd positions (1, 3, 5)
- Q2: Even positions (2, 4, 6)

**Type 4: Spatial Range**
- Q1: Left half (positions 1, 2, 3)
- Q2: Right half (positions 4, 5, 6)

**Type 5: Continuous Sequence**
- Q1: Positions 1-4
- Q2: Positions 3-6

**Type 6: Relative Position**
- Q1: Leftmost 2 (positions 1, 2)
- Q2: Rightmost 2 (positions 5, 6)

### 8 Lights

**Type 1: Single Point**
- Q1: Position 1
- Q2: Position 8

**Type 2: Multiple Points**
- Q1: Positions 2, 5, 7
- Q2: Positions 1, 4, 8

**Type 3: Mathematical Pattern**
- Q1: Odd positions (1, 3, 5, 7)
- Q2: Even positions (2, 4, 6, 8)

**Type 4: Spatial Range**
- Q1: Left half (positions 1, 2, 3, 4)
- Q2: Right half (positions 5, 6, 7, 8)

**Type 5: Continuous Sequence**
- Q1: Positions 2-5
- Q2: Positions 4-7

**Type 6: Relative Position**
- Q1: Leftmost 2 (positions 1, 2)
- Q2: Rightmost 2 (positions 7, 8)

### 10 Lights

**Type 1: Single Point**
- Q1: Position 1
- Q2: Position 10

**Type 2: Multiple Points**
- Q1: Positions 1, 4, 7, 9
- Q2: Positions 2, 5, 8, 10

**Type 3: Mathematical Pattern**
- Q1: Odd positions (1, 3, 5, 7, 9)
- Q2: Even positions (2, 4, 6, 8, 10)

**Type 4: Spatial Range**
- Q1: Left half (positions 1, 2, 3, 4, 5)
- Q2: Right half (positions 6, 7, 8, 9, 10)

**Type 5: Continuous Sequence**
- Q1: Positions 3-7
- Q2: Positions 4-8

**Type 6: Relative Position**
- Q1: Leftmost 2 (positions 1, 2)
- Q2: Rightmost 2 (positions 9, 10)

## 📊 Data Scale

- **4 lights**: 14 initial states × 6 types × 2 questions = 168 tasks
- **6 lights**: 14 initial states × 6 types × 2 questions = 168 tasks
- **8 lights**: 14 initial states × 6 types × 2 questions = 168 tasks
- **10 lights**: 14 initial states × 6 types × 2 questions = 168 tasks
- **Total**: 672 tasks

**Note**: All-on and all-off states are excluded to ensure models can observe both on and off lights in the initial state, enabling proper understanding of how to manipulate the lights.

## 🎨 Visual Design

The light sequence image features:
- **Horizontal row** of circular lights
- **Gold/yellow color** when light is on
- **Gray color** when light is off
- **Even spacing** between lights
- **Clean white background** for maximum contrast
- **Glow effect** for on lights to enhance visibility

## 🔗 Related Resources

- [VMEvalKit Documentation](../../../README.md)
- [Adding Tasks Guide](../../../docs/ADDING_TASKS.md)
- Other reasoning tasks: Chess, Maze, Sudoku, Clock, Control Panel

