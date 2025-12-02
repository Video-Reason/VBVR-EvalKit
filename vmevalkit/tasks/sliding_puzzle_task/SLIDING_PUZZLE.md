# Sliding Puzzle Task Documentation

## Overview

The Sliding Puzzle Task evaluates video generation models' ability to demonstrate **spatial reasoning**, **simple planning**, and **visual consistency** by generating videos that show the completion of near-complete sliding puzzles.

This task introduces a simplified, cognitively appropriate benchmark where models must:
- Understand near-complete puzzle states
- Recognize which tiles need to be moved
- Execute 1-2 simple sliding moves
- Generate visually consistent tile movements

## Task Description

### Core Challenge

Models must:
1. **Parse Puzzle State**: Understand the current arrangement of numbered tiles
2. **Identify Goal**: Recognize the target configuration (numbers in order)
3. **Plan Moves**: Determine which tile(s) to move (1-2 moves)
4. **Execute Moves**: Generate video showing tile(s) sliding into correct positions

### Visual Format

- **Canvas Size**: 400×400 pixels (default)
- **Puzzle Sizes**: 3×3 (8 tiles + 1 blank) or 4×4 (15 tiles + 1 blank)
- **Tile Design**: Numbered tiles (1, 2, 3, ...) with blue background
- **Empty Space**: Gray tile marked "EMPTY"
- **Grid Lines**: Clear borders between tiles

## Design Philosophy

### Why Near-Complete Puzzles?

**Key Simplification**: Puzzles are designed to be **near-complete**, requiring only **1-2 moves** to solve.

**Rationale**:
- ✅ **Appropriate for Video Models**: Avoids complex multi-step planning beyond model capabilities
- ✅ **Tests Core Abilities**: Focuses on spatial reasoning, move understanding, visual consistency
- ✅ **Easy to Verify**: Clear success criteria (1-2 moves)
- ✅ **Avoids Over-Complexity**: Planning difficulty doesn't mask other cognitive abilities

### Difficulty Levels

#### Easy
- **Size**: 3×3
- **Solution Length**: 1 move
- **Generation**: From complete state, make 1 random move
- **Test Focus**: Basic move understanding, single tile movement

#### Medium
- **Size**: 3×3 or 4×4 (random)
- **Solution Length**: 2 moves
- **Generation**: From complete state, make 2 random moves
- **Test Focus**: Sequential moves, multi-step understanding

#### Hard
- **Size**: 4×4
- **Solution Length**: 2-3 moves
- **Generation**: From complete state, make 2-3 random moves
- **Test Focus**: Slightly more complex planning, larger grid

## Cognitive Abilities Tested

### 1. Spatial Reasoning
- Understanding 2D grid space
- Recognizing tile positions relative to goal
- Understanding move constraints (only tiles adjacent to empty space can move)

### 2. Simple Planning
- Identifying which tile(s) need to move (1-2 tiles)
- Understanding move directions (up/down/left/right)
- Executing 1-2 simple moves

### 3. Visual Consistency
- Maintaining tile visual features (numbers, colors)
- Correctly displaying sliding animation
- Understanding empty space movement

### 4. Goal Recognition
- Recognizing near-complete states
- Understanding completion condition (numbers in order)
- Executing final move steps

## Generation Algorithm

### Step 1: Create Goal State
```python
# 3×3 example:
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 0]]  # 0 = empty space
```

### Step 2: Make Reverse Moves
- Start from goal state
- Make N random moves (N = 1-3 based on difficulty)
- Avoid moving back to previous position (unless only option)

### Step 3: Generate Images
- **first_frame.png**: Near-complete state (after N reverse moves)
- **final_frame.png**: Goal state (complete puzzle)

### Step 4: Create Prompt
- Instructions telling model to complete the puzzle
- Mention that only 1-2 moves are needed

## Example Task

### First Frame (Near-Complete)
```
┌─────┬─────┬─────┐
│  1  │  2  │  3  │
├─────┼─────┼─────┤
│  4  │  5  │  6  │
├─────┼─────┼─────┤
│  7  │EMPTY│  8  │
└─────┴─────┴─────┘
```
**Solution**: Move tile 8 left into empty space (1 move)

### Final Frame (Complete)
```
┌─────┬─────┬─────┐
│  1  │  2  │  3  │
├─────┼─────┼─────┤
│  4  │  5  │  6  │
├─────┼─────┼─────┤
│  7  │  8  │EMPTY│
└─────┴─────┴─────┘
```

## Prompt Examples

- "Complete this sliding puzzle by moving the numbered tiles to their correct positions. Only one or two moves are needed. Show the tile(s) sliding into place."
- "Finish solving this sliding puzzle. Move the tile(s) adjacent to the empty space to complete the puzzle. Demonstrate the final move(s)."
- "This sliding puzzle is almost complete. Make the final move(s) to arrange all numbered tiles in order. Show the tile(s) sliding into the correct position(s)."

## Data Structure

```python
{
    "id": "sliding_puzzle_0001",
    "prompt": "Complete this sliding puzzle...",
    "first_image_path": "sliding_puzzle_task/sliding_puzzle_0001/first_frame.png",
    "final_image_path": "sliding_puzzle_task/sliding_puzzle_0001/final_frame.png",
    "task_category": "SlidingPuzzle",
    "difficulty": "easy",
    "puzzle_size": (3, 3),
    "initial_state": [[1,2,3],[4,5,6],[7,0,8]],
    "goal_state": [[1,2,3],[4,5,6],[7,8,0]],
    "solution_length": 1,
    "num_moves_from_complete": 1,
    "puzzle_data": {
        "generation_method": "near_complete",
        "solution_length": 1,
        "num_moves_from_complete": 1
    }
}
```

## Implementation Details

### Puzzle Generation
- **Method**: Reverse moves from goal state
- **Guarantee**: Always solvable (since we reverse from goal)
- **Randomization**: Random valid moves, avoiding immediate backtracking

### Image Rendering
- **Library**: matplotlib
- **Style**: Clear grid, numbered tiles, high contrast
- **Size**: 400×400 pixels (adjustable)
- **Format**: PNG

### Validation
- All puzzles are guaranteed solvable (reverse from goal)
- Solution length matches num_moves_from_complete
- Images are clear and unambiguous

## Usage

```python
from vmevalkit.tasks.sliding_puzzle_task import create_dataset

# Generate 50 tasks
dataset = create_dataset(num_samples=50)

# Access pairs
for pair in dataset['pairs']:
    print(f"Task {pair['id']}: {pair['difficulty']}, {pair['solution_length']} moves")
```

## Domain Registry

Registered in `vmevalkit/runner/TASK_CATALOG.py`:

```python
'sliding_puzzle': {
    'name': 'Sliding Puzzle',
    'description': 'Spatial reasoning and simple planning through near-complete sliding puzzles',
    'module': 'vmevalkit.tasks.sliding_puzzle_task',
    'create_function': 'create_dataset',
    'process_dataset': lambda dataset, num_samples: dataset['pairs']
}
```

