# 5×5 Tetris Task for VMEvalKit

## 🎯 Task Description

The 5×5 Tetris task evaluates video generation models' capacity for **discrete grid dynamics reasoning** and **animation generation**. This task tests whether models can:

1. **Understand grid state** - Parse 5×5 board with filled/empty cells
2. **Apply Tetris rules** - Determine line clearing and gravity effects
3. **Generate animation sequences** - Produce videos showing flash-and-disappear line clear animations
4. **Demonstrate step-by-step reasoning** - Explain why lines clear or not

### Input Components
- **First Frame**: 5×5 grid with one falling piece (4 blocks in I/O/T/L/J shapes)
- **Prompt**: Text instruction to determine line clearing and animate the process
- **Format**: PNG image with clear grid lines and colored blocks

### Expected Output
- **Video Sequence**: Animation showing piece falling, line flash, and gravity application
- **Reasoning Text**: Step-by-step explanation referencing grid coordinates
- **Animation Specification**: Frame-indexed description of flash/disappear effects

### Core Rules
- **Line Clear**: Any row completely filled (all 5 cells) will flash and disappear
- **Gravity**: Blocks above cleared lines fall down to fill gaps
- **Piece Placement**: Falling piece locks at bottom-most valid position

## 📝 Data Structure

### Core Task Representation
```python
@dataclass
class TetrisTaskPair:
    # Identification
    id: str                      # Unique task identifier (e.g., "tetris_0042")
    task_category: str           # Always "Tetris"
    
    # Task Content
    prompt: str                  # Instruction text for the model
    first_image_path: str        # Path to initial board image
    final_image_path: str        # Path to final board (after clearing/gravity)
    
    # Grid Data
    grid_before: List[List[int]] # 5×5 array (0=empty, 1=filled)
    grid_after: List[List[int]]  # 5×5 array after simulation
    piece_shape: str             # "I", "O", "T", "L", or "J"
    piece_position: Tuple[int, int]  # (row, col) of piece anchor
    
    # Simulation Results
    lines_cleared: List[int]     # Row indices of cleared lines (e.g., [4])
    will_clear: bool            # True if any line clears
    
    # Metadata
    difficulty: str              # "easy", "medium", or "hard"
    guarantee_mode: str          # "clear", "no_clear", or "random"
    created_at: str             # ISO timestamp
```

### Grid Representation
```python
# 5×5 grid indexed [row][col], row 0 = top
grid = [
    [0, 0, 0, 0, 0],  # Row 0 (top)
    [0, 0, 0, 0, 0],  # Row 1
    [0, 0, 0, 0, 0],  # Row 2
    [0, 0, 0, 0, 0],  # Row 3
    ["I", "S", "Z", "J", 0],  # Row 4 (bottom) - 4/5 filled with shape labels
]
```

### Tetris Shapes (TetrisShape Enum)

Seven standard Tetromino shapes are supported:

```python
class TetrisShape(Enum):
    I = "I"  # Straight line (4 blocks)
    O = "O"  # Square (2×2, 4 blocks)
    T = "T"  # T shape (4 blocks)
    S = "S"  # S shape / zigzag (4 blocks)
    Z = "Z"  # Z shape / reverse zigzag (4 blocks)
    J = "J"  # J shape / reverse L (4 blocks)
    L = "L"  # L shape (4 blocks)
```

**Shape Visualizations:**

| Shape | Rotations | Example (TODO:make it correct) | Color |
|-------|-----------|-------------------------------|-------|
| **I** | 2 | `████` (horizontal) or vertical | Cyan (0, 255, 255) |
| **O** | 1 | `██`<br>`██` | Yellow (255, 255, 0) |
| **T** | 4 | `▀█▀`<br>` █ ` | Purple (128, 0, 128) |
| **S** | 2 | ` ██`<br>`██ ` | Green (0, 255, 0) |
| **Z** | 2 | `██ `<br>` ██` | Red (255, 0, 0) |
| **J** | 4 | `█  `<br>`███` | Blue (0, 0, 255) |
| **L** | 4 | `  █`<br>`███` | Orange (255, 165, 0) |

Each shape can be rotated (1-4 orientations depending on symmetry) and placed anywhere on the 5×5 grid. Grid cells are stored as shape labels (e.g., `"I"`, `"O"`) or `0` for empty.

## Generation Pipeline

(To be implemented)

## 🔧 Technical Implementation

### Dependencies
- **NumPy**: Grid array operations
- **Matplotlib**: Board visualization and image generation
- **Pillow (PIL)**: Image processing and file I/O
- **Dataclasses**: Structured task representation

### File Structure
```
vmevalkit/tasks/tetris_task/
├── __init__.py              # Module exports
├── PROMPTS.py              # Prompt templates with animation specs
├── tetris_reasoning.py     # Core implementation
└── README.md              # This documentation
```

### Key Classes
| Class | Purpose | Key Methods |
|-------|---------|-------------|
| `TetrisEasyTaskGenerator` | Easy difficulty (5x5, 2 rows) | `generate_single_task()`, `generate_dataset()` |
| `TetrisMediumTaskGenerator` | Medium difficulty (10x10, 3 rows) | `generate_single_task()`, `generate_dataset()` |
| `TetrisHardTaskGenerator` | Hard difficulty (10x10, block drop + clear) | `generate_single_task()`, `generate_dataset()` |
| `TetrisTaskPair` | Data structure | Dataclass with grid/piece data |
| `TetrisMap` | Game logic & state | `spawn_new_block()`, `hard_drop()`, `clear_lines()` |

## 🎚️ Difficulty Levels

### Easy
- **Map Size**: 5×5
- **Initial Rows**: 2 bottom rows pre-filled
- **Fill Ratio**: ~80%
- **Task**: Determine if any lines will clear (static puzzle, no block drop)
- **Guarantee Mode**: Can be `True` (guarantee clear), `False` (no clear), or `None` (random)

### Medium  
- **Map Size**: 10×10
- **Initial Rows**: 3 bottom rows pre-filled
- **Fill Ratio**: ~80%
- **Task**: Determine if any lines will clear (static puzzle, no block drop)
- **Guarantee Mode**: Same as Easy

### Hard
- **Map Size**: 10×10
- **Initial Rows**: 3 bottom rows pre-filled (guaranteed NO complete lines)
- **New Block**: A random Tetromino spawned at top
- **Task**: Simulate block drop (hard drop) → check line clear → show final state
- **Complexity**: Tests both **physics simulation** (block falling) AND **line clearing logic**
- **Key Difference**: Initial state MUST NOT have clearable lines; clearing only happens after block lands