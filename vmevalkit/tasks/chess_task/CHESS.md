# Chess Reasoning Task Documentation

## Overview

The Chess Reasoning Task evaluates video generation models' ability to demonstrate strategic thinking and tactical pattern recognition by generating videos that show the solution to chess mate-in-1 positions. This task tests spatial reasoning, pattern recognition, strategic thinking, and precise action demonstration capabilities.

## Task Types

### Mate-in-1 Tasks
- **Description**: Chess positions where one side can deliver checkmate in exactly one move
- **Visual Style**: Standard chess board with pieces in FEN notation
- **Goal**: Find and demonstrate the winning move that delivers immediate checkmate
- **Difficulty Levels**: Easy (basic patterns) → Medium (tactical patterns) → Hard (complex positions)
- **Pattern Types**: Back-rank mates, corner mates, queen/rook/knight tactics, discovered attacks

### Pattern Categories

#### Back-Rank Mates
- **Description**: King trapped on back rank by own pieces, attacked by rook/queen
- **Example**: `6k1/5ppp/8/8/8/8/8/R6K w - - 0 1` → Ra8#
- **Tags**: `["back_rank", "rook"]`

#### Queen Corner Mates  
- **Description**: Queen + King coordination against cornered enemy king
- **Example**: `6Qk/8/6K1/8/8/8/8/8 w - - 0 1` → Qh7#
- **Tags**: `["queen", "corner", "endgame"]`

#### Tactical Mates
- **Description**: Knight forks, discoveries, pins leading to mate
- **Example**: Various tactical motifs with immediate mate threats
- **Tags**: `["knight", "fork"]`, `["discovery"]`, `["pin"]`

## Data Structure

### ChessTaskPair
Each task consists of a pair of chess board images and a text prompt:

```python
@dataclass
class ChessTaskPair:
    id: str                         # Unique identifier (e.g., "chess_0001")
    prompt: str                     # Instructions for the video model
    first_image_path: str           # Path to initial chess position image
    final_image_path: str           # Path to position after mate move
    task_category: str              # "Mate-in-1"
    chess_data: Dict[str, Any]      # Position and solution metadata
    difficulty: str                 # "easy", "medium", or "hard"
    side_to_move: str              # "white" or "black"
    mate_moves: List[str]          # List of valid mate moves (SAN notation)
    pattern_tags: List[str]        # Pattern classification tags
    created_at: str                # Timestamp of creation
```

### ChessDataset  
A collection of chess reasoning task pairs with metadata:

```python
@dataclass
class ChessDataset:
    name: str                       # Dataset identifier
    description: str                # Human-readable description
    pairs: List[ChessTaskPair]     # List of chess task pairs
    metadata: Dict[str, Any]        # Additional dataset information
```

### Chess Data Structure
The `chess_data` field contains detailed position information:

```python
chess_data = {
    "initial_fen": "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",  # Starting position
    "final_fen": "R5k1/5ppp/8/8/8/8/8/6K b - - 0 1",    # Position after mate
    "mate_move": "Ra8#",                                   # Winning move (SAN)
    "all_mate_moves": ["Ra8#"],                           # All possible mates
    "piece_moved": "a1",                                  # From square
    "target_square": "a8",                                # To square
    "is_capture": False,                                  # Whether move captures
    "gives_check": True,                                  # Always true for mates
    "pattern_description": "Classic back-rank mate"       # Human description
}
```

## Visual Representation

### First Frame (Initial Position)
- Shows the chess board with the mate-in-1 position
- All pieces positioned according to the FEN string
- Standard chess board colors (light/dark squares)
- Clear piece distinction (white/black pieces)

### Final Frame (Solution)
- Shows the chess board after the mate move has been played
- The winning piece has moved to its target square
- Target square may be highlighted to show the move
- Position demonstrates checkmate has been achieved

### Board Rendering
- **Format**: PNG images (matching maze dataset format)
- **Size**: 400x400 pixels (configurable)
- **Style**: Standard chess notation and piece symbols
- **Highlighting**: Optional square highlighting for move visualization
- **Check Indication**: Visual indication when king is in check

## Prompts

### Basic Prompts
- "White to move. Find checkmate in one move."
- "Black can deliver checkmate in one move. Show the winning move."
- "It's white's turn. Demonstrate the checkmate in one move."
- "White to play and mate in 1. Show the solution."

### Pattern-Specific Prompts
- "White to move. Find the back-rank mate." (for back-rank patterns)
- "Deliver back-rank checkmate with white."
- "Use the queen to deliver mate." (for queen tactics)
- "Queen delivers mate in one - show the move."
- "Use the rook to deliver checkmate." (for rook patterns)

### Difficulty-Based Selection
- **Easy**: Basic prompts focusing on simple instruction
- **Medium**: Mix of basic and pattern-specific prompts
- **Hard**: Emphasis on pattern-specific prompts requiring tactical knowledge

## Usage

### Generating Datasets

```python
from vmevalkit.tasks.chess_task import (
    create_chess_dataset,
    ChessDataset,
    generate_chess_board_image
)

# Generate chess dataset with custom parameters
dataset = create_chess_dataset(
    num_samples=50,
    difficulty_distribution={"easy": 0.7, "medium": 0.2, "hard": 0.1},
    output_dir="data"
)

# Generate individual chess board image
generate_chess_board_image(
    fen="6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",
    output_path="chess_position.png",
    highlight_squares=["a8"]  # Highlight target square
)
```

### Loading Existing Datasets

```python
from vmevalkit.tasks.chess_task import ChessDataset

# Load from JSON file
dataset = ChessDataset.load("data/questions/chess_tasks/chess_mate_in_1_tasks.json")

# Filter by difficulty
easy_tasks = dataset.filter_by_difficulty("easy")
hard_tasks = dataset.filter_by_difficulty("hard")

# Filter by side to move
white_to_move = dataset.filter_by_side("white")
black_to_move = dataset.filter_by_side("black")

# Access individual pairs
for pair in dataset.pairs:
    print(f"Task: {pair.id}")
    print(f"Prompt: {pair.prompt}")
    print(f"Difficulty: {pair.difficulty}")
    print(f"Solutions: {', '.join(pair.mate_moves)}")
    print(f"First image: {pair.first_image_path}")
    print(f"Final image: {pair.final_image_path}")
```

### Validation

```python
from vmevalkit.tasks.chess_task import ChessMateValidator

validator = ChessMateValidator()

# Validate a solution
result = validator.validate_solution(puzzle, "Ra8#")
print(f"Valid: {result['is_correct']}")
print(f"Legal: {result['is_legal']}")  
print(f"Checkmate: {result['is_mate']}")

# Find all mate moves in position
all_mates = validator.get_all_mate_moves("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1")
print(f"All mates: {all_mates}")
```

## File Structure

```
data/
├── chess_tasks/
│   └── chess_mate_in_1_tasks.json    # Chess dataset
└── generated_chess/
    ├── chess_0000_first.png         # Initial position images
    ├── chess_0000_final.png         # Solution position images
    ├── chess_0001_first.png
    └── chess_0001_final.png
```

## Evaluation Criteria

Video models are evaluated on their ability to:

### Primary Capabilities
1. **Spatial Understanding**: Recognize chess board structure and piece positions
2. **Pattern Recognition**: Identify mate-in-1 tactical patterns
3. **Strategic Thinking**: Find the winning move among legal alternatives  
4. **Action Demonstration**: Generate clear piece movement in video
5. **Precision**: Execute exact piece movements accurately

### Evaluation Metrics
- **Move Accuracy**: Percentage of positions where model finds a correct mate move
- **Legal Move Rate**: Percentage of positions where model plays legal moves  
- **Pattern Recognition**: Success rate by pattern type (back-rank, corner, tactical)
- **Video Quality**: Clarity and accuracy of piece movement demonstration
- **Multiple Solution Handling**: Ability to find any correct mate (30% have multiple solutions)

### Advanced Metrics
- **Difficulty Scaling**: Performance across easy/medium/hard positions
- **Color Balance**: Success with both white-to-move and black-to-move positions
- **Tactical Diversity**: Recognition across different piece types and patterns
- **Solution Completeness**: Whether full move sequence is clearly demonstrated

## Difficulty Levels

### Easy (70% of dataset)
- **Characteristics**: Basic mate patterns with clear solutions
- **Patterns**: Simple back-rank mates, basic queen/rook mates
- **Example**: `6k1/5ppp/8/8/8/8/8/R6K w - - 0 1` → Ra8#
- **Target**: Fundamental pattern recognition

### Medium (20% of dataset)  
- **Characteristics**: Tactical patterns requiring some analysis
- **Patterns**: Queen corner mates, knight forks, discovered attacks
- **Example**: Multiple solution positions, piece coordination
- **Target**: Strategic thinking and pattern analysis

### Hard (10% of dataset)
- **Characteristics**: Complex positions with subtle solutions
- **Patterns**: Advanced tactical motifs, multiple piece coordination
- **Example**: Positions requiring deep tactical understanding
- **Target**: Advanced pattern recognition and calculation

## Position Collection

The chess task system includes **213 verified mate-in-1 positions** generated through:

### Generation Methods
- **Template-Based**: Systematic variations of proven patterns
- **Position Transformations**: Mirroring and rotations of base positions
- **Combinatorial**: Systematic piece placement for mate scenarios
- **Pattern Expansion**: Variations of manually verified positions

### Quality Assurance
- **100% Verification**: Every position tested with chess engine
- **No Duplicates**: Hash-based deduplication system
- **Multiple Solutions**: Automatic detection of all possible mates
- **Balance**: Mix of white/black to move, various piece types

## Integration with VMEvalKit

### Task Pipeline
1. **Input Generation**: Chess board images from FEN positions
2. **Model Inference**: Video showing piece movement solution
3. **Solution Extraction**: Parse video to identify move played
4. **Validation**: Verify move legality and checkmate delivery
5. **Scoring**: Rate accuracy, legality, and video quality

### Expected Model Behavior
```
INPUT:  Chess board image + "White to move. Find checkmate in one move."

MODEL:  Generates video showing:
        - Initial board position (first frame)
        - Piece movement animation
        - Final board position with checkmate (last frame)

OUTPUT: Video demonstrating Ra1→Ra8 movement with clear checkmate

VALIDATION: ✅ Move is legal  ✅ Results in checkmate  ✅ Video shows movement
```

## Notes

### Technical Considerations
- Chess positions stored in standard FEN notation
- Board images rendered as SVG (convertible to PNG if needed)
- Move notation uses Standard Algebraic Notation (SAN)
- Multiple solutions supported (63 positions have multiple correct answers)
- Both white and black to move positions included

### Model Requirements
- **Visual Processing**: Ability to parse chess board images
- **Pattern Recognition**: Understanding of chess pieces and positions
- **Strategic Reasoning**: Capability to find winning moves
- **Video Generation**: Production of coherent piece movement videos
- **Precision**: Accurate representation of chess moves

### Limitations
- Currently focuses only on mate-in-1 positions (immediate solutions)
- SVG rendering may need PNG conversion for some models
- Position collection emphasizes basic patterns over advanced compositions
- Video analysis requires extraction of moves from generated content

## Research Applications

The chess reasoning task enables research into:

- **Spatial Reasoning**: How models understand 2D board representations
- **Pattern Recognition**: Detection of tactical and strategic patterns
- **Planning**: Ability to identify optimal moves in goal-directed tasks
- **Action Demonstration**: Translation of understanding into video actions
- **Domain Transfer**: Application of reasoning across different problem domains

This task provides a rigorous evaluation framework for assessing video models' reasoning capabilities in the strategic domain of chess, complementing existing spatial reasoning tasks like maze navigation.
