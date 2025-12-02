# Edit Distance Task

## Overview

The **Edit Distance Task** is a visual calculation benchmark that tests video generation models' ability to perform string analysis and display numerical results. Models must calculate the Levenshtein (edit) distance between two strings and animate the answer into a designated box.

## Task Description

**Input Frame**: A static image containing:
- **String_A**: Displayed at the top of the frame
- **String_B**: Displayed at the bottom of the frame
- **Empty Answer Box**: Located on the right side

**Required Computation**: The model must:
1. **Read**: Identify and extract String_A and String_B from the image
2. **Calculate**: Compute the Levenshtein distance (minimum number of single-character edits needed to transform String_A into String_B)
3. **Generate**: Create a video where the calculated number appears in the answer box

**Output Video**: A short video where the only change from the initial frame is the appearance of the correct edit distance number inside the answer box.

## Example

**Input Frame:**
- String_A: `KITTEN` (top)
- String_B: `SITTING` (bottom)
- Answer Box: Empty

**Internal Calculation:**
The edit distance between "KITTEN" and "SITTING" is **3**:
- KITTEN → SITTEN (substitute K with S)
- SITTEN → SITTIN (substitute E with I)
- SITTIN → SITTING (insert G at end)

**Output Video:**
A video showing the number **3** appearing in the answer box.

## Difficulty Levels

The task includes progressive difficulty based on string characteristics:

### Easy
- **String Length**: 3-6 characters
- **Edit Distance**: 1-2 edits
- **Examples**: 
  - CAT → BAT (distance: 1)
  - DOG → DIG (distance: 1)
  - SUN → RUN (distance: 1)

### Medium
- **String Length**: 6-10 characters
- **Edit Distance**: 3-5 edits
- **Examples**:
  - KITTEN → SITTING (distance: 3)
  - HORSE → HOUSE (distance: 2)
  - WATER → LATER (distance: 1)

### Hard
- **String Length**: 10-15 characters
- **Edit Distance**: 5+ edits
- **Examples**:
  - ALGORITHM → LOGARITHM (distance: 4)
  - ELEPHANT → RELEVANT (distance: 5)
  - COMPUTER → COMPILER (distance: 3)

## String Generation

The task uses a **mix of two approaches**:

1. **Real English Words** (50% of tasks): Uses curated word lists with semantically related pairs (e.g., KITTEN/SITTING, HORSE/HOUSE)

2. **Random Character Strings** (50% of tasks): Generates random uppercase alphabetic strings with controlled edit distances

This combination ensures diversity while maintaining realistic visual computation challenges.

## Reasoning Type

This task evaluates:
- **Text Recognition**: OCR-like capability to read strings from images
- **Algorithmic Reasoning**: Understanding of edit distance computation
- **Numerical Output**: Ability to generate and display numerical results
- **Temporal Reasoning**: Creating appropriate video sequences showing the answer appearing

## Data Structure

Each task pair contains:

```json
{
    "id": "edit_distance_0000",
    "prompt": "Calculate the edit distance between the two strings shown and display the number in the answer box.",
    "first_image_path": "path/to/first_frame.png",
    "final_image_path": "path/to/final_frame.png",
    "domain": "edit_distance",
    "task_category": "EditDistance",
    "difficulty": "easy",
    "edit_distance_data": {
        "string_a": "KITTEN",
        "string_b": "SITTING",
        "distance": 3,
        "is_word_pair": true
    },
    "created_at": "2025-01-15T10:30:00.123456"
}
```

## Usage

### Generate Dataset

```python
from vmevalkit.tasks.edit_distance_task import create_dataset

# Generate 15 tasks (default)
dataset = create_dataset()

# Generate custom number of tasks
dataset = create_dataset(num_samples=50)
```

### Command Line

```bash
# Generate dataset with edit distance tasks
python -m vmevalkit.runner.create_dataset --pairs-per-domain 15

# Check generated files
ls data/questions/edit_distance_task/edit_distance_0000/
# Output: first_frame.png  final_frame.png  prompt.txt  question_metadata.json
```

## Visual Design

The task uses a **simple, clear layout**:

- **Background**: White
- **String_A**: Top of frame, blue background box, monospace font
- **String_B**: Bottom of frame, green background box, monospace font
- **Answer Box**: Right side, large outlined box (white when empty, yellow when filled)
- **Font**: Monospace for clarity
- **Size**: 6x6 inches at 150 DPI (~900x900 pixels)

This design prioritizes:
- Clear readability of strings
- Obvious answer location
- Minimal visual distractions
- High contrast for easy OCR

## Evaluation Criteria

Models are scored on **solution correctness** (1-5 scale):

- **5 (Perfect)**: Exact correct edit distance displayed
- **4 (Mostly Correct)**: Very close (off by 1)
- **3 (Partially Correct)**: Reasonable attempt but incorrect
- **2 (Mostly Wrong)**: Shows number but far from correct
- **1 (Completely Wrong)**: No number, wrong format, or completely incorrect

**Binary Grading**: Scores 4-5 are considered "correct" for analysis.

## Implementation Details

### Levenshtein Distance Algorithm

Uses dynamic programming approach:

```python
def calculate_levenshtein_distance(str1: str, str2: str) -> int:
    """
    Calculate edit distance using dynamic programming.
    O(n*m) time and space complexity.
    """
    len1, len2 = len(str1), len(str2)
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[len1][len2]
```

### Image Rendering

- Uses `matplotlib` for rendering
- Employs `tempfile` for temporary storage
- Consistent with other VMEvalKit tasks
- PNG format, high quality

## Related Tasks

This task complements other VMEvalKit reasoning tasks:

- **Chess**: Strategic planning and move execution
- **Maze**: Spatial navigation
- **Sudoku**: Logical constraint satisfaction
- **RAVEN**: Pattern recognition
- **Rotation**: Spatial transformation

The Edit Distance task specifically focuses on **string manipulation reasoning** and **numerical computation**, filling a gap in text-based algorithmic challenges.

## References

- Levenshtein, V. I. (1966). "Binary codes capable of correcting deletions, insertions, and reversals." Soviet Physics Doklady.
- Standard dynamic programming algorithm for edit distance computation
- Widely used in spell checking, DNA sequence alignment, and natural language processing

## Task Statistics

Default dataset (15 samples):
- Easy tasks: ~5
- Medium tasks: ~5
- Hard tasks: ~5
- Word pairs: ~7-8
- Random strings: ~7-8

Generation time: ~1-2 seconds for 15 tasks

---

**Created**: 2025-01-15  
**Task Type**: Locally Generated  
**Category**: String Reasoning, Numerical Computation  
**VMEvalKit Version**: 1.0+

