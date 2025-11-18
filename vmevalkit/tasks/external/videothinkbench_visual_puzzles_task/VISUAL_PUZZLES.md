# Visual Puzzles Task

## Overview

Visual Puzzles test multimodal reasoning and visual problem-solving abilities. This task is sourced from the VideoThinkBench dataset on HuggingFace.

## Task Description

Visual puzzles require models to:
- Solve diverse visual reasoning challenges
- Identify patterns in images and diagrams
- Apply logical reasoning to visual information
- Demonstrate compositional visual understanding

## Data Source

- **Dataset**: OpenMOSS-Team/VideoThinkBench
- **Subset**: Visual_Puzzles
- **Split**: test
- **Type**: HuggingFace download (not locally generated)

## Task Format

Each task pair consists of:
- **First Frame**: Initial puzzle with visual elements to analyze
- **Final Frame**: Solution showing the correct answer
- **Prompt**: Instructions from the dataset describing the puzzle

## Example Puzzle Types

- Visual analogies
- Pattern completion
- Object manipulation
- Spatial transformations
- Visual reasoning chains

## Evaluation

Models are evaluated on their ability to:
1. Understand complex visual relationships
2. Apply reasoning to solve visual problems
3. Generate coherent solution sequences

## Technical Details

- **Domain**: `visual_puzzles`
- **Module**: `vmevalkit.tasks.external.videothinkbench_visual_puzzles_task`
- **Download Function**: `create_dataset()`
- **Task ID Format**: `visual_puzzles_{id:04d}`

## References

- VideoThinkBench: https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench

