# Mazes Task

## Overview

The Mazes task tests path-finding and navigation planning abilities. This task is sourced from the VideoThinkBench dataset on HuggingFace.

## Task Description

Maze puzzles require models to:
- Find optimal paths from start to goal
- Navigate around obstacles
- Demonstrate spatial planning
- Apply search strategies

## Data Source

- **Dataset**: OpenMOSS-Team/VideoThinkBench
- **Subset**: Mazes
- **Split**: test
- **Type**: HuggingFace download (not locally generated)

## Task Format

Each task pair consists of:
- **First Frame**: Maze with start and goal positions
- **Final Frame**: Solution showing the correct path through the maze
- **Prompt**: Instructions from the dataset describing the navigation task

## Evaluation

Models are evaluated on their ability to:
1. Identify valid paths through the maze
2. Find optimal or efficient solutions
3. Demonstrate systematic exploration strategies
4. Show the solution process step-by-step

## Technical Details

- **Domain**: `mazes`
- **Module**: `vmevalkit.tasks.videothinkbench_mazes_task`
- **Download Function**: `create_dataset()`
- **Task ID Format**: `mazes_{id:04d}`

## Comparison with Local Maze Task

VMEvalKit has both:
- **Local `maze_task`**: Procedurally generated mazes using the maze-dataset library
- **VideoThinkBench `mazes`**: Pre-created maze challenges from HuggingFace

The VideoThinkBench version provides curated maze challenges with standardized difficulty levels.

## References

- VideoThinkBench: https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench

