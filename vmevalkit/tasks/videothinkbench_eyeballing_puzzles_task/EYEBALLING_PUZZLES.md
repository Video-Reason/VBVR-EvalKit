# Eyeballing Puzzles Task

## Overview

Eyeballing Puzzles test visual estimation and spatial judgment abilities. This task is sourced from the VideoThinkBench dataset on HuggingFace.

## Task Description

Eyeballing puzzles require models to:
- Make accurate visual estimations of distances, angles, and proportions
- Demonstrate spatial reasoning without precise measurements
- Identify visual relationships and alignments
- Judge geometric properties by visual inspection

## Data Source

- **Dataset**: OpenMOSS-Team/VideoThinkBench
- **Subset**: Eyeballing_Puzzles
- **Split**: test
- **Type**: HuggingFace download (not locally generated)

## Task Format

Each task pair consists of:
- **First Frame**: Initial puzzle showing geometric elements to evaluate
- **Final Frame**: Solution showing the correct answer or alignment
- **Prompt**: Instructions from the dataset describing what to eyeball

## Example Puzzle Types

- Distance comparisons
- Angle estimations
- Center point identification
- Parallelism detection
- Area comparisons

## Evaluation

Models are evaluated on their ability to:
1. Make accurate visual judgments
2. Demonstrate spatial reasoning
3. Show the solution process from problem to answer

## Technical Details

- **Domain**: `eyeballing_puzzles`
- **Module**: `vmevalkit.tasks.videothinkbench_eyeballing_puzzles_task`
- **Download Function**: `create_dataset()`
- **Task ID Format**: `eyeballing_puzzles_{id:04d}`

## References

- VideoThinkBench: https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench
- Eyeballing Game: http://woodgears.ca/eyeball/

