# ARC AGI 2 Task

## Overview

The ARC AGI 2 (Abstraction and Reasoning Corpus) task tests abstract reasoning and pattern recognition abilities. This task is sourced from the VideoThinkBench dataset on HuggingFace.

## Task Description

ARC AGI challenges require models to:
- Identify abstract patterns in grid-based puzzles
- Apply learned transformations to new examples
- Demonstrate compositional reasoning
- Handle novel problem types without explicit training

## Data Source

- **Dataset**: OpenMOSS-Team/VideoThinkBench
- **Subset**: ARC_AGI_2
- **Split**: test
- **Type**: HuggingFace download (not locally generated)

## Task Format

Each task pair consists of:
- **First Frame**: Initial puzzle state with input grid
- **Final Frame**: Solution state showing the correct output
- **Prompt**: Instructions from the dataset describing the task

## Evaluation

Models are evaluated on their ability to:
1. Understand the abstract pattern or rule
2. Apply the transformation correctly
3. Generate a video showing the reasoning process from input to output

## Technical Details

- **Domain**: `arc_agi_2`
- **Module**: `vmevalkit.tasks.external.videothinkbench_arc_agi_task`
- **Download Function**: `create_dataset()`
- **Task ID Format**: `arc_agi_2_{id:04d}`

## References

- VideoThinkBench: https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench
- ARC Challenge: https://github.com/fchollet/ARC

