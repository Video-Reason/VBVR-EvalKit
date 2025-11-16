# Text Centric Tasks

## Overview

Text Centric Tasks test mathematical reasoning and multimodal understanding combining text and visual elements. This task is sourced from the VideoThinkBench dataset on HuggingFace.

## Task Description

Text-centric tasks require models to:
- Understand text embedded in visual contexts
- Perform mathematical reasoning with visual aids
- Combine textual and visual information processing
- Solve problems requiring both reading and visual analysis

## Data Source

- **Dataset**: OpenMOSS-Team/VideoThinkBench
- **Subset**: Text_Centric_Tasks
- **Split**: test
- **Type**: HuggingFace download (not locally generated)

## Task Format

Each task pair consists of:
- **First Frame**: Problem with text and visual elements
- **Final Frame**: Solution showing the correct answer
- **Prompt**: Instructions from the dataset describing the task

## Example Task Types

- Mathematical word problems with diagrams
- Reading comprehension with visual context
- Data interpretation from charts/graphs
- Text-based logical puzzles
- Multimodal reasoning challenges

## Evaluation

Models are evaluated on their ability to:
1. Extract and understand textual information
2. Process visual context alongside text
3. Apply mathematical and logical reasoning
4. Integrate multimodal information to solve problems

## Technical Details

- **Domain**: `text_centric_tasks`
- **Module**: `vmevalkit.tasks.videothinkbench_text_centric_tasks_task`
- **Download Function**: `create_dataset()`
- **Task ID Format**: `text_centric_tasks_{id:04d}`

## References

- VideoThinkBench: https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench

