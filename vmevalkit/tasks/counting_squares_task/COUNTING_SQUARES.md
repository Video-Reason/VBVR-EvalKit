# Counting Squares Task

**Source**: Adapted from Tin's [simple_task_video_reasoning](https://github.com/tin-xai/simple_task_video_reasoning)

## Description

Visual counting task where models must count all nested squares including inner squares in recursively generated structures.

## Task Details

- **First Frame**: Image with nested squares with random offsets
- **Final Frame**: Same image with total count displayed
- **Goal**: Count ALL squares (not just outermost)

## Parameters (from Tin's original)

- **Depth**: 2-5 levels of nesting
- **Images per depth**: 10
- **Line thickness**: 2, 3, 4
- **Reduction factor**: 0.75
- **Padding**: 0.75
- **Text position**: top, middle, bottom

## Ground Truth

Each sample includes `ground_truth_count` with the exact total number of nested squares.

## Evaluation

Award 1 point if the model's count matches `ground_truth_count`, 0 otherwise.
This tests recursive/nested counting ability.

