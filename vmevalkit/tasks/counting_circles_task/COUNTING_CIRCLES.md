# Counting Circles Task

**Source**: Adapted from Tin's [simple_task_video_reasoning](https://github.com/tin-xai/simple_task_video_reasoning)

## Description

Visual counting task where models must count circles arranged in Olympic-logo-like patterns.

## Task Details

- **First Frame**: Image with circles arranged in patterns
- **Final Frame**: Same image with total count displayed
- **Goal**: Count all circles accurately

## Parameters (from Tin's original)

- **Number of circles**: 5-9
- **DPI**: 100, 200, 300
- **Radius**: Varies (5, 10)
- **Thickness**: 0.5, 1.0
- **Colors**: Black or colormap
- **Text position**: top, middle, bottom

## Ground Truth

Each sample includes `ground_truth_count` with the exact number of circles.

## Evaluation

Award 1 point if the model's count matches `ground_truth_count`, 0 otherwise.

