# Counting Pentagons Task

**Source**: Adapted from Tin's [simple_task_video_reasoning](https://github.com/tin-xai/simple_task_video_reasoning)

## Description

Visual counting task where models must count pentagons arranged in patterns.

## Task Details

- **First Frame**: Image with pentagons arranged in patterns
- **Final Frame**: Same image with total count displayed
- **Goal**: Count all pentagons accurately

## Parameters (from Tin's original)

- **Number of pentagons**: 6
- **DPI**: 100, 200, 300
- **Size**: Varies (5, 10)
- **Thickness**: 0.5, 1.0
- **Colors**: Black or colormap
- **Text position**: top, middle, bottom

## Ground Truth

Each sample includes `ground_truth_count` with the exact number of pentagons.

## Evaluation

Award 1 point if the model's count matches `ground_truth_count`, 0 otherwise.

