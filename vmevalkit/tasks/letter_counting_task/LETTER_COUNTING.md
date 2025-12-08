# Letter Counting Task

**Source**: Adapted from Tin's [simple_task_video_reasoning](https://github.com/tin-xai/simple_task_video_reasoning)

## Description

Text recognition and counting task where models must count occurrences of a specific letter within words.

## Task Details

- **First Frame**: Word displayed in large text
- **Final Frame**: Same word with red circles around target letters and count displayed
- **Goal**: Count exact number of target letter occurrences (case-insensitive)

## Parameters (from Tin's original)

- **Words**: 25 challenging words (STRAWBERRY, MISSISSIPPI, MASSACHUSETTS, etc.)
- **Letters**: All unique letters in each word
- **DPI**: 100, 150
- **Text position**: top, middle, bottom

## Ground Truth

Each sample includes:
- `ground_truth_count`: Exact number of target letter occurrences
- `word`: The displayed word
- `target_letter`: The letter to count

## Evaluation

Award 1 point if the model's count matches `ground_truth_count`, 0 otherwise.
Case-insensitive matching.

