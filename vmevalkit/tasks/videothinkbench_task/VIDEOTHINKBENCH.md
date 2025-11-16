# VideoThinkBench Meta-Task

## Overview

VideoThinkBench is a comprehensive benchmark for evaluating video models on reasoning tasks. This meta-task downloads all VideoThinkBench subsets in a single operation.

## Task Description

VideoThinkBench combines multiple reasoning domains:
- **ARC AGI 2**: Abstract reasoning and pattern recognition
- **Eyeballing Puzzles**: Visual estimation and spatial judgment
- **Visual Puzzles**: Multimodal reasoning challenges
- **Mazes**: Path-finding and navigation
- **Text Centric Tasks**: Mathematical and textual reasoning

## Data Source

- **Dataset**: OpenMOSS-Team/VideoThinkBench
- **Subsets**: All 5 subsets (ARC_AGI_2, Eyeballing_Puzzles, Visual_Puzzles, Mazes, Text_Centric_Tasks)
- **Split**: test
- **Type**: HuggingFace meta-download
- **Total Tasks**: ~4,100 reasoning tasks

## Usage

This is a convenience task for downloading all VideoThinkBench subsets at once:

```python
from vmevalkit.tasks.videothinkbench_task import create_dataset

# Download all VideoThinkBench tasks
dataset = create_dataset()
# Returns combined dataset with all subsets
```

Alternatively, you can download individual subsets:
- `vmevalkit.tasks.videothinkbench_arc_agi_task`
- `vmevalkit.tasks.videothinkbench_eyeballing_puzzles_task`
- `vmevalkit.tasks.videothinkbench_visual_puzzles_task`
- `vmevalkit.tasks.videothinkbench_mazes_task`
- `vmevalkit.tasks.videothinkbench_text_centric_tasks_task`

## Task Format

Each task pair consists of:
- **First Frame**: Initial problem state
- **Final Frame**: Solution state
- **Prompt**: Instructions describing the task
- **Domain**: Specific subset identifier

## Dataset Statistics

The complete VideoThinkBench includes:
- Diverse reasoning capabilities
- Multiple difficulty levels
- Curated quality benchmarks
- Standardized evaluation format

## Technical Details

- **Domain**: `videothinkbench`
- **Module**: `vmevalkit.tasks.videothinkbench_task`
- **Download Function**: `create_dataset()`
- **Special Flag**: `hf_meta: True` (downloads all subsets)
- **Task ID Format**: `{subset_name}_{id:04d}`

## Evaluation

Models are evaluated across all reasoning domains, providing comprehensive assessment of:
1. Abstract reasoning
2. Visual understanding
3. Spatial cognition
4. Logical deduction
5. Mathematical reasoning

## References

- VideoThinkBench Paper: [Link to paper if available]
- HuggingFace Dataset: https://huggingface.co/datasets/OpenMOSS-Team/VideoThinkBench
- VMEvalKit Documentation: See individual subset documentation

## See Also

- [ARC_AGI.md](../videothinkbench_arc_agi_task/ARC_AGI.md)
- [EYEBALLING_PUZZLES.md](../videothinkbench_eyeballing_puzzles_task/EYEBALLING_PUZZLES.md)
- [VISUAL_PUZZLES.md](../videothinkbench_visual_puzzles_task/VISUAL_PUZZLES.md)
- [MAZES.md](../videothinkbench_mazes_task/MAZES.md)
- [TEXT_CENTRIC_TASKS.md](../videothinkbench_text_centric_tasks_task/TEXT_CENTRIC_TASKS.md)

