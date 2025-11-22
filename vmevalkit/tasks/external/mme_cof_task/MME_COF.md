# MME-CoF Task

## Overview

MME-CoF (Multimodal Evaluation - Chain of Frames) is a comprehensive benchmark for evaluating video models on chain-of-frame reasoning across diverse cognitive domains.

## Task Description

MME-CoF tests models across **16 cognitive domains** with **59 distinct tasks**, requiring:
- Frame-by-frame reasoning
- Temporal understanding
- Multi-step problem solving
- Diverse cognitive capabilities

## Data Source

- **Dataset**: VideoReason/MME-CoF-VMEval
- **Split**: train
- **Type**: HuggingFace download (not locally generated)
- **Total Tasks**: 59 curated reasoning tasks

## Cognitive Domains

MME-CoF covers a wide range of reasoning capabilities including:

1. **Visual Reasoning**: Pattern recognition, visual analogies
2. **Spatial Reasoning**: 3D understanding, spatial transformations
3. **Temporal Reasoning**: Event sequences, causal chains
4. **Logical Reasoning**: Deduction, inference
5. **Mathematical Reasoning**: Arithmetic, geometry
6. **Physical Reasoning**: Physics simulation, dynamics
7. **Causal Reasoning**: Cause-effect relationships
8. **Planning**: Multi-step goal achievement
9. **And 8 more domains...**

## Task Format

Each task pair consists of:
- **First Frame**: Initial problem state
- **Final Frame**: Solution or target state
- **Prompt**: Instructions describing the chain-of-frame reasoning task

## Chain-of-Frame Evaluation

Unlike single-frame evaluation, MME-CoF focuses on:
- **Process**: How the model reasons through steps
- **Transitions**: Intermediate frame generation quality
- **Coherence**: Logical progression from start to finish
- **Completeness**: Whether the full reasoning chain is captured

## Technical Details

- **Domain**: `mme_cof`
- **Module**: `vmevalkit.tasks.external.mme_cof_task`
- **Download Function**: `create_dataset()`
- **Task ID Format**: `mme_cof_{id:04d}`

## Evaluation Metrics

Models can be evaluated on:
1. **Correctness**: Does the final frame match the solution?
2. **Reasoning Path**: Are intermediate steps logical?
3. **Visual Quality**: Are generated frames clear and coherent?
4. **Temporal Consistency**: Do frames flow smoothly?

## Usage

```python
from vmevalkit.tasks.external.mme_cof_task import create_dataset

# Download MME-CoF tasks
dataset = create_dataset()
# Returns dataset with 59 reasoning tasks across 16 domains
```

## Comparison with Other Benchmarks

| Benchmark | Domains | Tasks | Focus |
|-----------|---------|-------|-------|
| MME-CoF | 16 | 59 | Chain-of-frame reasoning |
| VideoThinkBench | 5 | ~4,100 | Volume and diversity |
| VMEval (Core) | 6 | Custom | Specific reasoning types |

MME-CoF emphasizes **reasoning process** over volume, with carefully curated tasks testing explicit cognitive capabilities.

## References

- MME-CoF Paper: [Link if available]
- HuggingFace Dataset: https://huggingface.co/datasets/VideoReason/MME-CoF-VMEval
- VideoReason Project: [Project page if available]

## Citation

If you use MME-CoF in your research, please cite:

```bibtex
@article{guo2025mme-cof,
  title={Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-COF Benchmark},
  author={Guo, Ziyu and Chen, Xinyan and Zhang, Renrui and An, Ruichuan and Qi, Yu and Jiang, Dongzhi and Li, Xiangtai and Zhang, Manyuan and Li, Hongsheng and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2510.26802},
  year={2025}
}
```

