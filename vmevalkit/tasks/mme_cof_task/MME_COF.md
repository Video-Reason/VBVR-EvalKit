# MME-CoF: Video Chain-of-Frame Reasoning Evaluation

## Overview

**MME-CoF** (Multimodal Model Evaluation - Chain of Frames) is a benchmark for evaluating video models as zero-shot reasoners. It assesses whether video generation models can perform reliable visual reasoning through frame-by-frame progression.

**Research Question**: _Are current video models reliable zero-shot reasoners?_

## Key Findings

From the original MME-CoF study:
- Current video models (e.g., Veo-3) are **not yet** dependable standalone zero-shot reasoners
- However, they show strong potential as visual perception and scene-understanding modules
- The benchmark reveals where chain-of-frame reasoning emerges, holds, or breaks

## Dataset Structure

- **Total Tasks**: 59 reasoning tasks
- **Categories**: 12 distinct reasoning domains
- **Format**: Single images that should be animated with reasoning progression
- **Evaluation**: Generated videos are assessed for chain-of-frame reasoning quality

## Reasoning Categories

The benchmark covers 12 cognitive reasoning domains:

### 1. **2D Geometry Reasoning** 🔷
- Geometric transformations in 2D space
- Shape manipulation and spatial relationships
- Pattern transformations

### 2. **3D Geometry Reasoning** 🎲
- Three-dimensional spatial reasoning
- Depth perception and perspective
- 3D object manipulation

### 3. **Abstract Reasoning** 🧩
- Pattern recognition
- Logical rule discovery
- Conceptual relationships

### 4. **Chess** ♟️
- Strategic planning
- Tactical move sequences
- Game state progression

### 5. **Common Sense Reasoning** 💡
- Real-world knowledge application
- Cause-and-effect understanding
- Practical scenario comprehension

### 6. **Counting Reasoning** 🔢
- Quantity estimation
- Numerical changes
- Object enumeration

### 7. **Logical Reasoning** 🧠
- Formal deduction
- Inference chains
- Logical rule application

### 8. **Physics Reasoning** ⚛️
- Physical causality
- Motion and forces
- Natural phenomena simulation

### 9. **Practical Reasoning** 🔧
- Problem-solving approaches
- Step-by-step solutions
- Applied intelligence

### 10. **Visual Analogy Reasoning** 🔄
- Pattern correspondence
- Analogical transformations
- Relationship mapping

### 11. **Visual Arithmetic Reasoning** ➕
- Mathematical operations
- Visual calculations
- Quantity manipulations

### 12. **Visual Trace Reasoning** 🛤️
- Path following
- Sequential navigation
- Trajectory planning

## Chain-of-Frame Evaluation

The key innovation of MME-CoF is evaluating **chain-of-frame** reasoning:

1. **Input**: Static image representing a reasoning task
2. **Generation**: Video model creates animated sequence (typically 6 videos per image)
3. **Evaluation**: Assess whether reasoning steps are visible frame-by-frame

### What Makes Good CoF Reasoning?

- **Progressive Steps**: Reasoning should unfold gradually across frames
- **Clear Intermediate States**: Each frame should show meaningful progression
- **Logical Coherence**: Frame transitions should follow reasoning logic
- **Visual Clarity**: Reasoning steps should be visually interpretable

## Integration in VMEvalKit

### Download Dataset

```bash
python examples/create_questions.py --task mme_cof
```

This will:
- Download 59 tasks from HuggingFace (`ZiyuG/MME-CoF`)
- Create folder structure: `data/questions/mme_cof_task/mme_cof_XXXX/`
- Generate category-specific prompts for each task
- Save first frames and metadata

### Generate Videos

```bash
python examples/generate_videos.py --task mme_cof --model veo
```

### Evaluate Results

The evaluation for MME-CoF focuses on:
- **Frame-by-frame reasoning coherence**
- **Step visibility and clarity**
- **Category-appropriate reasoning patterns**
- **Progressive problem-solving demonstration**

## Prompt Strategy

Each reasoning category has specialized prompts that encourage chain-of-frame reasoning:

- Prompts explicitly request "step-by-step" animation
- Emphasize showing intermediate reasoning states
- Guide the model to display progressive transformation
- Encourage clear visual indicators of reasoning progress

## References

- **Paper**: "Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-COF Benchmark"
- **GitHub**: https://github.com/ZiyuGuo99/MME-CoF
- **Dataset**: https://huggingface.co/datasets/ZiyuG/MME-CoF
- **arXiv**: arXiv:2510.26802

## Citation

```bibtex
@article{guo2025mme-cof,
  title={Are Video Models Ready as Zero-Shot Reasoners? An Empirical Study with the MME-COF Benchmark},
  author={Guo, Ziyu and Chen, Xinyan and Zhang, Renrui and An, Ruichuan and Qi, Yu and Jiang, Dongzhi and Li, Xiangtai and Zhang, Manyuan and Li, Hongsheng and Heng, Pheng-Ann},
  journal={arXiv preprint arXiv:2510.26802},
  year={2025}
}
```

## Usage Notes

1. **No Solution Images**: Unlike other tasks, MME-CoF doesn't have ground-truth final frames
2. **Evaluation Focus**: Emphasis is on the reasoning *process* shown in video frames
3. **Multiple Generations**: Original methodology suggests generating 6 videos per image
4. **LLM Evaluation**: Original work uses Gemini-2.5-Pro for evaluation

## Task-Specific Considerations

- **Image Format**: Images should be padded to 16:9 aspect ratio (as per original methodology)
- **Video Length**: Longer videos may better demonstrate reasoning steps
- **Frame Rate**: Higher frame rates can show finer reasoning granularity
- **Evaluation Criteria**: Focus on reasoning clarity rather than visual quality alone

