# VBVR-Bench
<div align="center">

<p align="center">
    <a href="https://video-reason.com/" target="_blank">
        <img alt="Homepage" src="https://img.shields.io/badge/Project%20-%20Homepage-4285F4" height="20" />
    </a>
    <a href="https://arxiv.org/abs/2602.20159" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-VBVR_paper-red?logo=arxiv" height="20" />
    </a>
    <a href="https://huggingface.co/Video-Reason/VBVR-Wan2.2" target="_blank">
        <img alt="VBVR-Wan2.2" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Wan2.2-Models-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/Video-Reason/VBVR-Dataset" target="_blank">
        <img alt="VBVR-Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR-Dataset-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/Video-Reason/VBVR-Bench-Data" target="_blank">
        <img alt="VBVR-Bench-Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Bench-Dataset-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/spaces/Video-Reason/VBVR-Bench-Leaderboard" target="_blank">
        <img alt="Leaderboard" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Bench-Leaderboard-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://video-reason.com/" target="_blank">
        <img alt="Homepage" src="https://img.shields.io/badge/Project%20-%20Homepage-4285F4" height="20" />
    </a>
    <a href="https://arxiv.org/abs/2602.20159" target="_blank">
        <img alt="arXiv" src="https://img.shields.io/badge/arXiv-VBVR_paper-red?logo=arxiv" height="20" />
    </a>
    <a href="https://huggingface.co/Video-Reason/VBVR-Wan2.2" target="_blank">
        <img alt="VBVR-Wan2.2" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Wan2.2-Models-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/Video-Reason/VBVR-Dataset" target="_blank">
        <img alt="VBVR-Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR-Dataset-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/datasets/Video-Reason/VBVR-Bench-Data" target="_blank">
        <img alt="VBVR-Bench-Data" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Bench-Dataset-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://huggingface.co/spaces/Video-Reason/VBVR-Bench-Leaderboard" target="_blank">
        <img alt="Leaderboard" src="https://img.shields.io/badge/%F0%9F%A4%97%20_VBVR_Bench-Leaderboard-ffc107?color=ffc107&logoColor=white" height="20" />
    </a>
    <a href="https://github.com/Video-Reason/VBVR-EvalKit" target="_blank">
        <img alt="Code" src="https://img.shields.io/badge/Evaluation_code-VBVR_Bench-100000?style=flat-square&logo=github&logoColor=white" height="20" />
    </a>
    <a href="https://github.com/Video-Reason/VBVR-Wan2.2" target="_blank">
        <img alt="Code" src="https://img.shields.io/badge/Training_code-VBVR_Wan2.2-100000?style=flat-square&logo=github&logoColor=white" height="20" />
    </a>
    <a href="https://github.com/Video-Reason/VBVR-DataFactory" target="_blank">
        <img alt="Code" src="https://img.shields.io/badge/Data_code-VBVR_DataFactory-100000?style=flat-square&logo=github&logoColor=white" height="20" />
    </a>
    <a href="https://www.youtube.com/watch?v=Gs9TPZmzo-s" target="_blank">
        <img alt="Video" src="https://img.shields.io/badge/YouTube-Video-FF0000?logo=YouTube&logoColor=white" height="20" />
    </a>
    <a href="https://www.youtube.com/watch?v=Gs9TPZmzo-s" target="_blank">
        <img alt="Video" src="https://img.shields.io/badge/YouTube-Video-FF0000?logo=YouTube&logoColor=white" height="20" />
    </a>
</p>

</div>

The official evaluation repository for [Very Big Video Reasoning (VBVR)](https://video-reason.com/).
A verifiable evaluation framework that moves beyond model-based judging by incorporating rule-based, human-aligned scorers, enabling reproducible and interpretable diagnosis of video reasoning capabilities.

This toolkit supports two evaluation modes:
- **Video evaluation** — evaluate Image-to-Video model outputs (`run_evaluation.py`)
- **Image evaluation (Preview)** — evaluate interleaved image generation outputs (`run_evaluation_image_preview.py`)

Both modes share the same 100 rule-based evaluators and scoring system.

## Overview

VBVR-Bench evaluates visual reasoning models across **100 tasks** spanning 5 cognitive categories:

| Category | Definition |
|---|---|
| **Abstraction** | To find rules from observations and use rules to deduce results. |
| **Knowledge** | Propositional truth statements one could utter, either learned or gifted since birth. |
| **Perception** | Immediate access to sense datum, no further justification could be provided, i.e. "Here is one hand". |
| **Spatiality** | The intuition of the basic properties of our world, such as three-dimensionality and Euclidean-ness. |
| **Transformation** | To simulate spatial-temporal continuities with internal models in one’s mind |

Tasks are split into two subsets:
- **In-Domain (50 tasks, 250 samples)** — tasks seen in [VBVR-Dataset](https://huggingface.co/datasets/Video-Reason/VBVR-Dataset/)
- **Out-of-Domain (50 tasks, 250 samples)** — held-out tasks for generalization testing

Each task has **5 samples**, totaling **500 evaluation samples**.

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/Video-Reason/VBVR-Bench.git
cd VBVR-Bench

pip install -r requirements.txt
# Or install as a package
pip install -e .
```

### 2. Download Ground Truth Data

#### Video GT Data

> **Dataset:** [VBVR-Bench-Data](https://huggingface.co/datasets/Video-Reason/VBVR-Bench-Data)

```bash
huggingface-cli download Video-Reason/VBVR-Bench-Data --repo-type dataset --local-dir /path/to/VBVR-Bench
```

```
/path/to/VBVR-Bench/
├── VBVR-Bench.json          # Metadata: prompts, first frames (base64), GT paths
├── In-Domain_50/
│   ├── G-131_select_next_figure_.../
│   │   ├── 00000/
│   │   │   ├── first_frame.png    # Input image (first frame)
│   │   │   ├── final_frame.png    # Expected final frame
│   │   │   ├── ground_truth.mp4   # Reference video
│   │   │   └── prompt.txt         # Text prompt
│   │   ├── 00001/
│   │   └── ...                    # 5 samples per task
│   └── ...                        # 50 tasks
└── Out-of-Domain_50/
    └── ...                        # 50 tasks
```

#### Image GT Data (Preview)

> **Dataset:** [VBVR-Bench-Data-Image-Preview](https://huggingface.co/datasets/Video-Reason/VBVR-Bench-Data-Image-Preview)

```bash
huggingface-cli download Video-Reason/VBVR-Bench-Data-Image-Preview \
    --include "*.tar" --repo-type dataset --local-dir /path/to/VBVR-Bench-Image

# Extract all tar files
cd /path/to/VBVR-Bench-Image
for f in *.tar; do tar xf "$f"; done
```

```
/path/to/VBVR-Bench-Image/
├── G-131_select_next_figure_.../
│   ├── 00000/
│   │   ├── first_frame.png    # Input image
│   │   ├── output_0.png       # Ground truth output
│   │   └── meta.json          # Prompt and chain-of-thought reasoning steps
│   ├── 00001/
│   └── ...                    # 5 samples per task
└── ...                        # 100 tasks
```

### 3. Prepare Model Outputs

#### Video outputs

Generate a video for each sample using `first_frame.png` + `prompt.txt`. Organize as:

```
/path/to/model_outputs/
├── In-Domain_50/
│   ├── G-131_select_next_figure_.../
│   │   ├── 00000.mp4              # Generated video for sample 0
│   │   ├── 00001.mp4
│   │   └── ...                    # 5 videos per task
│   └── ...
└── Out-of-Domain_50/
    └── ...
```

#### Image outputs

Generate output images for each sample using `first_frame.png` + prompt (from `meta.json`). Mirror the GT structure:

```
/path/to/model_outputs/
├── G-131_select_next_figure_.../
│   ├── 00000/
│   │   ├── output_0.png           # Model's predicted output
│   │   └── ...
│   └── ...
└── ...
```

### 4. Run Evaluation

#### Video evaluation

```bash
# Single model
python run_evaluation.py \
    --model_path /path/to/model_outputs \
    --gt_base /path/to/VBVR-Bench

# Multiple models
python run_evaluation.py \
    --models_base /path/to/all_model_outputs \
    --gt_base /path/to/VBVR-Bench \
    --models model_A model_B
```

#### Image evaluation (Preview)

```bash
# Single model
python run_evaluation_image_preview.py \
    --pred_path /path/to/model_outputs \
    --gt_path /path/to/VBVR-Bench-Image

# Multiple models
python run_evaluation_image_preview.py \
    --models_base /path/to/all_model_outputs \
    --gt_path /path/to/VBVR-Bench-Image \
    --models model_A model_B
```

---

## Detailed Usage

### `run_evaluation.py` Arguments

| Argument | Description |
|---|---|
| `--model_path` | Path to a single model's video directory |
| `--models_base` | Base directory containing multiple model folders |
| `--models` | Specific model names to evaluate (with `--models_base`) |
| `--gt_base` | **(Required)** Path to the downloaded VBVR-Bench ground truth data |
| `--output_dir` | Output directory for results (default: auto-generated) |
| `--device` | `cuda` or `cpu` (default: `cuda`) |

### `run_evaluation_image_preview.py` Arguments

| Argument | Description |
|---|---|
| `--pred_path` | Path to a single model's prediction directory |
| `--models_base` | Base directory containing multiple model folders |
| `--models` | Specific model names to evaluate (with `--models_base`) |
| `--gt_path` | **(Required)** Path to image ground truth data |
| `--output_dir` | Output directory for results (default: `./evaluation_results`) |
| `--device` | `cuda` or `cpu` (default: `cuda`) |

### `evaluate.py` Arguments

| Argument | Description |
|---|---|
| `--videos_path` | **(Required)** Path to model output videos |
| `--gt_path` | **(Required)** Path to ground truth data |
| `--output_path` | Output directory (default: `./evaluation_results/`) |
| `--name` | Evaluation run name (default: auto-generated) |
| `--split` | `In-Domain_50`, `Out-of-Domain_50`, or `all` (default: `all`) |
| `--tasks` | Specific task names to evaluate |
| `--device` | `cuda` or `cpu` (default: `cuda`) |

### Supported Directory Structures

The evaluation scripts auto-detect the following model output structures:

```
# Standard structure (matches VBVR-Bench data)
model_outputs/
├── In-Domain_50/
│   └── {task_name}/
│       └── {idx}.mp4
└── Out-of-Domain_50/
    └── ...

# Checkpoints (auto-detected)
model_outputs/
├── checkpoint-100/
│   ├── In-Domain_50/
│   └── Out-of-Domain_50/
└── checkpoint-200/
    └── ...
```

---

## Output Format

Results are saved as JSON with the following structure:

```json
{
  "model_name": "my_model",
  "summary": {
    "In_Domain":     { "mean_score": 0.72, "num_samples": 250, "by_task": {...}, "by_category": {...} },
    "Out_of_Domain": { "mean_score": 0.65, "num_samples": 250, "by_task": {...}, "by_category": {...} },
    "overall":       { "mean_score": 0.68, "num_samples": 500, "by_task": {...}, "by_category": {...} }
  },
  "samples": [
    {
      "video_path": "...",
      "task_name": "G-131_select_next_...",
      "split": "In_Domain",
      "category": "Abstraction",
      "score": 0.85,
      "dimensions": { ... }
    },
    ...
  ]
}
```

---


## Repository Structure

```
VBVR-EvalKit/
├── evaluate.py                       # Main evaluation script (VBVRBench API)
├── run_evaluation.py                 # Video evaluation with auto-detection
├── run_evaluation_image_preview.py   # Image evaluation (Preview)
├── requirements.txt                  # Python dependencies
├── setup.py                          # Package installation
└── vbvr_bench/
    ├── __init__.py                   # VBVRBench class
    ├── utils.py                      # Utility functions
    └── evaluators/
        ├── __init__.py               # Evaluator registry, task metadata, and split definitions
        ├── base_evaluator.py
        └── ...                       # Task-specific evaluator modules
```

---

## Citation

```bibtex
@article{vbvr2026,
  title   = {A Very Big Video Reasoning Suite},
  author  = {Wang, Maijunxian and Wang, Ruisi and Lin, Juyi and Ji, Ran and
             Wiedemer, Thadd{\"a}us and Gao, Qingying and Luo, Dezhi and
             Qian, Yaoyao and Huang, Lianyu and Hong, Zelong and Ge, Jiahui and
             Ma, Qianli and He, Hang and Zhou, Yifan and Guo, Lingzi and
             Mei, Lantao and Li, Jiachen and Xing, Hanwen and Zhao, Tianqi and
             Yu, Fengyuan and Xiao, Weihang and Jiao, Yizheng and
             Hou, Jianheng and Zhang, Danyang and Xu, Pengcheng and
             Zhong, Boyang and Zhao, Zehong and Fang, Gaoyun and Kitaoka, John and
             Xu, Yile and Xu, Hua bureau and Blacutt, Kenton and Nguyen, Tin and
             Song, Siyuan and Sun, Haoran and Wen, Shaoyue and He, Linyang and
             Wang, Runming and Wang, Yanzhi and Yang, Mengyue and Ma, Ziqiao and
             Milli{\`e}re, Rapha{\"e}l and Shi, Freda and Vasconcelos, Nuno and
             Khashabi, Daniel and Yuille, Alan and Du, Yilun and Liu, Ziming and
             Lin, Dahua and Liu, Ziwei and Kumar, Vikash and Li, Yijiang and
             Yang, Lei and Cai, Zhongang and Deng, Hokin},
  journal = {arXiv preprint arXiv:2602.20159},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.20159}
}

```
