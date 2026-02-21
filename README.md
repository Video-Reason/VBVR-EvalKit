<div align="center">

<img src="assets/logo.png" alt="VBVR Logo" width="160">

# VBVR-EvalKit

**The official evaluation toolkit for [Very Big Video Reasoning (VBVR)](https://video-reason.com/)**

Unified inference and evaluation across 37 video generation models.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Models](https://img.shields.io/badge/Models-37-green.svg)](docs/MODELS.md)
[![Evaluators](https://img.shields.io/badge/Evaluators-100%2B-orange.svg)](docs/SCORING.md)

</div>

---

<table>
<tr>
<td width="33%" valign="top">

### Commercial APIs
Luma · Veo · Kling · Sora · Runway
<br><sub>19 models, instant setup</sub>

</td>
<td width="33%" valign="top">

### Open-Source
LTX-Video · LTX-2 · HunyuanVideo · SVD · WAN · CogVideoX
<br><sub>18 models, local GPU</sub>

</td>
<td width="33%" valign="top">

### VBVR-Bench
100+ rule-based evaluators
<br><sub>Deterministic 0-1 scores, no API calls</sub>

</td>
</tr>
</table>

> **Coming Soon** — Human evaluation (Gradio) and VLM-as-a-Judge (GPT-4O, InternVL, Qwen3-VL)

---

## Quick Start

```bash
git clone https://github.com/Video-Reason/VBVR-EvalKit.git && cd VBVR-EvalKit
python -m venv venv && source venv/bin/activate
pip install -e .
```

**Generate videos** with any supported model:
```bash
bash setup/install_model.sh --model svd --validate
python examples/generate_videos.py --questions-dir setup/test_assets/ --output-dir ./outputs --model svd
```

**Evaluate** with VBVR-Bench:
```bash
python examples/score_videos.py --inference-dir ./outputs
python examples/score_videos.py --inference-dir ./outputs --full-score   # all 5 dimensions
```

---

## Evaluation

VBVR-Bench matches each task to a rule-based evaluator by the **generator name** in the directory path. The evaluator reads both the generated video and reference data:

```
{model}/{generator_name}/{task_type}/{task_id}/{run_id}/
    ├── video/output.mp4          # generated video
    └── question/                 # reference data
        ├── first_frame.png
        ├── final_frame.png
        ├── prompt.txt
        └── ground_truth.mp4     # optional
```

<details>
<summary><b>Scoring dimensions</b></summary>
<br>

| Dimension | Weight | Description |
|:--|:--:|:--|
| `task_specific` | 25 % | Task-specific reasoning logic |
| `final_frame_accuracy` | 35 % | Does the final frame match the expected result? |
| `first_frame_consistency` | 15 % | Does the first frame match the input image? |
| `temporal_smoothness` | 15 % | Are frame transitions smooth? |
| `visual_quality` | 10 % | Sharpness and noise levels |

</details>

See [docs/SCORING.md](docs/SCORING.md) for the full end-to-end workflow, output format, and CLI reference.

---

## API Keys

Only needed for commercial model inference.

```bash
cp env.template .env
```

```env
LUMA_API_KEY=...
OPENAI_API_KEY=...
GEMINI_API_KEY=...
KLING_API_KEY=...
RUNWAYML_API_SECRET=...
```

---

## Documentation

| | Topic | |
|:--|:--|:--|
| **Eval** | [Scoring (VBVR-Bench)](docs/SCORING.md) | Evaluators, dimensions, output format |
| **Run** | [Inference](docs/INFERENCE.md) | Running models end-to-end |
| **Models** | [Supported Models](docs/MODELS.md) | All 37 models with setup commands |
| **Extend** | [Adding Models](docs/ADDING_MODELS.md) | Integrate your own model |
| **Data** | [End-to-End Workflow](docs/DATA_GENERATOR.md) | Question generation pipeline |
| **Help** | [FAQ](docs/FAQ.md) | Common issues and solutions |
| **Map** | [Project Structure](docs/INDEX.md) | Codebase layout |

---

<div align="center">
<sub>Apache 2.0 · <a href="https://video-reason.com/">video-reason.com</a></sub>
</div>
