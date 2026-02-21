# VBVR-EvalKit

The official evaluation toolkit for [Very Big Video Reasoning (VBVR)](https://video-reason.com/). Unified inference and evaluation across 37 video generation models.

- **37 Models**: Commercial APIs (Luma, Veo, Kling, Sora, Runway) and open-source models (LTX-Video, LTX-2, HunyuanVideo, SVD, WAN, CogVideoX, and more)
- **VBVR-Bench**: 100+ rule-based evaluators with deterministic 0–1 scores and no API calls
- **Coming Soon**: Human evaluation (Gradio) and VLM-as-a-Judge (GPT-4o, InternVL, Qwen3-VL)

## Quick Start

```bash
# Install
git clone https://github.com/Video-Reason/VBVR-EvalKit.git && cd VBVR-EvalKit
python -m venv venv && source venv/bin/activate
pip install -e .

# Setup a model
bash setup/install_model.sh --model svd --validate

# Inference
python examples/generate_videos.py --questions-dir setup/test_assets/ --output-dir ./outputs --model svd

# Evaluation (VBVR-Bench)
python examples/score_videos.py --inference-dir ./outputs
```

## Evaluation

VBVR-Bench matches each task to a rule-based evaluator by the **generator name** in the directory path. The evaluator needs both the generated video and reference data side by side:

```
{model}/{generator_name}/{task_type}/{task_id}/{run_id}/
    ├── video/output.mp4          # generated video
    └── question/                 # reference data
        ├── first_frame.png
        ├── final_frame.png
        ├── prompt.txt
        └── ground_truth.mp4     # optional
```

```bash
python examples/score_videos.py --inference-dir ./outputs           # task_specific score only
python examples/score_videos.py --inference-dir ./outputs --full-score  # all 5 dimensions
```

See [docs/En/SCORING.md](docs/En/SCORING.md) for the full end-to-end workflow, scoring dimensions, output format, and CLI reference.

## API Keys (Inference Only)

```bash
cp env.template .env
# LUMA_API_KEY=... OPENAI_API_KEY=... GEMINI_API_KEY=... KLING_API_KEY=... RUNWAYML_API_SECRET=...
```

## Docs

| Topic | Link |
|-------|------|
| Scoring (VBVR-Bench) | [docs/SCORING.md](docs/SCORING.md) |
| Inference | [docs/INFERENCE.md](docs/INFERENCE.md) |
| Supported Models | [docs/MODELS.md](docs/MODELS.md) |
| Adding Models | [docs/ADDING_MODELS.md](docs/ADDING_MODELS.md) |
| End-to-End Workflow | [docs/DATA_GENERATOR.md](docs/DATA_GENERATOR.md) |
| FAQ | [docs/FAQ.md](docs/FAQ.md) |

## Citation

```bibtex
@article{vbvr2026,
  title={A Very Big Video Reasoning Suite},
  author={Wang, Maijunxian and Wang, Ruisi and Lin, Juyi and Ji, Ran and Wiedemer, Thaddäus and Gao, Qingying and Luo, Dezhi and Qian, Yaoyao and Huang, Lianyu and Hong, Zelong and Ge, Jiahui and Ma, Qianli and He, Hang and Zhou, Yifan and Guo, Lingzi and Mei, Lantao and Li, Jiachen and Xing, Hanwen and Zhao, Tianqi and Yu, Fengyuan and Xiao, Weihang and Jiao, Yizheng and Hou, Jianheng and Zhang, Danyang and Xu, Pengcheng and Zhong, Boyang and Zhao, Zehong and Fang, Gaoyun and Kitaoka, John and Xu, Yile and Xu, Hua and Blacutt, Kenton and Nguyen, Tin and Song, Siyuan and Sun, Haoran and Wen, Shaoyue and He, Linyang and Wang, Runming and Wang, Yanzhi and Yang, Mengyue and Ma, Ziqiao and Millière, Raphaël and Shi, Freda and Vasconcelos, Nuno and Khashabi, Daniel and Yuille, Alan and Du, Yilun and Liu, Ziming and Lin, Dahua and Liu, Ziwei and Kumar, Vikash and Li, Yijiang and Yang, Lei and Cai, Zhongang and Deng, Hokin},
  year={2026}
}
```

## License

Apache 2.0

<p align="center">
  <img src="assets/logo.png" alt="VBVR Logo" width="200">
</p>
