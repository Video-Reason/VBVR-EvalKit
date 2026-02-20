# VBVR-EvalKit Project Structure

## Top-Level Directory

```
VBVR-EvalKit/
├── vbvrevalkit/          # Core Python package (inference, evaluation, task generation)
├── examples/           # User-facing entry scripts
├── setup/              # Model installation and test scripts
├── submodules/         # Third-party Git submodules (open-source model repos & task data generators)
├── data/               # Runtime data (questions, output videos, evaluations, logs)
├── docs/               # Project documentation
├── script/             # Helper shell scripts (e.g., launching VLM servers for evaluation)
├── web/                # Web-related tools
├── sglang/             # SGLang inference acceleration framework integration
├── env.template        # Environment variable template (API keys, etc.)
├── pyproject.toml      # Python project configuration and dependency declarations
├── Dockerfile          # Docker containerization config (CUDA 11.8)
└── LICENSE             # Apache 2.0 open-source license
```

---

## `vbvrevalkit/` — Core Python Package

The main framework code, organized into the following subpackages by responsibility:

### `vbvrevalkit/models/` — Model Inference Wrappers

Each file corresponds to one model family's inference implementation, following the **Service + Wrapper** pattern:
- `base.py` — Abstract base classes `ModelWrapper` and `ModelService`, defining the unified `generate()` interface
- `luma_inference.py` — Luma Dream Machine (commercial API)
- `veo_inference.py` — Google Veo (Gemini API)
- `runway_inference.py` — Runway ML (commercial API)
- `openai_inference.py` — OpenAI Sora (commercial API)
- `kling_inference.py` — Kling AI (commercial API)
- `svd_inference.py` — Stable Video Diffusion (local open-source)
- `ltx_inference.py` / `ltx2_inference.py` — LTX-Video / LTX-2 (local open-source)
- `hunyuan_inference.py` — HunyuanVideo-I2V (local open-source)
- `cogvideox_inference.py` — CogVideoX (local open-source)
- `wan_inference.py` — Wan AI (local open-source)
- `sana_inference.py` — SANA Video (local open-source)
- `morphic_inference.py` — Morphic Frames-to-Video (local open-source)

### `vbvrevalkit/runner/` — Inference Orchestration

The orchestration layer that ties together model registration, loading, batch inference, and scoring:
- `MODEL_CATALOG.py` — Model registry, pure data (no imports), records all 33 models' wrapper paths, class names, family info, etc., for dynamic loading via `importlib`
- `inference.py` — `run_inference()` function and `InferenceRunner` class, responsible for task discovery, model loading, and batch video generation
- `score.py` — Scoring orchestration, connecting various evaluators

### `vbvrevalkit/eval/` — Evaluation Module

Implementations of multiple evaluation methods:
- `human_eval.py` — Gradio-based interactive human scoring interface
- `gpt4o_eval.py` — Automated single-frame scoring using GPT-4O
- `internvl.py` — Scoring using InternVL (local VLM)
- `qwen3vl.py` — Scoring using Qwen3-VL
- `multiframe_eval.py` — Multi-frame evaluation wrapper, supports sampling multiple frames from video for individual scoring
- `frame_sampler.py` — Video frame sampler (supports uniform sampling, keyframe, and hybrid strategies)
- `consistency.py` — Multi-frame consistency analysis
- `voting.py` — Weighted voting aggregation, combining multi-frame scores into final results
- `eval_prompt.py` — Prompt templates for evaluation
- `run_selector.py` — Utility for selecting inference results to evaluate

### `vbvrevalkit/tasks/` — Task Domain Implementations

Each subdirectory corresponds to one type of visual reasoning task's question generation logic:
- `chess_task/` — Chess (finding checkmate moves, etc.)
- `maze_task/` — Maze solving
- `sudoku_task/` — Sudoku
- `raven_task/` — Raven's Progressive Matrices reasoning
- `arc_agi_task/` — ARC-AGI abstract reasoning
- `rotation_task/` — Rotation transformation reasoning
- `physical_causality_task/` — Physical causality reasoning
- `match3/` — Match-3 game reasoning

### `vbvrevalkit/utils/` — Common Utilities

- `s3_uploader.py` — S3 image upload tool (some commercial APIs like Luma require image URLs instead of local paths)

---

## `examples/` — User Entry Scripts

Main scripts for end users:
- `generate_videos.py` — Batch video generation CLI, discovers tasks from the questions directory and invokes specified models to generate videos
- `score_videos.py` — Evaluation CLI, supports human scoring (Gradio) and automated scoring (GPT-4O / InternVL / Qwen3-VL)

---

## `setup/` — Model Installation and Testing

Environment setup for open-source models:
- `install_model.sh` — Model installation entry script, invokes the corresponding setup script based on the `--model` argument
- `test_model.sh` — Post-installation validation test
- `models/` — One subdirectory per model (33 total), each containing a `setup.sh` installation script responsible for creating an isolated venv, installing dependencies, and downloading checkpoints
- `lib/share.sh` — Shared functions for installation scripts (creating venvs, downloading checkpoints, etc.) and checkpoint path registry
- `test_assets/` — Sample test data (first_frame.png, prompt.txt, etc.)

---

## `submodules/` — Git Submodules

External repositories integrated as git submodules:
- `LTX-Video/` — LTX-Video model source code
- `HunyuanVideo-I2V/` — Tencent HunyuanVideo model source code
- `morphic-frames-to-video/` — Morphic frames-to-video model source code
- `maze-dataset/` — Maze task dataset generation tool
- `python-chess/` — Chess logic library for chess tasks

---

## `data/` — Runtime Data

Data produced and consumed during execution, organized by stage:
- `questions/` — Input data, organized as `{domain}_task/{domain}_{i:04d}/`, each question contains `first_frame.png` (required), `prompt.txt` (required), `final_frame.png` (optional), `ground_truth.mp4` (optional)
- `outputs/` — Model-generated videos, organized as `{model}/{domain}_task/{task_id}/{run_id}/`
- `evaluations/` — Evaluation results (JSON format, with scores, explanations, metadata)
- `data_logging/` — Execution logs

---

## `docs/` — Project Documentation

- `INFERENCE.md` — Inference module guide (data formats, CLI usage, Python API)
- `SCORING.md` — Evaluation module guide (configuration and usage for each evaluation method)
- `ADDING_MODELS.md` — New model integration guide (Service + Wrapper pattern, registration workflow)
- `MODELS.md` — Model reference information
- `index.md` — This file, project structure overview

---

## `script/` — Helper Scripts

- `lmdeploy_server.sh` — Launches the lmdeploy inference server required for InternVL evaluation (~30GB VRAM)

---

## `web/` — Web Tools

- `utils/` — Web-related utility tools

---

## `sglang/` — SGLang Integration

[SGLang](https://github.com/sgl-project/sglang) inference acceleration framework, used for efficient deployment and running of local VLM evaluation models.
