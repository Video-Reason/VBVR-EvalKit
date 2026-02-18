# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.



Apply KISS, YAGNI, and SOLID principles.


Don't delete anything in `docs/`.


don't write redundant code in markdown, keep it concise.


每次完成任务后，`git add` 改动的文件并 `git commit`。 不要提交有名字的绝对路径, 改成相对路径. 


## Project Overview

VMEvalKit is a unified inference and evaluation framework for 29+ video generation models. It evaluates video models' reasoning capabilities through task pairs — visual reasoning problems (chess, maze, sudoku, raven puzzles, arc-agi, rotation, physical causality, match3). Models receive an initial state image + text prompt and must generate videos demonstrating the reasoning process.




## Common Commands

```bash
# Install (editable mode)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Install an open-source model (creates isolated venv in ./envs/{model-id}/)
bash setup/install_model.sh --model svd --validate

# Run inference
python examples/generate_videos.py --questions-dir ./questions --output-dir ./outputs --model svd

# Run evaluation (configure eval_config.json first)
python examples/score_videos.py --eval-config eval_config.json

# Linting and formatting
black vmevalkit/
isort vmevalkit/
flake8 vmevalkit/
mypy vmevalkit/

# Tests
pytest
pytest tests/test_specific.py -k "test_name"
```

## Architecture

### Service + Wrapper Pattern

Every model integration follows a two-class pattern:
- **`ModelService`** (optional ABC in `vmevalkit/models/base.py`): handles actual API calls or local inference
- **`ModelWrapper`** (required ABC in `vmevalkit/models/base.py`): inherits from `ModelWrapper`, implements `generate(image_path, text_prompt, **kwargs)` returning a standardized 8-field dict: `{success, video_path, error, duration_seconds, generation_id, model, status, metadata}`

### Dynamic Model Registry

`vmevalkit/runner/MODEL_CATALOG.py` is a pure-data registry — no imports, just string paths. Models are loaded on-demand via `importlib` in `vmevalkit/runner/inference.py`. Each entry specifies `wrapper_module`, `wrapper_class`, `service_class`, `model`, and `family`. To add a model: create an inference file in `vmevalkit/models/`, then register it in `MODEL_CATALOG.py`.

### Lazy Loading

`vmevalkit/__init__.py` and `vmevalkit/eval/__init__.py` use `__getattr__()` for lazy imports to avoid loading heavy dependencies (torch, transformers, etc.) at startup.

### Key Entry Points

- `examples/generate_videos.py` — CLI for batch video generation across task directories
- `examples/score_videos.py` — CLI for evaluation (human via Gradio, automated via GPT-4O/InternVL/Qwen3-VL)
- `vmevalkit.runner.inference.InferenceRunner` — Python API for programmatic use

### Data Flow

```
questions/{domain}_task/{domain}_{i:04d}/{first_frame.png, prompt.txt}
    → model inference →
outputs/{model}/{domain}_task/{task_id}/{run_id}/video/generated_video.mp4
    → evaluation →
evaluations/ (structured JSON with scores, metadata, explanations)
```

### Package Layout

- `vmevalkit/models/` — 17 model wrapper implementations (one file per provider/model family)
- `vmevalkit/runner/` — `MODEL_CATALOG.py` (registry), `inference.py` (orchestration), `score.py` (scoring)
- `vmevalkit/eval/` — evaluators (human, gpt4o, internvl, qwen3vl, multiframe, vbvr_bench_eval), frame sampling, consistency analysis, voting aggregation
- `vmevalkit/tasks/` — 8 task domain implementations (chess, maze, sudoku, raven, rotation, arc_agi, physical_causality, match3)
- `setup/` — model installation; `setup/models/{name}/requirements.txt` for each model's dependencies
- `submodules/` — git submodules for open-source model repos and task data generators

## Conventions

- Model names use kebab-case: `luma-ray-2`, `ltx-video`, `wan-2.1-i2v-480p`
- Task IDs: `{domain}_{i:04d}` with auto-expanding padding (e.g., `chess_0000`, `chess_10000`)
- API keys loaded from `.env` via python-dotenv (see `env.template` for required keys)
- Open-source models get isolated venvs at `./envs/{model-id}/`; `ModelWrapper.get_model_python_interpreter()` resolves the correct Python binary
- Model checkpoints stored in `weights/` directory
- VLM scoring uses a 1-5 scale, converted to binary: 4-5 = success, 1-3 = failure
- Rubrics scoring (VBVR-Bench) uses 0-1 continuous scale, rule-based with no API calls
