# VBVR-EvalKit 项目结构说明

## 顶层目录

```
VBVR-EvalKit/
├── vbvrevalkit/          # 核心 Python 包（推理、评估）
├── examples/           # 用户入口脚本
├── setup/              # 模型安装与测试脚本
├── submodules/         # 第三方 Git 子模块（开源模型仓库 & 任务数据生成器）
├── data/               # 运行时数据（问题、输出视频、评估结果、日志）
├── docs/               # 项目文档
├── script/             # 辅助 Shell 脚本
├── web/                # Web 相关工具
├── env.template        # 环境变量模板（API Key 等）
├── pyproject.toml      # Python 项目配置与依赖声明
├── Dockerfile          # Docker 容器化配置（CUDA 11.8）
└── LICENSE             # Apache 2.0 开源协议
```

---

## `vbvrevalkit/` — 核心 Python 包

整个框架的主体代码，按职责分为以下子包：

### `vbvrevalkit/models/` — 模型推理封装

每个文件对应一个模型家族的推理实现，遵循 **Service + Wrapper** 模式：
- `base.py` — 抽象基类 `ModelWrapper` 和 `ModelService`，定义统一的 `generate()` 接口
- `luma_inference.py` — Luma Dream Machine（商业 API）
- `veo_inference.py` — Google Veo（Gemini API）
- `runway_inference.py` — Runway ML（商业 API）
- `openai_inference.py` — OpenAI Sora（商业 API）
- `kling_inference.py` — Kling AI（商业 API）
- `svd_inference.py` — Stable Video Diffusion（本地开源）
- `ltx_inference.py` / `ltx2_inference.py` — LTX-Video / LTX-2（本地开源）
- `hunyuan_inference.py` — HunyuanVideo-I2V（本地开源）
- `cogvideox_inference.py` — CogVideoX（本地开源）
- `wan_inference.py` — Wan AI（本地开源）
- `sana_inference.py` — SANA Video（本地开源）
- `morphic_inference.py` — Morphic Frames-to-Video（本地开源）

### `vbvrevalkit/runner/` — 推理调度

将模型注册、加载、批量推理和评分串联起来的调度层：
- `MODEL_CATALOG.py` — 模型注册表，纯数据（无 import），记录所有模型的 wrapper 路径、类名、家族信息等，供 `importlib` 动态加载
- `inference.py` — `run_inference()` 函数和 `InferenceRunner` 类，负责任务发现、模型加载、批量生成视频
- `score.py` — VBVR-Bench 评分入口

### `vbvrevalkit/eval/` — 评估模块

VBVR-Bench 规则评估：
- `vbvr_bench/` — VBVR-Bench 核心评估包
  - `evaluators/` — 100+ 个任务专用评估器（如 `animal_matching.py`, `maze_pathfinding.py`, `chess_task.py`）
  - `base_evaluator.py` — 所有任务评估器的基类
  - `utils.py` — 视频帧提取、图像比较、评分工具
- `vbvr_bench_eval.py` — `VBVRBenchEvaluator` 类，遍历推理目录、匹配评估器、产出评分结果

### `vbvrevalkit/utils/` — 通用工具

- `s3_uploader.py` — S3 图片上传工具（部分商业 API 如 Luma 需要图片 URL 而非本地路径）

---

## `examples/` — 用户入口脚本

面向用户的主要运行脚本：
- `generate_videos.py` — 批量视频生成 CLI，从 questions 目录发现任务并调用指定模型生成视频
- `score_videos.py` — VBVR-Bench 评估 CLI，对生成的视频运行规则评分

---

## `setup/` — 模型安装与测试

开源模型的环境搭建：
- `install_model.sh` — 模型安装入口脚本，根据 `--model` 参数调用对应的安装脚本
- `test_model.sh` — 模型安装后的验证测试
- `models/` — 每个模型一个子目录，各含 `setup.sh` 安装脚本，负责创建独立 venv、安装依赖、下载 checkpoint
- `lib/share.sh` — 安装脚本共享函数（创建 venv、下载 checkpoint 等）和 checkpoint 路径注册表
- `test_assets/` — 测试用的样例数据（first_frame.png、prompt.txt 等）

---

## `submodules/` — Git 子模块

以 git submodule 形式集成的外部仓库：
- `LTX-Video/` — LTX-Video 模型源码
- `HunyuanVideo-I2V/` — 腾讯混元视频模型源码
- `morphic-frames-to-video/` — Morphic 帧到视频模型源码
- `maze-dataset/` — 迷宫任务数据集生成工具
- `python-chess/` — 国际象棋任务所依赖的棋盘逻辑库

---

## `data/` — 运行时数据

运行过程中产生和使用的数据，按阶段组织：
- `questions/` — 输入数据，按 `{domain}_task/{domain}_{i:04d}/` 组织，每道题包含 `first_frame.png`（必需）、`prompt.txt`（必需）、`final_frame.png`（可选）、`ground_truth.mp4`（可选）
- `outputs/` — 模型生成的视频，按 `{model}/{domain}_task/{task_id}/{run_id}/` 组织
- `evaluations/` — 评估结果（JSON 格式，含分数和元数据）
- `data_logging/` — 运行日志

---

## `docs/` — 项目文档

- `INFERENCE.md` — 推理模块使用指南（数据格式、CLI 用法、Python API）
- `SCORING.md` — 评估模块使用指南（VBVR-Bench 配置与使用）
- `ADDING_MODELS.md` — 新模型集成指南（Service + Wrapper 模式、注册流程）
- `MODELS.md` — 模型参考信息
- `data-generator.md` — 端到端流程：数据生成、推理和评估
- `index.md` — 本文件，项目结构总览

---

## `script/` — 辅助脚本

- 服务管理和其他辅助 Shell 脚本

---

## `web/` — Web 工具

- `utils/` — Web 相关辅助工具
