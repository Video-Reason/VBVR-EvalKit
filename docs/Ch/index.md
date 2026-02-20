# VMEvalKit 项目结构说明

## 顶层目录

```
VMEvalKit/
├── vmevalkit/          # 核心 Python 包（推理、评估、任务生成）
├── examples/           # 用户入口脚本
├── setup/              # 模型安装与测试脚本
├── submodules/         # 第三方 Git 子模块（开源模型仓库 & 任务数据生成器）
├── data/               # 运行时数据（问题、输出视频、评估结果、日志）
├── docs/               # 项目文档
├── script/             # 辅助 Shell 脚本（如启动评估用 VLM 服务）
├── web/                # Web 相关工具
├── sglang/             # SGLang 推理加速框架集成
├── env.template        # 环境变量模板（API Key 等）
├── pyproject.toml      # Python 项目配置与依赖声明
├── Dockerfile          # Docker 容器化配置（CUDA 11.8）
└── LICENSE             # Apache 2.0 开源协议
```

---

## `vmevalkit/` — 核心 Python 包

整个框架的主体代码，按职责分为以下子包：

### `vmevalkit/models/` — 模型推理封装

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

### `vmevalkit/runner/` — 推理调度

将模型注册、加载、批量推理和评分串联起来的调度层：
- `MODEL_CATALOG.py` — 模型注册表，纯数据（无 import），记录所有 33 个模型的 wrapper 路径、类名、家族信息等，供 `importlib` 动态加载
- `inference.py` — `run_inference()` 函数和 `InferenceRunner` 类，负责任务发现、模型加载、批量生成视频
- `score.py` — 评分调度，串联各评估器

### `vmevalkit/eval/` — 评估模块

多种评估方法的实现：
- `human_eval.py` — 基于 Gradio 的人工评分界面
- `gpt4o_eval.py` — 使用 GPT-4O 的自动化单帧评分
- `internvl.py` — 使用 InternVL（本地 VLM）评分
- `qwen3vl.py` — 使用 Qwen3-VL 评分
- `multiframe_eval.py` — 多帧评估封装，支持从视频中采样多帧后分别评分
- `frame_sampler.py` — 视频帧采样器（支持均匀采样、关键帧等混合策略）
- `consistency.py` — 多帧一致性分析
- `voting.py` — 加权投票聚合，将多帧评分汇总为最终结果
- `eval_prompt.py` — 评估用的 Prompt 模板
- `run_selector.py` — 辅助工具，用于选择待评估的推理结果

### `vmevalkit/tasks/` — 任务域实现

每个子目录对应一种视觉推理任务的题目生成逻辑：
- `chess_task/` — 国际象棋（找将杀步骤等）
- `maze_task/` — 迷宫求解
- `sudoku_task/` — 数独
- `raven_task/` — Raven 渐进矩阵推理
- `arc_agi_task/` — ARC-AGI 抽象推理
- `rotation_task/` — 旋转变换推理
- `physical_causality_task/` — 物理因果推理
- `match3/` — 三消游戏推理

### `vmevalkit/utils/` — 通用工具

- `s3_uploader.py` — S3 图片上传工具（部分商业 API 如 Luma 需要图片 URL 而非本地路径）

---

## `examples/` — 用户入口脚本

面向用户的主要运行脚本：
- `generate_videos.py` — 批量视频生成 CLI，从 questions 目录发现任务并调用指定模型生成视频
- `score_videos.py` — 评估 CLI，支持人工评分（Gradio）和自动评分（GPT-4O / InternVL / Qwen3-VL）

---

## `setup/` — 模型安装与测试

开源模型的环境搭建：
- `install_model.sh` — 模型安装入口脚本，根据 `--model` 参数调用对应的安装脚本
- `test_model.sh` — 模型安装后的验证测试
- `models/` — 每个模型一个子目录（共 33 个），各含 `setup.sh` 安装脚本，负责创建独立 venv、安装依赖、下载 checkpoint
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
- `evaluations/` — 评估结果（JSON 格式，含分数、解释、元数据）
- `data_logging/` — 运行日志

---

## `docs/` — 项目文档

- `INFERENCE.md` — 推理模块使用指南（数据格式、CLI 用法、Python API）
- `SCORING.md` — 评估模块使用指南（各评估方法的配置与使用）
- `ADDING_MODELS.md` — 新模型集成指南（Service + Wrapper 模式、注册流程）
- `MODELS.md` — 模型参考信息
- `index.md` — 本文件，项目结构总览

---

## `script/` — 辅助脚本

- `lmdeploy_server.sh` — 启动 InternVL 评估所需的 lmdeploy 推理服务（需约 30GB VRAM）

---

## `web/` — Web 工具

- `utils/` — Web 相关辅助工具

---

## `sglang/` — SGLang 集成

[SGLang](https://github.com/sgl-project/sglang) 推理加速框架，用于高效部署和运行本地 VLM 评估模型。
