# VMEvalKit 评估打分

用于评估视频生成模型推理能力的综合打分方法。

## 可用的评估器

`examples/score_videos.py` 默认从配置读取 `evaluator`。命令行 `--evaluator` 仅支持 `gpt4o`、`internvl`、`qwen`。

### 人工评估
基于 Gradio 的交互式人工打分界面。

```bash
python examples/score_videos.py --eval-config eval_config.json
# 在配置中设置 "evaluator": "human"
```

### GPT-4O 评估
使用 OpenAI GPT-4O 视觉模型的自动化打分。

```bash
# 需要 OPENAI_API_KEY
python examples/score_videos.py --eval-config eval_config.json
# 在配置中设置 "evaluator": "gpt4o"
```

### InternVL 评估
开源 VLM 评估（需要 30GB 显存）。

```bash
# 启动 InternVL 服务
bash script/lmdeploy_server.sh

# 运行评估
python examples/score_videos.py --eval-config eval_config.json
# 在配置中设置 "evaluator": "internvl"
```

### Qwen3-VL 评估
使用 Qwen3-VL 的开源 VLM 评估，通过 OpenAI 兼容 API 提供服务。

```bash
# 启动 Qwen3-VL 服务（例如通过 vLLM 或 SGLang）
# 在 .env 中设置 QWEN_API_KEY 和 QWEN_API_BASE

# 运行评估
python examples/score_videos.py --eval-config eval_config.json
# 在配置中设置 "evaluator": "qwen"
```

### 多帧评估
高级评估方法，使用多个视频帧进行一致性分析和投票。

```bash
# 多帧 GPT-4O、InternVL 或 Qwen3-VL
# 在配置中设置 "evaluator" 为 "gpt4o"、"internvl" 或 "qwen"
# 并添加 "multiframe" 配置块（可选增加 "sampling_strategy"）
```

### VBVR-Bench 规则评估（Rubrics）
基于 VBVR-Bench 的 100 个任务专用评估器，无需 API 调用，完全确定性和可复现。

```bash
# 基本用法
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs

# 指定 GT 数据路径和设备
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs --gt-base-path /path/to/gt --device cuda

# 使用完整的 5 维加权评分（默认只用 task_specific 维度）
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs --full-score

# 指定输出目录
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs --eval-output-dir ./evaluations/rubrics
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--inference-dir, -i` | （必填） | 推理输出目录 |
| `--eval-output-dir, -o` | `./evaluations/rubrics` | 评估结果保存目录 |
| `--gt-base-path, -g` | 无 | VBVR-Bench GT 数据路径（可选，用于 ground_truth.mp4） |
| `--device` | `cuda` | 计算设备 (`cuda` 或 `cpu`) |
| `--full-score` | 关闭 | 启用后使用 5 维加权评分而非仅 task_specific |

## 评分标准

### VLM 评分（1-5 分制）

**1-5 分制**转换为**二分类**用于分析：
- **成功**：4-5 分（大部分/完全正确）
- **失败**：1-3 分（错误/部分正确）

### Rubrics 评分（0-1 连续分数）

VBVR-Bench 规则评估器产出 0-1 连续分数，有 5 个评估维度：
- `first_frame_consistency` (0.15)：首帧一致性
- `final_frame_accuracy` (0.35)：末帧准确性
- `temporal_smoothness` (0.15)：时间平滑性
- `visual_quality` (0.10)：视觉质量
- `task_specific` (0.25)：任务特定逻辑

默认模式 (`--full-score` 未启用) 只返回 `task_specific` 维度分数，关注任务推理正确性。

## 配置

### VLM 评估配置（eval_config.json）

创建 `eval_config.json` 来配置 VLM 评估：

```json
{
  "evaluator": "gpt4o",
  "inference_dir": "./outputs",
  "eval_output_dir": "./evaluations",
  "temperature": 0.0
}
```

多帧评估可在配置中增加 `multiframe`：

```json
{
  "evaluator": "qwen",
  "sampling_strategy": "hybrid",
  "inference_dir": "./outputs",
  "eval_output_dir": "./evaluations",
  "temperature": 0.0,
  "multiframe": {
    "n_frames": 5,
    "last_seconds": 3.0,
    "strategy": "hybrid",
    "voting": "weighted_majority",
    "metric": "histogram",
    "temporal_weight": 0.3
  }
}
```

### Rubrics 评估配置

Rubrics 评估通过命令行参数配置，不需要 `eval_config.json`：

```bash
python -m vmevalkit.runner.score rubrics \
  --inference-dir ./outputs \
  --eval-output-dir ./evaluations/rubrics \
  --gt-base-path /path/to/vbvr-bench-gt \
  --device cuda
```

## 使用方法

```bash
# VLM 评估（通过配置文件）
python examples/score_videos.py --eval-config eval_config.json

# 测试多帧管线（不调用 API）
python examples/score_videos.py --test-multiframe --video path/to/video.mp4

# Rubrics 评估（命令行直接运行）
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs
```

## 数据目录要求

### VLM 评估

推理输出只需包含 `video/` 子目录中的生成视频。

### Rubrics 评估

Rubrics 评估器需要 `question/` 子目录中的参考数据：

```
outputs/
└── {model_name}/
    └── {generator_name}/          # VBVR-Bench 任务名（如 G-3_stable_sort_data-generator）
        └── {task_type}/
            └── {task_id}/
                └── {run_id}/
                    ├── video/
                    │   └── output.mp4       # 生成的视频
                    └── question/
                        ├── first_frame.png  # 首帧参考图
                        ├── final_frame.png  # 末帧参考图
                        ├── prompt.txt       # 文本提示
                        └── ground_truth.mp4 # GT 视频（可选）
```

## 输出

### VLM 评估输出

评估结果保存到 `eval_output_dir` 目录，包含结构化 JSON 文件（含分数、元数据和解释）。结果支持断点续评和统计分析。

### Rubrics 评估输出

每个样本保存为独立的 `VBVRBenchEvaluator.json`，评估完成后自动生成汇总文件 `VBVRBenchEvaluator_summary.json`，包含：
- **全局统计**：所有模型的平均分、中位数、标准差
- **模型统计**：每个模型的得分汇总
- **按类别**：6 大类别（Abstraction, Categorization, Navigation, Perception, Physics, Transformation）的分数
- **按划分**：In_Domain / Out_of_Domain 的分数

支持断点续评——已完成的任务不会重新评估。
