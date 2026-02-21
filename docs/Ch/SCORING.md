# VBVR-EvalKit 评估打分

基于 VBVR-Bench 的规则评估系统，用于评估视频生成模型的推理能力。

## VBVR-Bench 评估

VBVR-Bench 提供 100+ 个任务专用评估器，产出确定性、完全可复现的 0-1 连续分数，无需任何 API 调用。

### 使用方法

```bash
# 基本评估
python examples/score_videos.py --inference-dir ./outputs

# 指定 GT 数据路径和设备
python examples/score_videos.py --inference-dir ./outputs --gt-base-path /path/to/gt --device cuda

# 使用完整的 5 维加权评分（默认只用 task_specific 维度）
python examples/score_videos.py --inference-dir ./outputs --full-score

# 指定输出目录
python examples/score_videos.py --inference-dir ./outputs --eval-output-dir ./evaluations/rubrics

# 通过 runner 模块运行
python -m vbvrevalkit.runner.score --inference-dir ./outputs
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--inference-dir, -i` | （必填） | 推理输出目录 |
| `--eval-output-dir, -o` | `./evaluations/rubrics` | 评估结果保存目录 |
| `--gt-base-path, -g` | 无 | VBVR-Bench GT 数据路径（可选，用于 ground_truth.mp4） |
| `--device` | `cuda` | 计算设备 (`cuda` 或 `cpu`) |
| `--full-score` | 关闭 | 启用后使用 5 维加权评分而非仅 task_specific |

## 评分标准

### 0-1 连续分数

VBVR-Bench 规则评估器产出 0-1 连续分数，有 5 个评估维度：

| 维度 | 权重 | 说明 |
|------|------|------|
| `first_frame_consistency` | 0.15 | 首帧与 GT 的一致性 |
| `final_frame_accuracy` | 0.35 | 末帧准确性 |
| `temporal_smoothness` | 0.15 | 帧间时间平滑性 |
| `visual_quality` | 0.10 | 视觉质量（清晰度、噪声） |
| `task_specific` | 0.25 | 任务特定推理逻辑 |

默认模式（`--full-score` 未启用）只返回 `task_specific` 维度分数，关注任务推理正确性。

## 数据目录要求

评估器需要 `question/` 子目录中的参考数据：

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

每个样本保存为独立的 `VBVRBenchEvaluator.json`，评估完成后自动生成汇总文件 `VBVRBenchEvaluator_summary.json`，包含：
- **全局统计**：所有模型的平均分、中位数、标准差
- **模型统计**：每个模型的得分汇总
- **按类别**：6 大类别（Abstraction, Categorization, Navigation, Perception, Physics, Transformation）的分数
- **按划分**：In_Domain / Out_of_Domain 的分数

支持断点续评——已完成的任务不会重新评估。
