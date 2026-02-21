# VBVR-EvalKit 评估打分

VBVR-Bench 规则评估系统，用于评估视频生成模型的推理能力。100+ 个任务专用评估器，确定性 0-1 分数，无需 API 调用。

## 工作原理

每个 VBVR-Bench 评估器通过目录路径中的 **generator 名** 进行匹配。例如 `O-9_shape_scaling_data-generator` 会匹配到形状缩放评估器。评估器读取生成的视频和参考数据，然后进行评分。

### 端到端流程

**1. 生成问题**，使用 [VBVR-DataFactory](https://github.com/VBVR-DataFactory) data-generator：
```bash
git clone https://github.com/VBVR-DataFactory/O-9_shape_scaling_data-generator.git
cd O-9_shape_scaling_data-generator && pip install -r requirements.txt
python examples/generate.py --num-samples 10 --seed 42 --output /path/to/questions
```

**2. 运行推理**，生成视频：
```bash
python examples/generate_videos.py --questions-dir ./questions --output-dir ./outputs --model svd
```

**3. 组织评估目录。** 评估器要求以下目录结构：
```
outputs_rubrics/
└── {model_name}/
    └── {generator_name}/                  # 如 O-9_shape_scaling_data-generator
        └── {task_type}/                   # 如 shape_scaling_task
            └── {task_id}/                 # 如 shape_scaling_00000000
                └── {run_id}/             # 任意名称（如 "default"）
                    ├── video/
                    │   └── output.mp4     # 模型生成的视频
                    └── question/
                        ├── first_frame.png   # 首帧参考
                        ├── final_frame.png   # 末帧参考
                        ├── prompt.txt        # 文本提示
                        └── ground_truth.mp4  # GT 视频（可选）
```

**generator 名** 目录（如 `O-9_shape_scaling_data-generator`）必须匹配 VBVR-Bench 的任务名，这样评估器才能选择正确的规则评估器。每个样本需要同时包含生成的 `video/` 和参考的 `question/` 文件。

**4. 运行评估：**
```bash
python examples/score_videos.py --inference-dir ./outputs_rubrics
```

### 批量处理

多个 generator 可以放在同一个根目录下，评估器会自动遍历：

```
outputs_rubrics/
└── svd/
    ├── G-3_stable_sort_data-generator/
    │   └── stable_sort_task/...
    ├── O-9_shape_scaling_data-generator/
    │   └── shape_scaling_task/...
    └── G-15_maze_solving_data-generator/
        └── maze_solving_task/...
```

```bash
python examples/score_videos.py --inference-dir ./outputs_rubrics
```

## 评分标准

默认只返回 `task_specific` 分数（推理正确性）。使用 `--full-score` 获取 5 个维度的加权组合分数：

| 维度 | 权重 | 衡量内容 |
|------|------|----------|
| `task_specific` | 25% | 任务特定推理逻辑 |
| `final_frame_accuracy` | 35% | 末帧是否匹配预期结果 |
| `first_frame_consistency` | 15% | 首帧是否匹配输入图像 |
| `temporal_smoothness` | 15% | 帧间过渡是否平滑 |
| `visual_quality` | 10% | 清晰度和噪声水平 |

## 命令行参考

```bash
python examples/score_videos.py --inference-dir ./outputs_rubrics                     # 基本用法
python examples/score_videos.py --inference-dir ./outputs_rubrics --full-score         # 5 维评分
python examples/score_videos.py --inference-dir ./outputs_rubrics --device cpu         # CPU 模式
python examples/score_videos.py --inference-dir ./outputs_rubrics --gt-base-path /path # 外部 GT 数据
python examples/score_videos.py --inference-dir ./outputs_rubrics -o ./my_evals        # 自定义输出目录
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-i, --inference-dir` | （必填） | 上述结构的根目录 |
| `-o, --eval-output-dir` | `./evaluations/rubrics` | 结果 JSON 的输出目录 |
| `-g, --gt-base-path` | 无 | 外部 GT 数据路径（可选） |
| `--device` | `cuda` | `cuda` 或 `cpu` |
| `--full-score` | 关闭 | 评估全部 5 个维度而非仅 `task_specific` |

也可以通过 runner 模块运行：
```bash
python -m vbvrevalkit.runner.score --inference-dir ./outputs_rubrics
```

## 输出

评估器产出两类文件：

**单样本** (`VBVRBenchEvaluator.json`)：
```json
{
  "metadata": {
    "evaluator": "VBVRBenchEvaluator",
    "model_name": "svd",
    "task_type": "O-9_shape_scaling_data-generator/shape_scaling_task",
    "task_id": "shape_scaling_00000000"
  },
  "result": {
    "score": 0.8667,
    "dimensions": { "task_specific": 0.8667 },
    "details": {
      "task_specific_details": {
        "element_preservation": 0.6667,
        "scaling_ratio": 1.0,
        "shape_type_matching": 1.0,
        "position_correctness": 1.0
      }
    },
    "evaluation_type": "rubrics",
    "vbvr_task_name": "O-9_shape_scaling_data-generator"
  }
}
```

**汇总** (`VBVRBenchEvaluator_summary.json`)：
```json
{
  "global_statistics": {
    "total_models": 1,
    "total_samples": 10,
    "mean_score": 0.8667
  },
  "models": {
    "svd": {
      "model_statistics": { "mean_score": 0.8667, "total_samples": 10 },
      "by_category": { "Transformation": { "mean_score": 0.8667 } },
      "by_split": { "Out_of_Domain": { "mean_score": 0.8667 } }
    }
  }
}
```

统计维度包括：
- **模型**：每个模型的平均分、中位数、标准差
- **类别**：6 大类别（Abstraction, Categorization, Navigation, Perception, Physics, Transformation）
- **划分**：In_Domain / Out_of_Domain（各 50 个任务）

评估支持**断点续评** — 重新运行相同命令会自动跳过已评估的任务。
