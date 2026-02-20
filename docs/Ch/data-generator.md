# 数据生成 → 推理 → 评估 端到端流程

本文档以 `O-9_shape_scaling_data-generator`（形状缩放类比推理）为例，演示从生成问题、模型推理到 Rubrics 评估的完整流程。其他 VBVR-Bench 的 100 个 data-generator 均遵循相同模式。

## 前置条件

```bash
# 1. 安装 VMEvalKit
cd /path/to/VMEvalKit
pip install -e .

# 2. 安装至少一个视频生成模型（以 SVD 为例）
bash setup/install_model.sh --model svd --validate

# 3. 克隆 data-generator（以 O-9 为例）
git clone https://github.com/VBVR-DataFactory/O-9_shape_scaling_data-generator.git
cd O-9_shape_scaling_data-generator
pip install -r requirements.txt
```

## 完整流程

### 第一步：生成问题

使用 data-generator 生成任务样本。每个样本包含：首帧 (`first_frame.png`)、末帧 (`final_frame.png`)、提示词 (`prompt.txt`) 和参考视频 (`ground_truth.mp4`)。

```bash
cd /path/to/O-9_shape_scaling_data-generator

# 生成 1 个样本（测试用）
python examples/generate.py --num-samples 1 --seed 42 --output /path/to/VMEvalKit/questions

# 生成 100 个样本（正式评估）
python examples/generate.py --num-samples 100 --seed 42 --output /path/to/VMEvalKit/questions
```

生成结果：

```
VMEvalKit/questions/
└── shape_scaling_task/
    └── shape_scaling_00000000/
        ├── first_frame.png       # 初始状态（类比 A:B :: C:?）
        ├── final_frame.png       # 目标状态（? 的正确答案）
        ├── prompt.txt            # 任务描述
        ├── ground_truth.mp4      # 参考视频（16fps, ~3.8s）
        └── metadata.json         # 生成元数据
```

### 第二步：模型推理

使用 VMEvalKit 的推理管线生成视频。

```bash
cd /path/to/VMEvalKit

# 使用 SVD 模型生成
python examples/generate_videos.py \
  --questions-dir ./questions \
  --output-dir ./outputs \
  --model svd

# 也可以用其他模型
python examples/generate_videos.py \
  --questions-dir ./questions \
  --output-dir ./outputs \
  --model luma-ray-2
```

推理输出为扁平结构：

```
outputs/svd/shape_scaling_task/shape_scaling_00000000.mp4
```

### 第三步：组织目录结构

Rubrics 评估器要求 VMEvalKit 的运行目录结构（含 `video/` 和 `question/` 子目录）。需要将推理输出重新组织：

```bash
# 设置变量
MODEL=svd
GENERATOR=O-9_shape_scaling_data-generator
TASK_TYPE=shape_scaling_task
TASK_ID=shape_scaling_00000000
QUESTIONS_DIR=./questions/${TASK_TYPE}/${TASK_ID}

# 创建目录结构
OUTPUT_DIR=./outputs_rubrics/${MODEL}/${GENERATOR}/${TASK_TYPE}/${TASK_ID}/default
mkdir -p ${OUTPUT_DIR}/video
mkdir -p ${OUTPUT_DIR}/question

# 复制生成的视频
cp ./outputs/${MODEL}/${TASK_TYPE}/${TASK_ID}.mp4 ${OUTPUT_DIR}/video/output.mp4

# 复制问题文件（评估需要参考数据）
cp ${QUESTIONS_DIR}/first_frame.png  ${OUTPUT_DIR}/question/
cp ${QUESTIONS_DIR}/final_frame.png  ${OUTPUT_DIR}/question/
cp ${QUESTIONS_DIR}/prompt.txt       ${OUTPUT_DIR}/question/
cp ${QUESTIONS_DIR}/ground_truth.mp4 ${OUTPUT_DIR}/question/   # 可选
```

最终目录结构：

```
outputs_rubrics/
└── svd/
    └── O-9_shape_scaling_data-generator/    # generator 名（必须匹配 VBVR-Bench 任务名）
        └── shape_scaling_task/
            └── shape_scaling_00000000/
                └── default/                 # run_id（任意名称）
                    ├── video/
                    │   └── output.mp4       # 模型生成的视频
                    └── question/
                        ├── first_frame.png  # 首帧参考
                        ├── final_frame.png  # 末帧参考
                        ├── prompt.txt       # 提示词
                        └── ground_truth.mp4 # GT 视频
```

> **关键**：顶层目录名必须是 VBVR-Bench 的任务名（如 `O-9_shape_scaling_data-generator`），格式为 `{大写字母}-{数字}_{描述}_data-generator`。评估器通过此名称匹配对应的规则评估器。

### 第四步：Rubrics 评估

```bash
cd /path/to/VMEvalKit

# 运行评估
python -m vmevalkit.runner.score rubrics \
  --inference-dir ./outputs_rubrics \
  --eval-output-dir ./evaluations/rubrics \
  --device cuda
```

### 第五步：查看结果

评估完成后生成两类文件：

**单样本结果** (`VBVRBenchEvaluator.json`)：

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

**汇总文件** (`VBVRBenchEvaluator_summary.json`)：

```json
{
  "global_statistics": {
    "total_models": 1,
    "total_samples": 1,
    "mean_score": 0.8667
  },
  "models": {
    "svd": {
      "model_statistics": { "mean_score": 0.8667, "total_samples": 1 },
      "by_category": { "Transformation": { "mean_score": 0.8667 } },
      "by_split": { "Out_of_Domain": { "mean_score": 0.8667 } }
    }
  }
}
```

## 批量处理多个 Data-Generator

对于多个 data-generator，重复上述流程。目录结构支持在同一 `outputs_rubrics/` 下放置多个 generator：

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

评估命令不变，评估器会自动遍历所有 generator 并匹配对应的规则评估器：

```bash
python -m vmevalkit.runner.score rubrics --inference-dir ./outputs_rubrics
```

汇总文件会自动包含按类别（6 类）和按划分（In_Domain / Out_of_Domain）的分数统计。

## 参数参考

### data-generator 参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--num-samples` | 生成样本数 | `100` |
| `--seed` | 随机种子（可复现） | `42` |
| `--output` | 输出目录 | `./questions` |

### 推理参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--model` | 模型名称 | `svd`, `luma-ray-2` |
| `--questions-dir` | 问题目录 | `./questions` |
| `--output-dir` | 输出目录 | `./outputs` |
| `--domains` | 仅处理指定领域 | `shape_scaling_task` |

### Rubrics 评估参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--inference-dir, -i` | （必填） | 推理输出目录（需 VMEvalKit 目录结构） |
| `--eval-output-dir, -o` | `./evaluations/rubrics` | 评估结果保存目录 |
| `--gt-base-path, -g` | 无 | VBVR-Bench GT 数据路径（可选） |
| `--device` | `cuda` | 计算设备 (`cuda` / `cpu`) |
| `--full-score` | 关闭 | 使用 5 维加权评分而非仅 task_specific |
