# VMEvalKit 推理模块

## 快速开始

```bash
# 1. 准备问题目录（仅需 first_frame.png 和 prompt.txt）
# questions/chess_task/chess_0000/{first_frame.png, prompt.txt}

# 2. 生成视频（自动发现并运行所有任务）
python examples/generate_videos.py --questions-dir ./questions --output-dir ./outputs --model svd

# 3. 使用指定模型运行
python examples/generate_videos.py --questions-dir ./questions --output-dir ./outputs --model luma-ray-2
```

## 核心概念

### 任务对：评估单元

VMEvalKit 通过**任务对**来评估视频模型的推理能力——这些是精心设计的视觉推理问题：

| 组件 | 文件 | 用途 | 是否必需 |
|------|------|------|----------|
| **初始状态** | `first_frame.png` | 待解决的问题/谜题 | 必需 |
| **文本提示** | `prompt.txt` | 自然语言指令 | 必需 |
| **最终状态** | `final_frame.png` | 解答/目标参考 | 可选 |
| **参考视频** | `ground_truth.mp4` | 参考视频 | 可选 |

**目录结构：**
```
questions/
├── chess_task/
│   ├── chess_0000/
│   │   ├── first_frame.png      # 初始状态（必需）
│   │   ├── prompt.txt           # 指令（必需）
│   │   ├── final_frame.png      # 目标状态（可选）
│   │   └── ground_truth.mp4     # 参考视频（可选）
│   └── chess_0001/...
├── maze_task/...
└── sudoku_task/...
```

模型接收初始状态和文本提示，需要生成展示推理过程的视频以达到最终状态。

## 架构

VMEvalKit 使用**模块化架构**和动态加载机制：

- **MODEL_CATALOG**：包含 37 个模型（覆盖 15 个家族）的注册表
- **动态加载**：通过 importlib 按需加载模型
- **统一接口**：所有模型继承自 `ModelWrapper`
- **两类模型**：
  - **商业 API**：仅需 API Key 即可使用（Luma、Veo、Kling、Sora、Runway）
  - **开源模型**：需要本地安装（LTX-Video、LTX-2、HunyuanVideo、DynamiCrafter、SVD）

## 输出结构

输出按层级组织：`模型/领域任务/任务ID/运行ID/`

```
outputs/
├── luma-ray-2/
│   └── chess_task/
│       └── chess_0000/
│           └── luma-ray-2_chess_0000_20250103_143025/
│               ├── video/generated_video.mp4
│               ├── question/{first_frame.png, prompt.txt, final_frame.png}
│               └── metadata.json  # 自动生成：运行信息、模型、耗时、状态
```



## Python API

```python
from vmevalkit.runner.inference import InferenceRunner

runner = InferenceRunner(output_dir="./outputs")
result = runner.run(
    model_name="luma-ray-2",
    image_path="questions/chess_task/chess_0000/first_frame.png",
    text_prompt="Find the checkmate move"
)
print(f"已生成: {result['video_path']}")
```


## 配置

### API Keys
```bash
cp env.template .env
# 编辑 .env 填写你的 API Key：
LUMA_API_KEY=your_key_here
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
KLING_API_KEY=your_kling_key
RUNWAYML_API_SECRET=your_runway_secret
```
