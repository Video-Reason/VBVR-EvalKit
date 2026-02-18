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

- **MODEL_CATALOG**：包含 33 个模型（覆盖 13 个家族）的注册表
- **动态加载**：通过 importlib 按需加载模型
- **统一接口**：所有模型继承自 `ModelWrapper`
- **两类模型**：
  - **商业 API**：仅需 API Key 即可使用（Luma、Veo、Kling、Sora、Runway）
  - **开源模型**：需要本地安装（LTX-Video、LTX-2、HunyuanVideo、SVD）

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


# Open-Source Model Test Results

Task: `shape_scaling_00000000` (1 task)

Env: 
RTX A6000 

lmms-eval 是用一个 transformers固定版本. 


## Tested & Succeeded


| Model | Time | Notes |
|---|---|---|
| svd | - | 之前已有结果 |
| ltx-video | 1m31s | |
| ltx-video-13b-distilled | 5m38s | |
| wan-2.2-ti2v-5b | 9m19s | |
| sana-video-2b-480p | ~12s | 修复后成功 |
| cogvideox1.5-5b-i2v | 5m22s | 重装 venv + 强制使用模型原生分辨率/帧数 |

## TODO (需要重跑)

| Model | 原因 |
|---|---|
| wan-2.2-i2v-a14b |  |

## Failed - 环境/依赖问题

| Model | 错误 | 修复方法 |
|---|---|---|
| cogvideox-5b-i2v | venv 缺 torch | 重装 venv: `bash setup/install_model.sh --model cogvideox-5b-i2v` |

## Failed - 缺少权重/安装

| Model | 错误 | 修复方法 |
|---|---|---|
| LTX-2 | `RuntimeError: LTX-2 is not installed` | `bash setup/install_model.sh --model LTX-2`。LTX-2 使用自有 pipeline（非 diffusers），checkpoint: `ltx-2-19b-distilled-fp8.safetensors`，文本编码器: Gemma-3 12B（`gemma3-12b-it-qat-q4_0-unquantized`），默认分辨率 512x768，97 帧 @25fps，需要 ~40GB 显存。**注意:** Gemma-3 12B 文本编码器约 24GB，500Mbps 带宽下载约需 10 分钟。setup 脚本通过检查 `.safetensors` 权重文件判断是否已下载完成，不会因为目录非空而跳过 |
| hunyuan-video-i2v | 1) `HF_HOME` 默认指向不存在的 `/mnt/aigc/shared_env/huggingface`，已修复为 `~/.cache/huggingface`； | `bash setup/models/hunyuan-video-i2v/setup.sh` |
| morphic-frames-to-video | 缺 Wan2.2 + morphic LoRA 权重 | `huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./weights/wan/Wan2.2-I2V-A14B` + `huggingface-cli download morphic/Wan2.2-frames-to-video --local-dir ./weights/morphic` |
