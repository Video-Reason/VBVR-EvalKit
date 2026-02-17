# VMEvalKit 评估打分

用于评估视频生成模型推理能力的综合打分方法。

## 可用的评估器

### 人工评估
基于 Gradio 的交互式人工打分界面。

```bash
python examples/score_videos.py --eval-config eval_config.json
# 在配置中设置 "method": "human"
```

### GPT-4O 评估
使用 OpenAI GPT-4O 视觉模型的自动化打分。

```bash
# 需要 OPENAI_API_KEY
python examples/score_videos.py --eval-config eval_config.json
# 在配置中设置 "method": "gpt4o"
```

### InternVL 评估
开源 VLM 评估（需要 30GB 显存）。

```bash
# 启动 InternVL 服务
bash script/lmdeploy_server.sh

# 运行评估
python examples/score_videos.py --eval-config eval_config.json
# 在配置中设置 "method": "internvl"
```

### Qwen3-VL 评估
使用 Qwen3-VL 的开源 VLM 评估，通过 OpenAI 兼容 API 提供服务。

```bash
# 启动 Qwen3-VL 服务（例如通过 vLLM 或 SGLang）
# 在 .env 中设置 QWEN_API_KEY 和 QWEN_API_BASE

# 运行评估
python examples/score_videos.py --eval-config eval_config.json
# 在配置中设置 "method": "qwen"
```

### 多帧评估
高级评估方法，使用多个视频帧进行一致性分析和投票。

```bash
# 多帧 GPT-4O、InternVL 或 Qwen3-VL
# 在配置中设置 "method": "multiframe_gpt4o"、"multiframe_internvl" 或 "multiframe_qwen"
```



## 评分标准

**1-5 分制**转换为**二分类**用于分析：
- **成功**：4-5 分（大部分/完全正确）
- **失败**：1-3 分（错误/部分正确）

## 配置

创建 `eval_config.json` 来配置评估：

```json
{
  "method": "gpt4o",
  "inference_dir": "./outputs",
  "eval_output_dir": "./evaluations",
  "temperature": 0.0,
  "multiframe": {
    "n_frames": 5,
    "strategy": "hybrid",
    "voting": "weighted_majority"
  }
}
```

## 使用方法

```bash
# 运行评估
python examples/score_videos.py --eval-config eval_config.json

# 测试多帧管线（不调用 API）
python examples/score_videos.py --test-multiframe --video path/to/video.mp4
```

## 输出

评估结果保存到配置中 `eval_output_dir` 指定的目录，包含结构化 JSON 文件（含分数、元数据和解释）。结果支持断点续评和统计分析。
