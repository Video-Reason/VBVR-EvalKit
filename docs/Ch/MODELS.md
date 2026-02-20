
# 支持的模型

VMEvalKit 提供对 **33 个视频生成模型**（覆盖 **13 个模型家族**）的统一访问。

## 商业 API（19 个模型）

### Luma Dream Machine（2 个模型）
**API Key:** `LUMA_API_KEY`
- `luma-ray-2` - 最新模型，画质最佳
- `luma-ray-flash-2` - 更快的生成速度

### Google Veo（6 个模型）
**API Key:** `GEMINI_API_KEY`
- `veo-2` - GA 版本，支持文本+图片→视频
- `veo-2.0-generate` - GA 版本，支持文本+图片→视频
- `veo-3.0-generate` - 高级视频生成模型
- `veo-3.0-fast-generate` - 更快的生成模型
- `veo-3.1-generate` - 最新模型，原生 1080p 和音频（预览版）
- `veo-3.1-fast` - Veo 3.1 快速变体（预览版）


### Kling AI（5 个模型）
**API Key:** `KLING_API_KEY`
- `kling-v2-6` - 最新 Kling 模型，画质最佳
- `kling-v2-5-turbo` - 快速生成模型
- `kling-v2-1-master` - 高画质模型
- `kling-v2-master` - 画质与速度的平衡
- `kling-v1-6` - 改进版原始模型

### Runway ML（4 个模型）
**API Key:** `RUNWAYML_API_SECRET`
- `runway-gen45` - 全球最高评分的视频模型（5s 或 10s）
- `runway-gen4-turbo` - 快速高画质生成（5s 或 10s）
- `runway-gen4-aleph` - 顶级画质（5s）
- `runway-gen3a-turbo` - 稳定可靠的性能（5s 或 10s）

### OpenAI Sora（2 个模型）
**API Key:** `OPENAI_API_KEY`
- `openai-sora-2` - 高画质视频生成（4s/8s/12s）
- `openai-sora-2-pro` - 增强版模型，支持更多分辨率

## 开源模型（14 个模型）

### LTX-Video（3 个模型）
**显存:** 16-40GB | **安装:** `bash setup/install_model.sh ltx-video`（仅装依赖）或 `bash setup/models/LTX-2/setup.sh`（完整安装含模型权重）
- `ltx-video` - 高画质图生视频（704x480, 24fps）
- `ltx-video-13b-distilled` - 13B 参数蒸馏版
- `LTX-2` - 19B FP8 文本/图片生视频，支持音频生成（~40GB 显存）

### HunyuanVideo（1 个模型）
**显存:** 24GB+ | **安装:** `bash setup/models/hunyuan-video-i2v/setup.sh`（需要 conda，Python 3.10）
- `hunyuan-video-i2v` - 高画质图生视频，最高 720p

### Morphic（1 个模型）
**显存:** 20GB+ | **安装:** `bash setup/install_model.sh morphic-frames-to-video`
- `morphic-frames-to-video` - 基于 Wan2.2 的高画质插帧

### Stable Video Diffusion（1 个模型）
**显存:** 20GB | **安装:** `bash setup/install_model.sh svd`
- `svd` - 高画质图生视频

### WAN (Wan-AI)（4 个模型）
**显存:** 48GB+ | **安装:** `bash setup/install_model.sh wan-2.2-ti2v-5b`
- `wan-2.1-i2v-480p` - 480p 图生视频
- `wan-2.1-i2v-720p` - 720p 图生视频
- `wan-2.2-i2v-a14b` - 14B 参数图生视频
- `wan-2.2-ti2v-5b` - 5B 参数文本+图片生视频

### CogVideoX（2 个模型）
**显存:** 20GB+ | **安装:** `bash setup/install_model.sh cogvideox-5b-i2v`
- `cogvideox-5b-i2v` - 6s 图+文生视频（720x480）
- `cogvideox1.5-5b-i2v` - 10s 图+文生视频（1360x768）

### SANA-Video（1 个模型）
**显存:** 16GB+ | **安装:** `bash setup/install_model.sh sana-video-2b-480p`
- `sana-video-2b-480p` - 高效文本+图片生视频（480x832）

### Sana（1 个模型）
**显存:** 16GB+ | **安装:** `bash setup/install_model.sh sana`
- `sana` - 支持运动控制的图生视频


## 使用方法

### 列出可用模型
```bash
python examples/generate_videos.py --list-models
```

### 快速开始示例

#### 商业 API（即装即用）
```bash
# Luma Dream Machine - 最佳画质
python examples/generate_videos.py --questions-dir ./questions --model luma-ray-2

# Google Veo 3.1 - 最新，支持 1080p + 音频
python examples/generate_videos.py --questions-dir ./questions --model veo-3.1-generate

# Kling AI 2.6 - 最新 Kling，最佳画质
python examples/generate_videos.py --questions-dir ./questions --model kling-v2-6

# Runway Gen-4.5 - 全球最高评分视频模型
python examples/generate_videos.py --questions-dir ./questions --model runway-gen45

# OpenAI Sora 2 - 高画质生成
python examples/generate_videos.py --questions-dir ./questions --model openai-sora-2
```

#### 开源模型（需要安装）
```bash
# LTX-Video - 轻量级，画质好
bash setup/install_model.sh ltx-video
python examples/generate_videos.py --questions-dir ./questions --model ltx-video

# Stable Video Diffusion - 经典模型
bash setup/install_model.sh svd
python examples/generate_videos.py --questions-dir ./questions --model svd

# HunyuanVideo - 高画质，最高 720p（需要 conda）
bash setup/models/hunyuan-video-i2v/setup.sh
python examples/generate_videos.py --questions-dir ./questions --model hunyuan-video-i2v

# CogVideoX - 长视频生成
bash setup/install_model.sh cogvideox-5b-i2v
python examples/generate_videos.py --questions-dir ./questions --model cogvideox-5b-i2v
```
