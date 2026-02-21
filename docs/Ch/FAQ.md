# 常见问题 (FAQ)

## 安装相关

### Q: `pip install -e .` 安装后，运行开源模型报 `ModuleNotFoundError: No module named 'diffusers'`

**原因：** 开源模型（如 SVD、LTX-Video 等）的依赖（torch、diffusers、transformers 等）不包含在 VBVR-EvalKit 核心依赖中，需要额外安装。

**解决方法：**

方法一：在主 venv 中安装模型依赖（推荐快速测试时使用）：
```bash
source venv/bin/activate
pip install diffusers transformers accelerate torch torchvision
```

方法二：使用模型安装脚本（会创建独立 venv）：
```bash
bash setup/install_model.sh --model svd
```

> 注意：安装脚本在 `envs/{model-name}/` 下创建独立 venv，但当前推理脚本直接在主进程中 import 模型依赖，因此主 venv 中也需要有这些包。

---

### Q: 安装脚本报 `No matching distribution found for torchvision==0.20.1`

**原因：** Python 3.13 下旧版本的 torch/torchvision 不可用。固定版本号在新 Python 版本中可能失效。

**解决方法：** 修改 `setup/models/{model}/setup.sh`，去掉固定版本号：
```bash
# 修改前（可能不兼容）
pip install -q torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1

# 修改后（自动安装兼容版本）
pip install -q torch torchvision torchaudio
```

---

### Q: `nvidia-smi` 报 `Driver/library version mismatch` 或 CUDA 不可用

**原因：** NVIDIA 驱动版本与 CUDA 库版本不匹配。

**解决方法：**
1. SVD 等模型支持 CPU 回退，会自动使用 `float32` 在 CPU 上运行（速度较慢但可用）
2. 如需 GPU 加速，更新 NVIDIA 驱动或安装匹配的 CUDA 版本：
   ```bash
   # 查看当前驱动版本
   cat /proc/driver/nvidia/version
   # 安装匹配的 CUDA toolkit
   ```

---

## 推理相关

### Q: 运行 `generate_videos.py` 时任务被 `Skipped (existing)`

**原因：** 输出目录中已存在 `{task_id}.mp4`，默认跳过已完成的任务。

**解决方法：**
```bash
# 方法一：使用 --override 清空输出目录重新运行
python examples/generate_videos.py --questions-dir ./questions --model svd --override

# 方法二：手动删除对应的 mp4 文件
rm outputs/svd/test_task/test_0000.mp4
```

---

### Q: `--list-models` 报错 `the following arguments are required: --model`

**原因：** `--model` 是必需参数，即使使用 `--list-models` 也需要提供。

**解决方法：** 使用 Python 直接列出模型：
```bash
python -c "
from vbvrevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS, MODEL_FAMILIES
for f, ms in MODEL_FAMILIES.items():
    print(f'{f} ({len(ms)}):')
    for m in ms: print(f'  {m}')
print(f'\nTotal: {len(AVAILABLE_MODELS)} models')
"
```

---

### Q: CPU 模式下推理非常慢

**原因：** 开源模型在 CPU 上使用 `float32` 运算，计算量巨大。

**参考耗时（SVD，CPU，单张图）：** 约 2 分钟/任务

**建议：**
- 使用 GPU 环境（推荐 16GB+ 显存）
- 或使用商业 API 模型（无需 GPU）：
  ```bash
  python examples/generate_videos.py --questions-dir ./questions --model luma-ray-2
  ```

---

## 评估相关

### Q: 如何运行 VBVR-Bench 评估？

VBVR-Bench 是 VBVR-EvalKit 的评估系统，使用 100+ 个任务专用规则评估器，无需 API 调用：

```bash
# 基本评估（仅 task_specific 维度）
python examples/score_videos.py --inference-dir ./outputs

# 完整 5 维加权评分
python examples/score_videos.py --inference-dir ./outputs --full-score

# 指定 GT 数据
python examples/score_videos.py --inference-dir ./outputs --gt-base-path /path/to/gt --device cuda
```

---

### Q: 评分维度是什么意思？

| 维度 | 权重 | 说明 |
|------|------|------|
| `first_frame_consistency` | 15% | 首帧与输入图像的匹配程度 |
| `final_frame_accuracy` | 35% | 末帧与预期结果的匹配程度 |
| `temporal_smoothness` | 15% | 相邻帧之间的连续性 |
| `visual_quality` | 10% | 清晰度、噪声水平 |
| `task_specific` | 25% | 任务特定推理正确性 |

默认模式只返回 `task_specific`。使用 `--full-score` 获取加权组合分数。

---

### Q: 中断的评估可以恢复吗？

可以。VBVR-Bench 在每个任务完成后保存进度。只需重新运行相同命令，已评估的任务会自动跳过。

---

## 环境变量

### Q: 需要哪些 API Key？

API Key 仅在使用商业模型进行**推理**时需要，评估不需要 API Key。

| 模型家族 | 环境变量 | 获取方式 |
|----------|----------|----------|
| Luma | `LUMA_API_KEY` | Luma AI 官网 |
| Google Veo | `GEMINI_API_KEY` | [Google AI Studio](https://aistudio.google.com/app/apikey) |
| Kling AI | `KLING_API_KEY` | Kling AI 官网 |
| Runway | `RUNWAYML_API_SECRET` | Runway ML 官网 |
| OpenAI Sora | `OPENAI_API_KEY` | OpenAI 平台 |

```bash
cp env.template .env
# 编辑 .env 填写对应的 Key
```
