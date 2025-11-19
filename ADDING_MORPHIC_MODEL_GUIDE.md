# 添加 Morphic 模型到 VMEvalKit - 完整指南

## 📋 任务概述

将 Morphic Frames-to-Video 模型集成到 VMEvalKit 框架中，使其能够像其他模型一样参与推理测试（chess、maze、sudoku、rotation、raven 等任务）。

## 🎯 目标

完成以下工作，使 Morphic 模型能够：
1. 通过 `examples/generate_videos.py` 运行推理
2. 处理所有任务类型（chess、maze、sudoku、rotation、raven）
3. 生成符合 VMEvalKit 格式的视频输出
4. 参与评分和评估流程
5. 在 Web Dashboard 中显示结果

---

## 📝 实施步骤

### 阶段一：添加 Submodule

#### 步骤 1.1：添加 Morphic 作为 Git Submodule

```bash
cd /Users/maiwang/VMEvalKit-feature-add-morphic
git submodule add https://github.com/morphicfilms/frames-to-video.git submodules/morphic-frames-to-video
```

**验证**：检查 `submodules/morphic-frames-to-video/` 目录是否存在，并包含 `generate.py` 文件。

---

### 阶段二：实现核心代码

#### 步骤 2.1：完善 `morphic_inference.py`

**文件位置**：`vmevalkit/models/morphic_inference.py`

**当前状态**：只有文档字符串，需要完整实现。

**参考文件**：
- `vmevalkit/models/hunyuan_inference.py` - subprocess 模式参考
- `vmevalkit/models/videocrafter_inference.py` - 复杂命令构建参考
- `vmevalkit/models/base.py` - 接口定义

**需要实现的类和方法**：

1. **MorphicService 类**
   ```python
   class MorphicService:
       def __init__(self, model_id, output_dir, **kwargs):
           # 1. 定义 submodule 路径
           # 2. 检查 submodule 是否存在
           # 3. 从环境变量或 kwargs 读取权重路径
           # 4. 验证权重路径存在
           # 5. 初始化配置参数
       
       def _validate_paths(self):
           # 验证所有必需路径存在
           # - submodules/morphic-frames-to-video/generate.py
           # - Wan2.2 权重目录
           # - LoRA 权重文件
       
       def _run_morphic_inference(self, image_path, text_prompt, final_image_path, **kwargs):
           # 1. 构建 torchrun 命令
           # 2. 使用 subprocess 执行
           # 3. 处理输出和错误
           # 4. 返回标准格式结果
       
       def generate(self, image_path, text_prompt, duration, output_filename, **kwargs):
           # 统一接口，调用 _run_morphic_inference
   ```

2. **MorphicWrapper 类**
   ```python
   class MorphicWrapper(ModelWrapper):
       def __init__(self, model, output_dir, **kwargs):
           # 初始化 wrapper，创建 MorphicService 实例
       
       def generate(self, image_path, text_prompt, duration, output_filename, **kwargs):
           # 实现 ModelWrapper 接口
           # 从 kwargs 或 question_data 获取 final_image_path
           # 调用 service.generate()
           # 返回标准格式结果
   ```

**关键实现点**：

1. **路径定义**：
   ```python
   MORPHIC_PATH = Path(__file__).parent.parent.parent / "submodules" / "morphic-frames-to-video"
   ```

2. **获取 final_image_path**：
   ```python
   question_data = kwargs.get('question_data', {})
   final_image_path = question_data.get('final_image_path')
   if not final_image_path:
       # 错误处理或 fallback
   ```

3. **构建 torchrun 命令**：
   ```python
   cmd = [
       "torchrun",
       f"--nproc_per_node={nproc}",
       str(MORPHIC_PATH / "generate.py"),
       "--task", "i2v-A14B",
       "--size", "1280*720",
       "--frame_num", "81",
       "--ckpt_dir", wan2_ckpt_dir,
       "--high_noise_lora_weights_path", lora_weights_path,
       "--dit_fsdp",
       "--t5_fsdp",
       "--ulysses_size", "8",
       "--image", str(image_path),
       "--prompt", text_prompt,
       "--img_end", str(final_image_path),
   ]
   ```

4. **执行 subprocess**：
   ```python
   result = subprocess.run(
       cmd,
       cwd=str(MORPHIC_PATH),
       capture_output=True,
       text=True,
       timeout=900  # 15分钟超时
   )
   ```

5. **返回标准格式**：
   ```python
   return {
       "success": bool,
       "video_path": str | None,
       "error": str | None,
       "duration_seconds": float,
       "generation_id": str,
       "model": str,
       "status": str,
       "metadata": dict
   }
   ```

---

### 阶段三：注册模型

#### 步骤 3.1：在 `MODEL_CATALOG.py` 中添加模型定义

**文件位置**：`vmevalkit/runner/MODEL_CATALOG.py`

**在 "OPEN-SOURCE MODELS (SUBMODULES)" 部分添加**：

```python
# Morphic Frames-to-Video Models
MORPHIC_MODELS = {
    "morphic-frames-to-video": {
        "wrapper_module": "vmevalkit.models.morphic_inference",
        "wrapper_class": "MorphicWrapper",
        "service_class": "MorphicService",
        "model": "morphic-frames-to-video",
        "description": "Morphic Frames to Video - High-quality interpolation using Wan2.2",
        "family": "Morphic",
        "args": {
            "size": "1280*720",
            "frame_num": 81,
            "nproc_per_node": 8
        }
    }
}
```

**在文件底部合并到统一注册表**：

```python
AVAILABLE_MODELS = {
    **LUMA_MODELS,
    **VEO_MODELS,
    # ... 其他模型
    **MORPHIC_MODELS,  # 添加这一行
    # ... 其他模型
}

MODEL_FAMILIES = {
    "Luma Dream Machine": LUMA_MODELS,
    # ... 其他家族
    "Morphic": MORPHIC_MODELS,  # 添加这一行
    # ... 其他家族
}
```

#### 步骤 3.2：更新 `models/__init__.py`

**文件位置**：`vmevalkit/models/__init__.py`

**在 `__all__` 列表中添加**：
```python
"MorphicService", "MorphicWrapper",
```

**在 `_MODULE_MAP` 字典中添加**：
```python
"morphic_inference": ["MorphicService", "MorphicWrapper"],
```

---

### 阶段四：配置环境变量

#### 步骤 4.1：更新 `env.template`

**文件位置**：`env.template`

**添加 Morphic 相关配置**：
```bash
# Morphic Frames-to-Video Configuration
MORPHIC_WAN2_CKPT_DIR=./Wan2.2-I2V-A14B
MORPHIC_LORA_WEIGHTS_PATH=./morphic-frames-lora-weights/lora_interpolation_high_noise_final.safetensors
MORPHIC_NPROC_PER_NODE=8
```

---

### 阶段五：测试验证

#### 步骤 5.1：基础功能测试

```bash
# 测试模型注册
python -c "from vmevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS; print('morphic-frames-to-video' in AVAILABLE_MODELS)"

# 测试动态加载
python -c "from vmevalkit.runner.inference import _load_model_wrapper; wrapper = _load_model_wrapper('morphic-frames-to-video'); print(wrapper)"
```

#### 步骤 5.2：单任务推理测试

```bash
# 确保有测试数据
python examples/create_questions.py --task chess --pairs-per-domain 1

# 测试单个任务
python examples/generate_videos.py --model morphic-frames-to-video --task-id chess_0000
```

#### 步骤 5.3：批量任务测试

```bash
# 测试单个域
python examples/generate_videos.py --model morphic-frames-to-video --task chess

# 测试多个域
python examples/generate_videos.py --model morphic-frames-to-video --task chess maze
```

---

## 🔑 关键注意事项

### 1. final_image_path 处理

Morphic 模型需要两个图像输入：
- `--image`：起始帧（VMEvalKit 的 `first_frame.png`）
- `--img_end`：结束帧（VMEvalKit 的 `final_frame.png`）

**获取方式**：
```python
# 在 MorphicWrapper.generate() 中
question_data = kwargs.get('question_data', {})
final_image_path = question_data.get('final_image_path')

if not final_image_path:
    return {
        "success": False,
        "error": "Morphic requires final_image_path in question_data",
        # ... 其他必需字段
    }
```

### 2. 权重路径配置

权重路径通过环境变量配置：
```python
wan2_ckpt_dir = os.getenv(
    "MORPHIC_WAN2_CKPT_DIR",
    "./Wan2.2-I2V-A14B"  # 默认路径
)

lora_weights_path = os.getenv(
    "MORPHIC_LORA_WEIGHTS_PATH",
    "./morphic-frames-lora-weights/lora_interpolation_high_noise_final.safetensors"
)
```

### 3. GPU 要求

Morphic 使用 `torchrun --nproc_per_node=8`，需要 8 个 GPU。如果 GPU 不足：
- 可以尝试 `--nproc_per_node=1`（单 GPU）
- 或在初始化时检查 GPU 数量并给出清晰错误提示

### 4. 错误处理

确保所有错误场景都有处理：
- Submodule 不存在
- 权重路径不存在
- GPU 数量不足
- final_image_path 不存在
- torchrun 执行失败
- 超时

### 5. 输出路径

确保生成的视频保存到 `self.output_dir`，并且路径正确。

---

## 📚 参考实现

### 类似模型实现参考

1. **HunyuanVideo** (`hunyuan_inference.py`)
   - 使用 subprocess 调用 Python 脚本
   - 简单的命令构建

2. **VideoCrafter** (`videocrafter_inference.py`)
   - 复杂的命令构建
   - 临时脚本创建

3. **WAN** (`wan_inference.py`)
   - 直接 Python 调用（不是 subprocess）
   - 注意：WAN 有 `last_image` 参数，但实际传入的是同一个图像

### 接口规范参考

查看 `vmevalkit/models/base.py` 中的 `ModelWrapper` 抽象基类，确保实现符合接口要求。

---

## ✅ 完成检查清单

- [ ] Submodule 已添加
- [ ] `morphic_inference.py` 完整实现
- [ ] `MorphicService` 类实现完整
- [ ] `MorphicWrapper` 类实现完整
- [ ] 所有方法都有错误处理
- [ ] 返回格式符合 `ModelWrapper` 接口
- [ ] `MODEL_CATALOG.py` 中已注册
- [ ] `models/__init__.py` 中已导出
- [ ] `env.template` 中已添加配置
- [ ] 基础功能测试通过
- [ ] 单任务推理测试通过
- [ ] 批量任务测试通过

---

## 🚀 开始实施

按照以上步骤逐步实施，每完成一个阶段就进行测试验证，确保功能正常后再继续下一步。

如果在实施过程中遇到问题，参考现有模型的实现，或查看 VMEvalKit 的文档。

