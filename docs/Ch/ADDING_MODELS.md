# 向 VMEvalKit 添加模型

VMEvalKit 支持两种模型类型，分别采用不同的集成方式：

## 快速参考

### 商业 API 模型
1. 创建 `{provider}_inference.py`，包含 Service + Wrapper 类
2. 在 `MODEL_CATALOG.py` 中添加条目
3. 在 `.env` 中设置 API Key
4. 使用 `examples/generate_videos.py` 测试

### 开源模型
1. 创建 `{model}_inference.py`，包含 Service + Wrapper 类
2. 创建 `setup/models/{model-name}/setup.sh` 安装脚本
3. 在 `setup/lib/share.sh` 中注册 checkpoint
4. 在 `MODEL_CATALOG.py` 中添加条目
5. 运行 `bash setup/install_model.sh --model {model-name}` 安装

| 方面 | 商业 API | 开源模型 |
|------|---------|---------|
| **安装** | 仅需 API Key | 完整安装（10-30 分钟） |
| **存储** | 无 | 每个模型 5-25 GB |
| **GPU** | 不需要 | 需要（8-24GB 显存） |
| **示例** | Luma、Veo、Kling、Sora | LTX-Video、LTX-2、SVD、HunyuanVideo |

## 架构

VMEvalKit 采用 **Service + Wrapper 模式**：
- **Service**：处理 API 调用或模型推理
- **Wrapper**：继承自 `ModelWrapper`，提供统一接口
- **注册表**：`MODEL_CATALOG.py` 列出所有模型的动态加载路径
- **安装**：开源模型需要 `setup/models/{name}/setup.sh` 脚本

## 必需接口

所有模型必须继承 `ModelWrapper` 并实现：

```python
class YourModelWrapper(ModelWrapper):
    def generate(self, image_path, text_prompt, **kwargs) -> Dict[str, Any]:
        # 必须返回以下 8 个字段：
        return {
            "success": bool,
            "video_path": str | None,
            "error": str | None,
            "duration_seconds": float,
            "generation_id": str,
            "model": str,
            "status": str,
            "metadata": Dict[str, Any]
        }
```

## 安装

### 商业 API
```bash
# 将 API Key 添加到 .env
echo 'YOUR_PROVIDER_API_KEY=your_key' >> .env
# 即可使用！
```

### 开源模型
```bash
# 安装模型和依赖
bash setup/install_model.sh --model your-model-name

# 测试安装
python examples/generate_videos.py --model your-model-name --task-id test_0001
```

## 开源模型安装

### 安装脚本模板
创建 `setup/models/{model-name}/setup.sh`：

```bash
#!/bin/bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../lib/share.sh"

MODEL="your-model-name"

print_section "Virtual Environment"
create_model_venv "$MODEL"
activate_model_venv "$MODEL"

print_section "Dependencies"
pip install -q torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install -q transformers==4.25.1 diffusers==0.31.0

deactivate

print_section "Checkpoints"
download_checkpoint_by_path "${MODEL_CHECKPOINT_PATHS[$MODEL]}"

print_success "${MODEL} setup complete"
```

### 在 `setup/lib/share.sh` 中注册
```bash
# 添加到 OPENSOURCE_MODELS 数组
OPENSOURCE_MODELS+=("your-model-name")

# 添加 checkpoint 信息
CHECKPOINTS+=("your-model/model.ckpt|https://huggingface.co/.../model.ckpt|5.2GB")
MODEL_CHECKPOINT_PATHS["your-model-name"]="your-model/model.ckpt"
```

## 注册

### 添加到 MODEL_CATALOG.py

```python
# 在 vmevalkit/runner/MODEL_CATALOG.py 中
YOUR_MODELS = {
    "your-model-v1": {
        "wrapper_module": "vmevalkit.models.your_inference",
        "wrapper_class": "YourWrapper",
        "model": "v1",
        "description": "Your model description",
        "family": "YourProvider"
    }
}

# 添加到 AVAILABLE_MODELS
AVAILABLE_MODELS = {**EXISTING_MODELS, **YOUR_MODELS}
```

## 测试

```bash
# 测试安装
bash setup/install_model.sh --model your-model-name

# 测试推理
python examples/generate_videos.py --model your-model-name --task-id test_0001

# 验证返回字典中的 8 个必需字段
```

## 关键要求

- **继承 ModelWrapper**：使用抽象基类
- **返回 8 个必需字段**：success、video_path、error、duration_seconds、generation_id、model、status、metadata
- **优雅处理错误**：返回错误字典，不要抛出异常
- **使用环境变量**：存放 API Key（禁止硬编码）
- **精确包版本**：在安装脚本中使用 `package==X.Y.Z`
- **Temperature = 0**：保持结果稳定可复现

## 参考示例

- **商业 API**：`vmevalkit/models/luma_inference.py`、`vmevalkit/models/kling_inference.py`
- **开源模型**：`vmevalkit/models/svd_inference.py`、`vmevalkit/models/ltx_inference.py`、`vmevalkit/models/ltx2_inference.py`
- **安装脚本**：`setup/models/*/setup.sh`
