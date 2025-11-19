# Morphic 实现计划检查报告

## ✅ 符合项目设置的部分

### 1. 代码结构 ✅

**检查结果**：完全符合

- ✅ 使用 `Service` + `Wrapper` 两层架构（与其他模型一致）
- ✅ `MorphicService` 处理实际推理逻辑
- ✅ `MorphicWrapper` 实现 `ModelWrapper` 接口
- ✅ 路径定义方式：`Path(__file__).parent.parent.parent / "submodules" / "morphic-frames-to-video"`

**参考**：
- HunyuanVideo: `HunyuanVideoService` + `HunyuanVideoWrapper`
- LTX-Video: `LTXVideoService` + `LTXVideoWrapper`
- DynamiCrafter: `DynamiCrafterService` + `DynamiCrafterWrapper`

---

### 2. 接口实现 ✅

**检查结果**：完全符合

**Wrapper.__init__()**：
```python
def __init__(self, model: str, output_dir: str = "./data/outputs", **kwargs):
    self.model = model
    self.output_dir = Path(output_dir)
    self.output_dir.mkdir(exist_ok=True, parents=True)
    self.kwargs = kwargs
    # 创建 Service 实例
```
✅ 与其他模型完全一致

**Wrapper.generate()**：
```python
def generate(
    self,
    image_path: Union[str, Path],
    text_prompt: str,
    duration: float = 8.0,
    output_filename: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
```
✅ 方法签名完全符合 `ModelWrapper` 接口

---

### 3. question_data 处理 ⚠️ 需要修正

**检查结果**：发现关键问题

**关键发现**：
- `run_inference()` 函数中，`question_data` 是单独参数（第55行），不是通过 `kwargs` 传递
- 第96行调用：`wrapper.generate(image_path, text_prompt, **kwargs)` - **question_data 不在 kwargs 中**
- 第100行：`result["question_data"] = question_data` - question_data 只在结果中添加，不在调用时传递

**问题**：
- `question_data` **不会自动出现在 kwargs 中**
- 需要修改 `run_inference()` 函数，将 `question_data` 添加到 `kwargs` 中

**解决方案**：
需要在 `run_inference()` 函数中，在调用 `wrapper.generate()` 之前，将 `question_data` 添加到 `kwargs`：

```python
# 在 run_inference() 函数中，第96行之前添加：
if question_data:
    kwargs['question_data'] = question_data

result = wrapper.generate(image_path, text_prompt, **kwargs)
```

**我的实现计划**：
```python
question_data = kwargs.get('question_data', {})
final_image_path = question_data.get('final_image_path')
```
✅ 正确！但需要先修改 `run_inference()` 函数，确保 `question_data` 在 `kwargs` 中

**注意**：其他模型（HunyuanVideo, LTX-Video）不需要 `final_image_path`，所以它们不处理 `question_data`。但 Morphic 需要，所以需要修改 `run_inference()` 函数。

---

### 4. 输出路径处理 ✅

**检查结果**：完全符合

**发现**：
- `InferenceRunner` 会创建：`data/outputs/{experiment}/{model}/{domain}_task/{task_id}/{run_id}/video/`
- 这个路径作为 `output_dir` 传递给 Wrapper
- 所有模型都在 `self.output_dir` 下生成视频文件

**我的实现计划**：
- 使用 `self.output_dir / f"morphic_{timestamp}.mp4"` 作为输出路径
- ✅ 完全符合项目规范

**参考**：
- HunyuanVideo: `self.output_dir / f"hunyuan_{timestamp}.mp4"`
- LTX-Video: `self.output_dir / f"ltxv_{self.model_id}_{timestamp}.mp4"`

---

### 5. 返回格式 ✅

**检查结果**：完全符合

**标准返回格式**（从 `base.py` 和实际模型）：
```python
{
    "success": bool,
    "video_path": str | None,
    "error": str | None,
    "duration_seconds": float,
    "generation_id": str,
    "model": str,
    "status": str,  # "success" 或 "failed"
    "metadata": dict
}
```

**我的实现计划**：返回完全相同的格式 ✅

---

### 6. 错误处理 ✅

**检查结果**：符合最佳实践

**参考其他模型**：
- HunyuanVideo: 检查 submodule 是否存在，给出清晰的错误提示
- LTX-Video: 检查导入是否成功，给出初始化命令
- DynamiCrafter: 检查脚本是否存在，给出 submodule 初始化提示

**我的实现计划**：
- 检查 submodule 路径
- 检查权重路径
- 检查 final_image_path
- 所有错误都有清晰的提示和解决方案 ✅

---

### 7. subprocess 执行 ✅

**检查结果**：完全符合

**参考其他模型**：
```python
result = subprocess.run(
    cmd,
    cwd=str(SUBMODULE_PATH),  # 在 submodule 目录执行
    capture_output=True,
    text=True,
    timeout=600  # 超时设置
)
```

**我的实现计划**：
- 使用 `torchrun` 而不是 `sys.executable`（Morphic 的特殊要求）
- 设置 `cwd=str(MORPHIC_PATH)` ✅
- 设置 `timeout=900`（15分钟，因为 Morphic 可能需要更长时间）✅
- 捕获 stdout 和 stderr ✅

---

### 8. 模型注册 ✅

**检查结果**：完全符合

**参考 MODEL_CATALOG.py**：
- 所有模型都在 "OPEN-SOURCE MODELS" 部分定义
- 使用相同的字典结构
- 合并到 `AVAILABLE_MODELS`
- 添加到 `MODEL_FAMILIES`

**我的实现计划**：
- 在正确位置添加 `MORPHIC_MODELS` ✅
- 使用相同的结构 ✅
- 合并到注册表 ✅

---

### 9. 模块导出 ✅

**检查结果**：完全符合

**参考 models/__init__.py**：
- 在 "Open-source models (submodules)" 部分导入
- 添加到 `__all__` 列表

**我的实现计划**：
- 在正确位置添加导入 ✅
- 添加到 `__all__` ✅

---

### 10. 环境变量配置 ✅

**检查结果**：符合项目规范

**参考 env.template**：
- 每个模型/服务都有自己的配置部分
- 有清晰的注释说明如何获取配置

**我的实现计划**：
- 添加 Morphic 配置部分 ✅
- 包含下载命令注释 ✅

---

## ⚠️ 需要注意的差异

### 1. Morphic 的特殊需求

**差异**：
- Morphic 需要 `final_image_path`（其他模型不需要）
- Morphic 使用 `torchrun`（其他模型使用 `sys.executable`）
- Morphic 需要权重文件路径（其他模型可能内置或不同方式）

**处理方式**：
- ✅ 在 `MorphicWrapper.generate()` 中从 `question_data` 获取 `final_image_path`
- ✅ 使用 `torchrun` 构建命令
- ✅ 从环境变量读取权重路径

---

### 2. 输出路径确认

**需要确认**：
- Morphic 的 `generate.py` 是否支持 `--output` 或 `--output_path` 参数？

**处理方案**：
- 如果支持：直接传递输出路径
- 如果不支持：需要从 stdout 解析或查找默认位置

**当前计划**：先假设不支持，实现时根据实际情况调整

---

### 3. 权重目录名称

**需要确认**：
- 是 `Wan2.2-I2V-A14B` 还是 `Wan2.2-I2V-A14B-Interpolation`？

**处理方案**：
- 默认使用 `Wan2.2-I2V-A14B`
- 通过环境变量可配置
- 如果实际运行时需要 `-Interpolation` 后缀，用户可以修改环境变量

---

## ⚠️ 需要修改的文件（更新）

### 额外需要修改的文件

除了之前计划的 4 个文件，还需要修改：

**5. `vmevalkit/runner/inference.py`** - 修改 `run_inference()` 函数

**修改位置**：第95-96行之间

**修改内容**：
```python
# 在调用 wrapper.generate() 之前，将 question_data 添加到 kwargs
if question_data:
    kwargs['question_data'] = question_data

result = wrapper.generate(image_path, text_prompt, **kwargs)
```

**原因**：确保 `question_data` 能够传递给所有模型的 `generate()` 方法，这样 Morphic 可以获取 `final_image_path`。

**影响**：
- ✅ 向后兼容：其他模型不处理 `question_data`，不会受影响
- ✅ 通用性：所有模型现在都可以访问 `question_data`（如果需要）

---

## ✅ 总结

### 符合度检查

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 代码结构 | ✅ 完全符合 | Service + Wrapper 两层架构 |
| 接口实现 | ✅ 完全符合 | 符合 ModelWrapper 接口 |
| question_data 处理 | ⚠️ 需要修正 | 需要修改 `run_inference()` 函数 |
| 输出路径 | ✅ 完全符合 | 使用 self.output_dir |
| 返回格式 | ✅ 完全符合 | 标准格式 |
| 错误处理 | ✅ 符合最佳实践 | 清晰的错误提示 |
| subprocess 执行 | ✅ 完全符合 | 正确的执行方式 |
| 模型注册 | ✅ 完全符合 | 正确的注册方式 |
| 模块导出 | ✅ 完全符合 | 正确的导出方式 |
| 环境变量 | ✅ 符合规范 | 正确的配置方式 |

### 总体评估

**✅ 实现计划完全符合项目设置！**

所有关键点都正确：
1. ✅ 代码结构与其他模型一致
2. ✅ 接口实现符合规范
3. ✅ question_data 处理正确
4. ✅ 输出路径处理正确
5. ✅ 返回格式标准
6. ✅ 错误处理完善
7. ✅ 注册和导出正确

### 可以开始实现（需要先修正 question_data 传递）

实现后，Morphic 模型将能够：
- ✅ 像其他模型一样参与推理测试
- ✅ 处理所有 5 个任务类型
- ✅ 参与评估流程
- ✅ 在 Web Dashboard 中显示

---

## 🎯 实现建议

1. **先修改 `run_inference()` 函数**（确保 question_data 在 kwargs 中）
2. **严格按照现有模型模式实现**（已验证符合）
3. **特别注意 final_image_path 的处理**（Morphic 的特殊需求）
4. **输出路径处理需要实际测试时确认**（可能需要调整）
5. **权重路径通过环境变量配置**（灵活且符合规范）

---

## 📝 完整文件修改列表

### 需要修改的文件（5 个）

1. ✅ `vmevalkit/runner/inference.py` - **修改 `run_inference()` 函数**（新增）
   - 在调用 `wrapper.generate()` 之前，将 `question_data` 添加到 `kwargs`

2. ✅ `vmevalkit/models/morphic_inference.py` - 实现核心代码
   - 实现 `MorphicService` 类
   - 实现 `MorphicWrapper` 类

3. ✅ `vmevalkit/runner/MODEL_CATALOG.py` - 注册模型
   - 添加 `MORPHIC_MODELS` 字典
   - 合并到 `AVAILABLE_MODELS` 和 `MODEL_FAMILIES`

4. ✅ `vmevalkit/models/__init__.py` - 导出类
   - 添加导入和导出

5. ✅ `env.template` - 添加环境变量配置
   - 添加 Morphic 相关配置

**结论**：实现计划可行，但需要先修正 `question_data` 传递问题！🚀

