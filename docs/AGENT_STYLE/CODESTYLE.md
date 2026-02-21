# 代码风格

- 不要用 `sys.exit(1)`，用 raise 或 return。
- 不要用 `sys.path.append`，包通过 `pip install -e .` 安装。

## 图片加载

所有 model wrapper 统一使用 `vbvrevalkit.utils.image.load_image_rgb(path)` 加载图片并转 RGB，不要在每个 wrapper 里重复写 `Image.open(...).convert("RGB")`。

```python
from vbvrevalkit.utils.image import load_image_rgb

image = load_image_rgb(image_path)
```

例外：如果需要 `with Image.open() as img:` 管理文件句柄（如 openai_inference.py 的 pad 逻辑），保留原写法。

## utils 模块

`vbvrevalkit/utils/__init__.py` 使用 `__getattr__` 懒加载重依赖（如 boto3 的 S3ImageUploader），避免在模型 venv 中拉入不必要的依赖。轻量工具（如 `load_image_rgb`）直接 import。
