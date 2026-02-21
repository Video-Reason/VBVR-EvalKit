# Code Style

- Do not use `sys.exit(1)` — use `raise` or `return` instead.
- Do not use `sys.path.append` — the package is installed via `pip install -e .`.

## Image Loading

All model wrappers must use `vbvrevalkit.utils.image.load_image_rgb(path)` to load images and convert to RGB. Do not write `Image.open(...).convert("RGB")` in each wrapper.

```python
from vbvrevalkit.utils.image import load_image_rgb

image = load_image_rgb(image_path)
```

Exception: if you need `with Image.open() as img:` to manage the file handle (e.g., padding logic in `openai_inference.py`), keep the original approach.

## Utils Module

`vbvrevalkit/utils/__init__.py` uses `__getattr__` to lazy-load heavy dependencies (e.g., boto3's `S3ImageUploader`), avoiding pulling unnecessary dependencies into model venvs. Lightweight utilities (e.g., `load_image_rgb`) are imported directly.
