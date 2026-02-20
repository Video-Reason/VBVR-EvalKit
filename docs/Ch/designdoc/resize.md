# 图片 Resize 策略

## 问题

Wan 等 diffusion pipeline 要求 `height` 和 `width` 必须能被 `mod_value`（通常为 16）整除，否则报错。

## 两条路径

`WanService.generate_video` 中有两条 resize 路径：

### 1. Ground-truth 路径（传了 height/width）

保留原始分辨率，仅对齐到 `mod_value`：

```python
mod_value = vae_scale_factor_spatial * patch_size[1]
height = round(height / mod_value) * mod_value
width = round(width / mod_value) * mod_value
image = image.resize((width, height))
```

尺寸变化很小（最多 ±8 像素），保持原图比例和大小。

### 2. Aspect-ratio 路径（没传 height/width）

按 `max_area`（默认 720×1280 = 921600）重新计算尺寸：

```python
height = round(sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width  = round(sqrt(max_area / aspect_ratio)) // mod_value * mod_value
```

输出面积总是 ≈ `max_area`，不管原图大小。自动对齐到 `mod_value`。

## 选择逻辑

| 条件 | 路径 | 效果 |
|------|------|------|
| 调用方传了 height/width | ground-truth | 保留原始分辨率 + 对齐 |
| 未传 height/width | aspect-ratio | 缩放到 max_area + 对齐 |
