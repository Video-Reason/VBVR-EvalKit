# Image Resize Strategy

## Problem

Diffusion pipelines like Wan require `height` and `width` to be divisible by `mod_value` (typically 16), otherwise they error out.

## Two Paths

`WanService.generate_video` has two resize paths:

### 1. Ground-truth Path (height/width provided)

Preserves the original resolution, only aligns to `mod_value`:

```python
mod_value = vae_scale_factor_spatial * patch_size[1]
height = round(height / mod_value) * mod_value
width = round(width / mod_value) * mod_value
image = image.resize((width, height))
```

Size change is minimal (at most +/-8 pixels), preserving the original aspect ratio and dimensions.

### 2. Aspect-ratio Path (no height/width provided)

Recalculates dimensions based on `max_area` (default 720x1280 = 921600):

```python
height = round(sqrt(max_area * aspect_ratio)) // mod_value * mod_value
width  = round(sqrt(max_area / aspect_ratio)) // mod_value * mod_value
```

Output area is always approximately `max_area`, regardless of original image size. Automatically aligned to `mod_value`.

## Selection Logic

| Condition | Path | Effect |
|-----------|------|--------|
| Caller provides height/width | ground-truth | Preserves original resolution + alignment |
| No height/width provided | aspect-ratio | Scales to max_area + alignment |
