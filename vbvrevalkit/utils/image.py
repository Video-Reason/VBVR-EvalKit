"""Image utility functions shared across model wrappers."""

from pathlib import Path
from typing import Union

from PIL import Image


def load_image_rgb(image_path: Union[str, Path]) -> Image.Image:
    """Load an image from disk and convert to RGB mode."""
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image
