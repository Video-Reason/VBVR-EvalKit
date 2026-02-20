# Utils module - lazy imports to avoid pulling heavy deps (boto3) into model venvs
from .image import load_image_rgb

__all__ = ['S3ImageUploader', 'load_image_rgb']


def __getattr__(name):
    if name == "S3ImageUploader":
        from .s3_uploader import S3ImageUploader
        return S3ImageUploader
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
