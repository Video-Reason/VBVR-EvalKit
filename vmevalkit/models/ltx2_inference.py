"""
LTX-2 Inference Service for VMEvalKit

Wrapper for the LTX-2 model to integrate with VMEvalKit's unified inference interface.
Supports text-to-video and image-to-video generation using the LTX-2 19B model.

The model is downloaded to LTX-2 directory at project root.
Requires ~40GB VRAM, inference takes ~6 minutes on a single A6000.
"""

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image

from .base import ModelWrapper
from ..utils.image import load_image_rgb

# Paths
VMEVAL_ROOT = Path(__file__).parent.parent.parent
LTX2_PATH = VMEVAL_ROOT / "LTX-2"


def ensure_ltx2_installed() -> bool:
    """
    Ensure LTX-2 is cloned and available in the project root directory.
    
    Returns:
        True if LTX-2 is ready, False otherwise.
    """
    if LTX2_PATH.exists() and (LTX2_PATH / "packages").exists():
        print(f"LTX-2 repository found at {LTX2_PATH}")
        return True
    
    print(f"LTX-2 not found at {LTX2_PATH}")
    print("Please clone LTX-2 repository:")
    print(f"  git clone https://github.com/Lightricks/LTX-2.git {LTX2_PATH}")
    return False


def ensure_checkpoints_downloaded(checkpoint_path: Path, gemma_path: Path) -> bool:
    """
    Check if model checkpoints are already downloaded.
    
    Args:
        checkpoint_path: Path to the main model checkpoint
        gemma_path: Path to the Gemma model directory
        
    Returns:
        True if all checkpoints exist, False otherwise.
    """
    checkpoint_exists = checkpoint_path.exists()
    gemma_exists = gemma_path.exists() and any(gemma_path.iterdir())
    
    if checkpoint_exists:
        print(f"Checkpoint found: {checkpoint_path.name}")
    else:
        print(f"Checkpoint missing: {checkpoint_path}")
        
    if gemma_exists:
        print(f"Gemma model found: {gemma_path.name}")
    else:
        print(f"Gemma model missing: {gemma_path}")
        
    return checkpoint_exists and gemma_exists


def add_ltx2_to_path():
    """Add LTX-2 packages to Python path."""
    ltx_core_path = LTX2_PATH / "packages" / "ltx-core" / "src"
    ltx_pipelines_path = LTX2_PATH / "packages" / "ltx-pipelines" / "src"
    
    for path in [ltx_core_path, ltx_pipelines_path]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def preprocess_image_for_ltx2(
    image_path: Path, 
    target_width: int, 
    target_height: int,
    output_dir: Path
) -> Path:
    """
    Preprocess image for LTX-2: scale proportionally to target width (no cropping).
    
    The image is scaled so that its width matches target_width, maintaining aspect ratio.
    Height is then padded or cropped to match target_height if needed.
    
    Args:
        image_path: Path to input image
        target_width: Target width in pixels
        target_height: Target height in pixels  
        output_dir: Directory to save preprocessed image
        
    Returns:
        Path to preprocessed image
    """
    img = load_image_rgb(image_path)
    orig_w, orig_h = img.size
    
    # Scale proportionally based on target width
    scale = target_width / orig_w
    new_w = target_width
    new_h = int(orig_h * scale)
    
    # Resize image proportionally
    img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Handle height: pad or crop to match target_height
    if new_h < target_height:
        # Pad with black bars (letterbox)
        result = Image.new("RGB", (target_width, target_height), (0, 0, 0))
        paste_y = (target_height - new_h) // 2
        result.paste(img_resized, (0, paste_y))
    elif new_h > target_height:
        # Crop from center
        crop_y = (new_h - target_height) // 2
        result = img_resized.crop((0, crop_y, target_width, crop_y + target_height))
    else:
        result = img_resized
    
    # Save preprocessed image
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessed_path = output_dir / f"preprocessed_{image_path.name}"
    result.save(preprocessed_path)
    
    print(f"Preprocessed image: {orig_w}x{orig_h} -> {target_width}x{target_height} (saved to {preprocessed_path})")
    
    return preprocessed_path


def _newest_hf_snapshot(snapshots_dir: Path) -> Optional[Path]:
    """Return the newest snapshot directory under an HF cache snapshots path, or None."""
    if not snapshots_dir.exists():
        return None
    dirs = sorted(
        (d for d in snapshots_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    return dirs[0] if dirs else None


class LTX2Service:
    """
    Service class for LTX-2 inference integration.
    Loads the model once and runs inference in-process.
    """

    # Model configuration mapping
    CONFIG_MAPPING = {
        "LTX-2": {
            "checkpoint": "ltx-2-19b-distilled-fp8.safetensors",
            "gemma_dir": "gemma3-12b-it-qat-q4_0-unquantized",  # Local directory name
            "gemma_hf_name": "gemma-3-12b-it-qat-q4_0-unquantized",  # HuggingFace cache name (with dash)
            "resolution": (768, 512),  # (height, width)
            "num_frames": 97,  # LTX-2 default
            "frame_rate": 25.0,
        },
    }

    def __init__(
        self,
        model_id: str = "LTX-2",
        output_dir: str = "./outputs",
        device: str = "cuda",
        enable_fp8: bool = True,
        **kwargs
    ):
        """
        Initialize LTX-2 service and load model.

        Args:
            model_id: LTX-2 model variant
            output_dir: Directory to save generated videos
            device: Device to run inference on
            enable_fp8: Enable FP8 transformer for memory efficiency
            **kwargs: Additional parameters
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = torch.device(device)
        self.enable_fp8 = enable_fp8
        self.kwargs = kwargs

        # Validate model_id
        assert model_id in self.CONFIG_MAPPING, (
            f"Unknown model_id: {model_id}. "
            f"Available: {list(self.CONFIG_MAPPING.keys())}"
        )

        self.model_config = self.CONFIG_MAPPING[model_id]
        self.resolution = self.model_config["resolution"]  # (H, W)
        self.num_frames = self.model_config["num_frames"]
        self.frame_rate = self.model_config["frame_rate"]

        # Ensure LTX-2 is installed and add to path
        if not ensure_ltx2_installed():
            raise RuntimeError("LTX-2 is not installed. Please run the setup script.")
        
        add_ltx2_to_path()

        # Resolve checkpoint and Gemma paths (HF cache first, then local)
        checkpoint_filename = self.model_config["checkpoint"]
        gemma_dirname = self.model_config["gemma_dir"]
        gemma_hf_name = self.model_config.get("gemma_hf_name", gemma_dirname)

        hf_cache_base = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()

        # LTX-2 checkpoint: search HF cache snapshots for the file, then local
        self.checkpoint_path = None
        snapshot = _newest_hf_snapshot(hf_cache_base / "models--Lightricks--LTX-2" / "snapshots")
        if snapshot:
            candidate = snapshot / checkpoint_filename
            if candidate.exists() or candidate.is_symlink():
                self.checkpoint_path = str(candidate)
                print(f"Using LTX-2 checkpoint from HuggingFace cache: {self.checkpoint_path}")
        if self.checkpoint_path is None:
            local_checkpoint = LTX2_PATH / checkpoint_filename
            if local_checkpoint.exists():
                self.checkpoint_path = str(local_checkpoint)
                print(f"Using local LTX-2 checkpoint: {self.checkpoint_path}")

        # Gemma model: search HF hub cache, then local directory
        self.gemma_root = None
        gemma_snapshot = _newest_hf_snapshot(
            hf_cache_base / "hub" / f"models--google--{gemma_hf_name}" / "snapshots"
        )
        if gemma_snapshot:
            self.gemma_root = str(gemma_snapshot)
            print(f"Using Gemma model from HuggingFace cache: {self.gemma_root}")
        if self.gemma_root is None:
            local_gemma = LTX2_PATH / gemma_dirname
            if local_gemma.exists():
                has_model_files = any(
                    f.suffix in ('.safetensors', '.bin', '.gguf')
                    for f in local_gemma.rglob('*') if f.is_file()
                )
                if has_model_files:
                    self.gemma_root = str(local_gemma)
                    print(f"Using local Gemma model: {self.gemma_root}")

        assert self.checkpoint_path is not None and Path(self.checkpoint_path).exists(), (
            f"LTX-2 checkpoint not found: {checkpoint_filename}\n"
            f"Searched in HuggingFace cache ({hf_cache_base}) and local directory ({LTX2_PATH})\n"
            f"Please run the setup script to download model weights."
        )
        assert self.gemma_root is not None and Path(self.gemma_root).exists(), (
            f"Gemma model not found: {gemma_dirname}\n"
            f"Searched in HuggingFace cache ({hf_cache_base}/hub/) and local directory ({LTX2_PATH})\n"
            f"Please run the setup script to download Gemma weights."
        )

        # Lazy load the pipeline
        self._pipeline = None
        print(f"LTX-2 service initialized: {model_id} @ {self.resolution}")

    def _load_pipeline(self):
        """Lazily load the LTX-2 pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        print("Loading LTX-2 pipeline...")
        from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
        from ltx_core.quantization.policy import QuantizationPolicy

        quantization = QuantizationPolicy.fp8_cast() if self.enable_fp8 else None

        self._pipeline = TI2VidOneStagePipeline(
            checkpoint_path=self.checkpoint_path,
            gemma_root=self.gemma_root,
            loras=[],  # No LoRAs by default
            device=self.device,
            quantization=quantization,
        )
        print("LTX-2 pipeline loaded.")
        return self._pipeline

    @torch.inference_mode()
    def generate(
        self,
        image_path: Optional[Union[str, Path]] = None,
        text_prompt: str = "",
        duration: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        frame_rate: Optional[float] = None,
        seed: Optional[int] = None,
        num_inference_steps: int = 30,
        cfg_guidance_scale: float = 3.0,
        negative_prompt: str = "",
        enhance_prompt: bool = False,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video from text prompt and optional image using LTX-2.

        Args:
            image_path: Optional path to input image for image-to-video
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (used to compute num_frames if not set)
            height: Video height (uses model default if None)
            width: Video width (uses model default if None)
            num_frames: Number of frames to generate
            frame_rate: Frames per second for output video
            seed: Random seed for reproducibility
            num_inference_steps: Number of diffusion sampling steps
            cfg_guidance_scale: Classifier-free guidance scale
            negative_prompt: Negative prompt for generation
            enhance_prompt: Whether to enhance the prompt using Gemma
            output_filename: Optional output filename (auto-generated if None)
            **kwargs: Additional parameters

        Returns:
            Dictionary with generation results and metadata
        """
        start_time = time.time()
        generation_id = f"ltx2_{uuid.uuid4().hex[:8]}"

        # Set defaults
        h = height if height is not None else self.resolution[0]
        w = width if width is not None else self.resolution[1]
        fps = frame_rate if frame_rate is not None else self.frame_rate
        
        # Store original dimensions for logging
        orig_h, orig_w = h, w
        
        # Round height and width to multiples of 32 (required by LTX-2)
        h = ((h + 31) // 32) * 32
        w = ((w + 31) // 32) * 32
        
        if h != orig_h or w != orig_w:
            print(f"Resolution: {orig_w}x{orig_h} -> {w}x{h} (rounded to multiple of 32)")
        
        # Calculate num_frames from duration if not specified
        if num_frames is None:
            num_frames = int(duration * fps) + 1  # +1 for initial frame
            # LTX-2 has specific frame requirements (must be compatible with VAE)
            num_frames = max(17, min(num_frames, 257))  # Clamp to valid range

        # Set seed
        if seed is None:
            seed = int(time.time()) % 2147483647

        # Prepare image conditioning with preprocessing
        images: list[tuple[str, int, float]] = []
        if image_path is not None:
            image_path = Path(image_path)
            if image_path.exists():
                # Preprocess image: scale proportionally to target width (no left/right cropping)
                preprocessed_path = preprocess_image_for_ltx2(
                    image_path=image_path,
                    target_width=w,
                    target_height=h,
                    output_dir=self.output_dir / "_preprocessed"
                )
                # (image_path, frame_index, strength)
                images = [(str(preprocessed_path), 0, 1.0)]
            else:
                print(f"Warning: Image not found: {image_path}")

        # Load pipeline
        pipeline = self._load_pipeline()

        try:
            # Build guider params (replaces cfg_guidance_scale in newer LTX-2 API)
            from ltx_core.components.guiders import MultiModalGuiderParams
            guider_params = MultiModalGuiderParams(cfg_scale=cfg_guidance_scale)

            # Run generation
            video, audio = pipeline(
                prompt=text_prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=h,
                width=w,
                num_frames=num_frames,
                frame_rate=fps,
                num_inference_steps=num_inference_steps,
                video_guider_params=guider_params,
                audio_guider_params=guider_params,
                images=images,
                enhance_prompt=enhance_prompt,
            )

            # Generate output filename
            if output_filename is None:
                output_filename = f"ltx2_{generation_id}.mp4"

            output_path = self.output_dir / output_filename

            # Import and use LTX-2's encode_video function
            from ltx_pipelines.utils.media_io import encode_video
            from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE

            encode_video(
                video=video,
                fps=fps,
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=str(output_path),
                video_chunks_number=1,
            )

            duration_seconds = time.time() - start_time

            return {
                "success": True,
                "video_path": str(output_path),
                "error": None,
                "duration_seconds": duration_seconds,
                "generation_id": generation_id,
                "model": self.model_id,
                "status": "success",
                "metadata": {
                    "text_prompt": text_prompt,
                    "negative_prompt": negative_prompt,
                    "image_path": str(image_path) if image_path else None,
                    "height": h,
                    "width": w,
                    "num_frames": num_frames,
                    "frame_rate": fps,
                    "seed": seed,
                    "num_inference_steps": num_inference_steps,
                    "cfg_guidance_scale": cfg_guidance_scale,
                    "enhance_prompt": enhance_prompt,
                }
            }

        except Exception as e:
            duration_seconds = time.time() - start_time
            return {
                "success": False,
                "video_path": None,
                "error": str(e),
                "duration_seconds": duration_seconds,
                "generation_id": generation_id,
                "model": self.model_id,
                "status": "failed",
                "metadata": {
                    "text_prompt": text_prompt,
                    "image_path": str(image_path) if image_path else None,
                }
            }


class LTX2Wrapper(ModelWrapper):
    """
    Wrapper for LTX2Service to match VMEvalKit's standard interface.
    """

    def __init__(
        self,
        model: str = "LTX-2",
        output_dir: str = "./outputs",
        device: str = "cuda",
        enable_fp8: bool = True,
        **kwargs
    ):
        """Initialize LTX-2 wrapper."""
        self.model = model
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs

        # Create LTX2Service instance
        self.service = LTX2Service(
            model_id=model,
            output_dir=str(self._output_dir),
            device=device,
            enable_fp8=enable_fp8,
            **kwargs
        )

    @property
    def output_dir(self) -> Path:
        """Get the current output directory."""
        return self._output_dir

    @output_dir.setter
    def output_dir(self, value: Union[str, Path]):
        """Set the output directory and update the service's output_dir too."""
        self._output_dir = Path(value)
        self._output_dir.mkdir(exist_ok=True, parents=True)
        self.service.output_dir = self._output_dir

    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 4.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video using LTX-2 (matches VMEvalKit interface)."""
        kwargs.pop("question_data", None)
        return self.service.generate(
            image_path=image_path,
            text_prompt=text_prompt,
            duration=duration,
            output_filename=output_filename,
            **kwargs
        )