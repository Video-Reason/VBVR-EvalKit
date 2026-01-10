"""
DynamiCrafter Inference Service for VMEvalKit

Wrapper for the DynamiCrafter model (submodules/DynamiCrafter) to integrate with VMEvalKit's
unified inference interface. Supports image animation using video diffusion priors.
"""

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Union
from collections import OrderedDict

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat, rearrange
from pytorch_lightning import seed_everything

from .base import ModelWrapper

# Add DynamiCrafter submodule to path
DYNAMICRAFTER_PATH = Path(__file__).parent.parent.parent / "submodules" / "DynamiCrafter"
VMEVAL_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(DYNAMICRAFTER_PATH))

# Import DynamiCrafter modules
from lvdm.models.samplers.ddim import DDIMSampler
from utils.utils import instantiate_from_config


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    """Load model weights from checkpoint."""
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in list(state_dict.keys()):
        state_dict = state_dict["state_dict"]
        # Handle key renaming for 256x256 model compatibility
        new_pl_sd = OrderedDict()
        for k, v in state_dict.items():
            new_pl_sd[k] = v
        for k in list(new_pl_sd.keys()):
            if "framestride_embed" in k:
                new_key = k.replace("framestride_embed", "fps_embedding")
                new_pl_sd[new_key] = new_pl_sd[k]
                del new_pl_sd[k]
        model.load_state_dict(new_pl_sd, strict=False)
    else:
        # DeepSpeed checkpoint format
        new_pl_sd = OrderedDict()
        for key in state_dict['module'].keys():
            new_pl_sd[key[16:]] = state_dict['module'][key]
        model.load_state_dict(new_pl_sd, strict=False)
    return model


def get_latent_z(model: torch.nn.Module, videos: torch.Tensor) -> torch.Tensor:
    """Encode video frames to latent space."""
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z


def batch_ddim_sampling(
    model: torch.nn.Module,
    cond: Dict[str, Any],
    noise_shape: list,
    n_samples: int = 1,
    ddim_steps: int = 50,
    ddim_eta: float = 1.0,
    cfg_scale: float = 7.5,
    **kwargs
) -> torch.Tensor:
    """Run DDIM sampling for video generation."""
    ddim_sampler = DDIMSampler(model)
    uncond_type = model.uncond_type
    batch_size = noise_shape[0]
    fs = cond["fs"]
    del cond["fs"]
    
    # Timestep spacing depends on resolution
    if noise_shape[-1] == 32:  # 256 resolution
        timestep_spacing = "uniform"
        guidance_rescale = 0.0
    else:  # 512, 1024 resolution
        timestep_spacing = "uniform_trailing"
        guidance_rescale = 0.7

    # Construct unconditional guidance
    uc = None
    if cfg_scale != 1.0:
        if uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif uncond_type == "zero_embed":
            c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
            uc_emb = torch.zeros_like(c_emb)

        # Process image embedding token
        if hasattr(model, 'embedder'):
            uc_img = torch.zeros(noise_shape[0], 3, 224, 224).to(model.device)
            uc_img = model.embedder(uc_img)
            uc_img = model.image_proj_model(uc_img)
            uc_emb = torch.cat([uc_emb, uc_img], dim=1)

        if isinstance(cond, dict):
            uc = {key: cond[key] for key in cond.keys()}
            uc.update({'c_crossattn': [uc_emb]})
        else:
            uc = uc_emb

    batch_variants = []
    for _ in range(n_samples):
        kwargs.update({"clean_cond": True})
        samples, _ = ddim_sampler.sample(
            S=ddim_steps,
            conditioning=cond,
            batch_size=noise_shape[0],
            shape=noise_shape[1:],
            verbose=False,
            unconditional_guidance_scale=cfg_scale,
            unconditional_conditioning=uc,
            eta=ddim_eta,
            temporal_length=noise_shape[2],
            x_T=None,
            fs=fs,
            timestep_spacing=timestep_spacing,
            guidance_rescale=guidance_rescale,
            **kwargs
        )
        # Reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)

    # batch, <samples>, c, t, h, w
    batch_variants = torch.stack(batch_variants, dim=1)
    return batch_variants


class DynamiCrafterService:
    """
    Service class for DynamiCrafter inference integration.
    Loads the model once and runs inference in-process.
    """

    # Model configuration mapping
    CONFIG_MAPPING = {
        "dynamicrafter-256": {
            "config": "configs/inference_256_v1.0.yaml",
            "resolution": (256, 256),
            "frame_stride": 3,  # For 256 model: controls motion (1-6, larger = more motion)
        },
        "dynamicrafter-512": {
            "config": "configs/inference_512_v1.0.yaml",
            "resolution": (320, 512),
            "frame_stride": 24,  # For 512 model: FPS control (15-30, smaller = more motion)
        },
        "dynamicrafter-1024": {
            "config": "configs/inference_1024_v1.0.yaml",
            "resolution": (576, 1024),
            "frame_stride": 10,  # For 1024 model: FPS control (5-15, smaller = more motion)
        },
    }

    def __init__(
        self,
        model_id: str = "dynamicrafter-512",
        output_dir: str = "./outputs",
        device: str = "cuda",
        **kwargs
    ):
        """
        Initialize DynamiCrafter service and load model.

        Args:
            model_id: DynamiCrafter model variant
            output_dir: Directory to save generated videos
            device: Device to run inference on
            **kwargs: Additional parameters
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.device = device
        self.kwargs = kwargs

        # Validate model_id
        assert model_id in self.CONFIG_MAPPING, (
            f"Unknown model_id: {model_id}. "
            f"Available: {list(self.CONFIG_MAPPING.keys())}"
        )

        self.model_config = self.CONFIG_MAPPING[model_id]
        self.resolution = self.model_config["resolution"]  # (H, W)
        self.frame_stride = self.model_config["frame_stride"]

        # Build paths
        config_path = DYNAMICRAFTER_PATH / self.model_config["config"]
        ckpt_dir = model_id.replace('-', '_')  # dynamicrafter-512 -> dynamicrafter_512
        ckpt_path = VMEVAL_ROOT / "weights" / "dynamicrafter" / f"{ckpt_dir}_v1" / "model.ckpt"

        assert config_path.exists(), f"Config not found: {config_path}"
        assert ckpt_path.exists(), (
            f"Checkpoint not found: {ckpt_path}\n"
            f"Please download the model weights using the setup script."
        )

        # Load model
        print(f"Loading DynamiCrafter model: {model_id}")
        config = OmegaConf.load(str(config_path))
        model_config = config.pop("model", OmegaConf.create())
        model_config['params']['unet_config']['params']['use_checkpoint'] = False

        self.model = instantiate_from_config(model_config)
        self.model = load_model_checkpoint(self.model, str(ckpt_path))
        self.model = self.model.to(self.device)
        self.model.eval()

        # Pre-build transform for image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(min(self.resolution)),
            transforms.CenterCrop(self.resolution),
        ])

        self.save_fps = 8
        print(f"DynamiCrafter model loaded: {model_id} @ {self.resolution}")

    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input."""
        image = image.convert("RGB")
        img_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        img_tensor = (img_tensor / 255.0 - 0.5) * 2.0
        img_tensor = self.transform(img_tensor)
        return img_tensor

    def _save_video(
        self,
        batch_tensors: torch.Tensor,
        output_path: Path,
        fps: int = 8
    ) -> None:
        """Save generated video tensor to file."""
        # batch_tensors: b, samples, c, t, h, w
        n_samples = batch_tensors.shape[1]
        vid_tensor = batch_tensors[0]  # Take first batch item
        video = vid_tensor.detach().cpu()
        video = torch.clamp(video.float(), -1.0, 1.0)
        video = video.permute(2, 0, 1, 3, 4)  # t, n, c, h, w

        frame_grids = [
            torchvision.utils.make_grid(framesheet, nrow=int(n_samples))
            for framesheet in video
        ]
        grid = torch.stack(frame_grids, dim=0)  # t, 3, h, w
        grid = (grid + 1.0) / 2.0
        grid = (grid * 255).to(torch.uint8).permute(0, 2, 3, 1)  # t, h, w, c

        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchvision.io.write_video(
            str(output_path),
            grid,
            fps=fps,
            video_codec='h264',
            options={'crf': '10'}
        )

    @torch.no_grad()
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 2.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: int = 16,
        fps: int = 8,
        seed: Optional[int] = None,
        ddim_steps: int = 50,
        ddim_eta: float = 1.0,
        cfg_scale: float = 7.5,
        frame_stride: Optional[int] = None,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video from image and text prompt using DynamiCrafter.

        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (used to compute num_frames if not set)
            height: Video height (uses model default if None)
            width: Video width (uses model default if None)
            num_frames: Number of frames to generate
            fps: Frames per second for output video
            seed: Random seed for reproducibility
            ddim_steps: Number of DDIM sampling steps
            ddim_eta: DDIM eta parameter
            cfg_scale: Classifier-free guidance scale
            frame_stride: Frame stride / FPS control (uses model default if None)
            output_filename: Optional output filename (auto-generated if None)
            **kwargs: Additional parameters

        Returns:
            Dictionary with generation results and metadata
        """
        start_time = time.time()
        generation_id = f"dynamicrafter_{uuid.uuid4().hex[:8]}"

        # Validate image path
        image_path = Path(image_path)
        assert image_path.exists(), f"Input image not found: {image_path}"

        # Set seed for reproducibility
        if seed is not None:
            seed_everything(seed)
        else:
            seed = int(time.time()) % 100000
            seed_everything(seed)

        # Use model default resolution if not specified
        h = height if height is not None else self.resolution[0]
        w = width if width is not None else self.resolution[1]
        fs = frame_stride if frame_stride is not None else self.frame_stride

        # Load and preprocess image
        image = Image.open(image_path)
        img_tensor = self._preprocess_image(image).to(self.device)

        # Prepare model inputs
        batch_size = 1
        channels = self.model.model.diffusion_model.out_channels
        frames = self.model.temporal_length if hasattr(self.model, 'temporal_length') else num_frames
        latent_h, latent_w = h // 8, w // 8
        noise_shape = [batch_size, channels, frames, latent_h, latent_w]

        # Encode image to latent
        videos = img_tensor.unsqueeze(0)  # b, c, h, w
        z = get_latent_z(self.model, videos.unsqueeze(2))  # b, c, 1, h, w
        img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)

        # Get text embedding
        with torch.cuda.amp.autocast():
            text_emb = self.model.get_learned_conditioning([text_prompt])

            # Get image embedding
            cond_images = self.model.embedder(img_tensor.unsqueeze(0))  # b, l, c
            img_emb = self.model.image_proj_model(cond_images)

            # Combine text and image embeddings
            imtext_cond = torch.cat([text_emb, img_emb], dim=1)

            # Build conditioning dict
            fs_tensor = torch.tensor([fs], dtype=torch.long, device=self.device)
            cond = {
                "c_crossattn": [imtext_cond],
                "fs": fs_tensor,
                "c_concat": [img_tensor_repeat]
            }

            # Run DDIM sampling
            batch_samples = batch_ddim_sampling(
                self.model,
                cond,
                noise_shape,
                n_samples=1,
                ddim_steps=ddim_steps,
                ddim_eta=ddim_eta,
                cfg_scale=cfg_scale,
            )

        # Generate output filename
        if output_filename is None:
            output_filename = "video.mp4"

        output_path = self.output_dir / output_filename

        # Save video
        self._save_video(batch_samples, output_path, fps=fps)

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
                "image_path": str(image_path),
                "height": h,
                "width": w,
                "num_frames": frames,
                "fps": fps,
                "seed": seed,
                "ddim_steps": ddim_steps,
                "ddim_eta": ddim_eta,
                "cfg_scale": cfg_scale,
                "frame_stride": fs,
            }
        }


class DynamiCrafterWrapper(ModelWrapper):
    """
    Wrapper for DynamiCrafterService to match VMEvalKit's standard interface.
    """

    def __init__(
        self,
        model: str,
        output_dir: str = "./outputs",
        device: str = "cuda",
        **kwargs
    ):
        """Initialize DynamiCrafter wrapper."""
        self.model = model
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs

        # Create DynamiCrafterService instance
        self.service = DynamiCrafterService(
            model_id=model,
            output_dir=str(self._output_dir),
            device=device,
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
        duration: float = 2.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using DynamiCrafter (matches VMEvalKit interface).

        Args:
            image_path: Path to input image
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds
            output_filename: Optional output filename
            **kwargs: Additional parameters (ddim_steps, cfg_scale, seed, etc.)

        Returns:
            Dictionary with generation results
        """
        return self.service.generate(
            image_path=image_path,
            text_prompt=text_prompt,
            duration=duration,
            output_filename=output_filename,
            **kwargs
        )
