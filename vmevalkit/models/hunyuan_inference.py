"""HunyuanVideo-I2V Inference Service for VMEvalKit"""

import os
import sys
import subprocess
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .base import ModelWrapper
import json
import time

HUNYUAN_PATH = Path(__file__).parent.parent.parent / "submodules" / "HunyuanVideo-I2V"
sys.path.insert(0, str(HUNYUAN_PATH))

# HuggingFace cache paths for shared checkpoints
HF_HOME = os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
HF_HUB_CACHE = Path(HF_HOME) / "hub"


def _get_hf_snapshot_path(repo_id: str) -> Optional[Path]:
    """Get the snapshot path for a HuggingFace model from cache."""
    model_dir = HF_HUB_CACHE / f"models--{repo_id.replace('/', '--')}"
    if not model_dir.exists():
        return None
    snapshots_dir = model_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    # Get the first (usually only) snapshot
    snapshots = list(snapshots_dir.iterdir())
    if snapshots:
        return snapshots[0]
    return None


class HunyuanVideoService:
    """Service class for HunyuanVideo-I2V inference integration."""
    
    def __init__(
        self,
        model_id: str = "hunyuan-video-i2v",
        output_dir: str = "./outputs",
        model_python_interpreter: str = None,
        **kwargs
    ):
        self.model_id = model_id
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.model_python_interpreter = model_python_interpreter or sys.executable
        self.kwargs = kwargs
        
        self.inference_script = HUNYUAN_PATH / "sample_image2video.py"
        if not self.inference_script.exists():
            raise FileNotFoundError(
                f"HunyuanVideo-I2V inference script not found at {self.inference_script}.\n"
                f"Please initialize submodule:\n"
                f"cd {HUNYUAN_PATH.parent} && git submodule update --init HunyuanVideo-I2V"
            )

    def _run_hunyuan_inference(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        height: int = 720,
        width: int = 1280,
        video_length: int = 129,
        seed: Optional[int] = None,
        use_i2v_stability: bool = True,
        flow_shift: float = 7.0,
        **kwargs
    ) -> Dict[str, Any]:
        """Run HunyuanVideo-I2V inference using subprocess."""
        start_time = time.time()
        timestamp = int(start_time)
        
        output_dir = self.output_dir
        output_dir.mkdir(exist_ok=True, parents=True)
        
        if height >= 720:
            i2v_resolution = "720p"
        elif height >= 540:
            i2v_resolution = "540p"
        else:
            i2v_resolution = "360p"
        
        # Try to get model_base from HuggingFace cache first, fallback to local ckpts
        default_model_base = None
        hf_model_path = _get_hf_snapshot_path("tencent/HunyuanVideo-I2V")
        if hf_model_path and (hf_model_path / "hunyuan-video-i2v-720p").exists():
            default_model_base = str(hf_model_path / "hunyuan-video-i2v-720p")
            print(f"Using HunyuanVideo model from HF cache: {default_model_base}")
        else:
            default_model_base = str(HUNYUAN_PATH / "ckpts" / "hunyuan-video-i2v-720p")
            print(f"Using HunyuanVideo model from local path: {default_model_base}")
        
        model_base = kwargs.get('model_base', default_model_base)
        model_type = kwargs.get('model_type', 'HYVideo-T/2')
        infer_steps = kwargs.get('infer_steps', 50)
        embedded_cfg_scale = kwargs.get('embedded_cfg_scale', 6.0)
        
        # Get the absolute path for i2v-dit-weight
        # HunyuanVideo expects the transformer weights path
        i2v_dit_weight = Path(model_base) / "transformers" / "mp_rank_00_model_states.pt"
        if not i2v_dit_weight.exists():
            # Try alternative path structure
            i2v_dit_weight = Path(model_base).parent / "hunyuan-video-i2v-720p" / "transformers" / "mp_rank_00_model_states.pt"
        
        num_gpus = kwargs.get('num_gpus', 1)
        use_cpu_offload = kwargs.get('use_cpu_offload', True)
        
        cmd = [
            self.model_python_interpreter,
            str(HUNYUAN_PATH / "sample_image2video.py"),
        ]
        
        cmd.extend([
            "--prompt", text_prompt,
            "--i2v-image-path", str(image_path),
            "--video-size", str(width), str(height),
            "--video-length", str(video_length),
            "--save-path", str(output_dir),
            "--model", model_type,
            "--model-base", model_base,
            "--i2v-dit-weight", str(i2v_dit_weight),
            "--i2v-mode",
            "--i2v-resolution", i2v_resolution,
            "--infer-steps", str(infer_steps),
            "--embedded-cfg-scale", str(embedded_cfg_scale),
        ])
        
        if use_cpu_offload and num_gpus == 1:
            cmd.append("--use-cpu-offload")
        
        if use_i2v_stability:
            cmd.append("--i2v-stability")
        
        if flow_shift:
            cmd.extend(["--flow-shift", str(flow_shift)])
        
        cmd.append("--flow-reverse")
        
        if seed is not None:
            cmd.extend(["--seed", str(seed)])
        else:
            cmd.extend(["--seed", "0"])
            
        skip_keys = ['use_i2v_stability', 'flow_shift', 'model_base', 'model_type', 
                     'infer_steps', 'embedded_cfg_scale', 'question_data', 'duration', 
                     'output_filename', 'fps', 'num_gpus', 'use_cpu_offload',
                     'num_frames', 'height', 'width']
        for key, value in kwargs.items():
            if value is not None and key not in skip_keys:
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])

        # Check for CLIP text encoder (text_encoder_2) - try HF cache first, then local ckpts
        clip_text_encoder_dir = _get_hf_snapshot_path("openai/clip-vit-large-patch14")
        if clip_text_encoder_dir is None or not clip_text_encoder_dir.exists():
            # Fallback to local ckpts directory
            clip_text_encoder_dir = HUNYUAN_PATH / "ckpts" / "text_encoder_2"
        
        if not clip_text_encoder_dir.exists() or not any(clip_text_encoder_dir.iterdir()):
            raise FileNotFoundError(
                f"Missing CLIP text encoder. Expected at HF cache (openai/clip-vit-large-patch14) "
                f"or local: {HUNYUAN_PATH / 'ckpts' / 'text_encoder_2'}\n"
                f"Run: hf download openai/clip-vit-large-patch14"
            )
        
        # Check for LLaVA text encoder (text_encoder_i2v) - try HF cache first, then local ckpts
        llava_text_encoder_dir = _get_hf_snapshot_path("xtuner/llava-llama-3-8b-v1_1-transformers")
        if llava_text_encoder_dir is None or not llava_text_encoder_dir.exists():
            # Fallback to local ckpts directory
            llava_text_encoder_dir = HUNYUAN_PATH / "ckpts" / "text_encoder_i2v"
        
        if not llava_text_encoder_dir.exists():
            raise FileNotFoundError(
                f"Missing LLaVA text encoder. Expected at HF cache (xtuner/llava-llama-3-8b-v1_1-transformers) "
                f"or local: {HUNYUAN_PATH / 'ckpts' / 'text_encoder_i2v'}\n"
                f"Run: hf download xtuner/llava-llama-3-8b-v1_1-transformers"
            )
        
        # Set text encoder paths in environment for HunyuanVideo
        try:
            env = os.environ.copy()
            # Set MODEL_BASE to where the main model weights are
            hf_model_path = _get_hf_snapshot_path("tencent/HunyuanVideo-I2V")
            if hf_model_path and hf_model_path.exists():
                env["MODEL_BASE"] = str(hf_model_path)
                
                # HunyuanVideo's constants.py reads MODEL_BASE at import time and expects
                # text_encoder_i2v and text_encoder_2 to be inside MODEL_BASE.
                # Create symlinks in the HF cache to make this work.
                text_encoder_i2v_link = hf_model_path / "text_encoder_i2v"
                if not text_encoder_i2v_link.exists() and llava_text_encoder_dir.exists():
                    try:
                        text_encoder_i2v_link.symlink_to(llava_text_encoder_dir)
                        print(f"Created symlink: {text_encoder_i2v_link} -> {llava_text_encoder_dir}")
                    except (OSError, FileExistsError) as e:
                        print(f"Warning: Could not create symlink for text_encoder_i2v: {e}")
                
                text_encoder_2_link = hf_model_path / "text_encoder_2"
                if not text_encoder_2_link.exists() and clip_text_encoder_dir.exists():
                    try:
                        text_encoder_2_link.symlink_to(clip_text_encoder_dir)
                        print(f"Created symlink: {text_encoder_2_link} -> {clip_text_encoder_dir}")
                    except (OSError, FileExistsError) as e:
                        print(f"Warning: Could not create symlink for text_encoder_2: {e}")
            else:
                env["MODEL_BASE"] = str(HUNYUAN_PATH / "ckpts")
            env["TEXT_ENCODER_2"] = str(clip_text_encoder_dir)
            env["TEXT_ENCODER_I2V"] = str(llava_text_encoder_dir)
            
            if num_gpus > 1:
                env["ALLOW_RESIZE_FOR_SP"] = "1"

            result = subprocess.run(
                cmd,
                cwd=str(HUNYUAN_PATH),
                env=env,
                capture_output=True,
                text=True,
                timeout=7200
            )
            
            success = result.returncode == 0
            error_msg = result.stderr if result.returncode != 0 else None
            
            output_video = None
            if success and output_dir.exists():
                # Find all video files and sort by modification time (newest first)
                video_files = list(output_dir.glob("**/*.mp4"))
                if video_files:
                    # Get the most recently created/modified video file
                    source_video = max(video_files, key=lambda p: p.stat().st_mtime)
                    final_video_path = output_dir / "video.mp4"
                    
                    # Only move if it's not already named video.mp4
                    if source_video != final_video_path:
                        # If video.mp4 already exists, remove it first
                        if final_video_path.exists():
                            final_video_path.unlink()
                        shutil.move(str(source_video), str(final_video_path))
                    
                    # Clean up subdirectories created by Hunyuan
                    for item in output_dir.iterdir():
                        if item.is_dir():
                            shutil.rmtree(item)
                    
                    output_video = str(final_video_path)
                else:
                    success = False
                    error_msg = f"Video generation succeeded but no .mp4 file found in {output_dir}"
            
        except subprocess.TimeoutExpired:
            success = False
            error_msg = "HunyuanVideo inference timed out"
            output_video = None
        except Exception as e:
            success = False
            error_msg = f"HunyuanVideo inference failed: {str(e)}"
            output_video = None
        
        duration = time.time() - start_time
        
        return {
            "success": success,
            "video_path": output_video,
            "error": error_msg,
            "duration_seconds": duration,
            "generation_id": f"hunyuan_{timestamp}",
            "model": self.model_id,
            "status": "success" if success else "failed",
            "metadata": {
                "text_prompt": text_prompt,
                "image_path": str(image_path),
                "height": height,
                "width": width,
                "video_length": video_length,
                "seed": seed,
                "use_i2v_stability": use_i2v_stability,
                "flow_shift": flow_shift,
                "stdout": result.stdout if 'result' in locals() else None,
                "stderr": result.stderr if 'result' in locals() else None,
            }
        }

    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        height: int = 720,
        width: int = 1280,
        seed: Optional[int] = None,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video from image and text prompt."""
        num_frames = kwargs.pop('num_frames', None)
        fps = kwargs.get('fps', 24)
        
        if num_frames is not None:
            video_length = num_frames
        else:
            video_length = max(1, int(duration * fps))
        
        original_length = video_length
        
        # HunyuanVideo requires: (video_length - 1) % 4 == 0
        # Valid lengths: 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61, 65, 69, 73, 77, 81, 85, 89, 93, 97, 101, 105, 109, 113, 117, 121, 125, 129
        # Formula: video_length = 4*n + 1, where n >= 0
        remainder = (video_length - 1) % 4
        if remainder != 0:
            # Round up to next valid length
            video_length = video_length + (4 - remainder)
        
        # Cap at maximum
        video_length = min(video_length, 129)
        
        if video_length != original_length:
            print(f"    Note: Adjusted frames from {original_length} to {video_length} (HunyuanVideo: (n-1) divisible by 4, max 129)")
        print(f"    Video generation: num_frames={video_length}, fps={fps}")
        
        image_path = Path(image_path)
        if not image_path.exists():
            return {
                "success": False,
                "video_path": None,
                "error": f"Input image not found: {image_path}",
                "duration_seconds": 0,
                "generation_id": f"hunyuan_error_{int(time.time())}",
                "model": self.model_id,
                "status": "failed",
                "metadata": {"text_prompt": text_prompt, "image_path": str(image_path)},
            }
        
        result = self._run_hunyuan_inference(
            image_path=image_path,
            text_prompt=text_prompt,
            height=height,
            width=width,
            video_length=video_length,
            seed=seed,
            **kwargs
        )
        
        if output_filename and result["success"] and result["video_path"]:
            old_path = Path(result["video_path"])
            new_path = self.output_dir / output_filename
            if old_path.exists():
                old_path.rename(new_path)
                result["video_path"] = str(new_path)
        
        return result


class HunyuanVideoWrapper(ModelWrapper):
    """Wrapper for HunyuanVideoService to match VMEvalKit's standard interface."""
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./outputs",
        **kwargs
    ):
        super().__init__(model, output_dir, **kwargs)
        self.hunyuan_service = HunyuanVideoService(
            model_id=model, 
            output_dir=str(self.output_dir), 
            model_python_interpreter=self.get_model_python_interpreter(),
            **kwargs
        )
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate video using HunyuanVideo-I2V."""
        self.hunyuan_service.output_dir = self.output_dir
        return self.hunyuan_service.generate(
            image_path=image_path,
            text_prompt=text_prompt,
            duration=duration,
            output_filename=output_filename,
            **kwargs
        )
