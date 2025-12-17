"""
Morphic Inference Service for VMEvalKit

Wrapper for the Morphic Frames-to-Video model (submodules/morphic-frames-to-video) to integrate with VMEvalKit's
unified inference interface. Supports high-quality frame-to-video interpolation using Wan2.2.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .base import ModelWrapper
import time

# Add Morphic submodule to path
MORPHIC_PATH = Path(__file__).parent.parent.parent / "submodules" / "morphic-frames-to-video"
sys.path.insert(0, str(MORPHIC_PATH))


class MorphicService:
    """
    Service class for Morphic Frames-to-Video inference integration.
    """
    
    def __init__(
        self,
        model_id: str = "morphic-frames-to-video",
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """
        Initialize Morphic Frames-to-Video service.
        
        Args:
            model_id: Morphic model variant (currently only one available)
            output_dir: Directory to save generated videos
            **kwargs: Additional parameters including:
                - size: Video size (default: "1280*720")
                - frame_num: Number of frames (default: 81)
                - nproc_per_node: Number of GPUs (default: 8)
        """
        self.model_id = model_id
        self.output_dir = Path(output_dir).resolve()  # Always use absolute path
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Get configuration from kwargs or defaults
        self.size = kwargs.get('size', "1280*720")
        self.frame_num = kwargs.get('frame_num', 81)
        
        # Auto-detect number of available GPUs
        import torch
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.nproc_per_node = kwargs.get('nproc_per_node', available_gpus)
        self.ulysses_size = kwargs.get('ulysses_size', self.nproc_per_node)
        
        # Get weight paths from environment variables or kwargs
        self.wan2_ckpt_dir = kwargs.get(
            'wan2_ckpt_dir',
            os.getenv('MORPHIC_WAN2_CKPT_DIR', './weights/wan/Wan2.2-I2V-A14B')
        )
        self.lora_weights_path = kwargs.get(
            'lora_weights_path',
            os.getenv(
                'MORPHIC_LORA_WEIGHTS_PATH',
                './weights/morphic/lora_interpolation_high_noise_final.safetensors'
            )
        )
        
        # Override nproc_per_node from environment if set
        env_nproc = os.getenv('MORPHIC_NPROC_PER_NODE')
        if env_nproc:
            self.nproc_per_node = int(env_nproc)
            self.ulysses_size = self.nproc_per_node
        
        # Validate paths
        self._validate_paths()
    
    def _validate_paths(self):
        """
        Validate that all required paths exist.
        
        Raises:
            FileNotFoundError: If any required path is missing
        """
        errors = []
        
        # Check submodule
        generate_script = MORPHIC_PATH / "generate.py"
        if not generate_script.exists():
            errors.append(
                f"Morphic submodule not found at {MORPHIC_PATH}.\n"
                f"Please initialize submodule:\n"
                f"git submodule update --init --recursive submodules/morphic-frames-to-video"
            )
        
        # Check Wan2.2 weights directory
        wan2_path = Path(self.wan2_ckpt_dir)
        if not wan2_path.exists():
            errors.append(
                f"Wan2.2 weights directory not found at {self.wan2_ckpt_dir}.\n"
                f"Please download weights:\n"
                f"huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir ./weights/wan/Wan2.2-I2V-A14B"
            )
        
        # Check LoRA weights file
        lora_path = Path(self.lora_weights_path)
        if not lora_path.exists():
            errors.append(
                f"LoRA weights file not found at {self.lora_weights_path}.\n"
                f"Please download weights:\n"
                f"huggingface-cli download morphic/Wan2.2-frames-to-video --local-dir ./weights/morphic"
            )
        
        if errors:
            raise FileNotFoundError("\n".join(errors))
    
    def _run_morphic_inference(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        final_image_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run Morphic Frames-to-Video inference using torchrun.
        
        Args:
            image_path: Path to input image (first frame)
            text_prompt: Text prompt for video generation
            final_image_path: Path to final frame image
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with inference results and metadata
        """
        start_time = time.time()
        
        # Generate output filename
        timestamp = int(time.time())
        output_filename = f"morphic_{timestamp}.mp4"
        output_path = (self.output_dir / output_filename).resolve()  # Convert to absolute path
        
        # Build torchrun command
        cmd = [
            "torchrun",
            f"--nproc_per_node={self.nproc_per_node}",
            str(MORPHIC_PATH / "generate.py"),
            "--task", "i2v-A14B",
            "--size", self.size,
            "--frame_num", str(self.frame_num),
            "--ckpt_dir", str(self.wan2_ckpt_dir),
            "--high_noise_lora_weights_path", str(self.lora_weights_path),
            "--dit_fsdp",
            "--t5_fsdp",
            "--ulysses_size", str(self.ulysses_size),
            "--image", str(image_path),
            "--prompt", text_prompt,
            "--img_end", str(final_image_path),
            "--save_file", str(output_path),  # Directly specify output path
        ]
        
        # Add seed if provided
        if kwargs.get('seed') is not None:
            cmd.extend(["--seed", str(kwargs['seed'])])
        
        try:
            # Change to Morphic directory and run inference
            result = subprocess.run(
                cmd,
                cwd=str(MORPHIC_PATH),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for distributed large model inference
            )
            
            success = result.returncode == 0
            error_msg = result.stderr if result.returncode != 0 else None
            
        except subprocess.TimeoutExpired:
            success = False
            error_msg = "Morphic inference timed out (exceeded 30 minutes)"
        except Exception as e:
            success = False
            error_msg = f"Morphic inference failed: {str(e)}"
        
        duration = time.time() - start_time
        
        return {
            "success": success,
            "video_path": str(output_path) if success and output_path.exists() else None,
            "error": error_msg,
            "duration_seconds": duration,
            "generation_id": f"morphic_{int(time.time())}",
            "model": self.model_id,
            "status": "success" if success else "failed",
            "metadata": {
                "text_prompt": text_prompt,
                "image_path": str(image_path),
                "final_image_path": str(final_image_path),
                "size": self.size,
                "frame_num": self.frame_num,
                "nproc_per_node": self.nproc_per_node,
                "stdout": result.stdout if 'result' in locals() else None,
                "stderr": result.stderr if 'result' in locals() else None,
            }
        }
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        final_image_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video from image frames and text prompt.
        
        Args:
            image_path: Path to input image (first frame)
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds (not directly used, frame_num controls length)
            output_filename: Optional output filename (auto-generated if None)
            final_image_path: Path to final frame image (required for Morphic)
            **kwargs: Additional parameters passed to Morphic
            
        Returns:
            Dictionary with generation results and metadata
        """
        # Convert to Path objects (let Python/PyTorch raise errors if files don't exist)
        image_path = Path(image_path)
        
        # final_image_path is required for Morphic - check only once at the beginning
        if not final_image_path:
            raise ValueError("Morphic requires final_image_path parameter")
        
        final_image_path = Path(final_image_path)
        
        # Run inference
        result = self._run_morphic_inference(
            image_path=image_path,
            text_prompt=text_prompt,
            final_image_path=final_image_path,
            **kwargs
        )
        
        # Handle custom output filename
        if output_filename and result["success"] and result["video_path"]:
            old_path = Path(result["video_path"])
            new_path = self.output_dir / output_filename
            if old_path.exists():
                old_path.rename(new_path)
                result["video_path"] = str(new_path)
            else:
                print(f"Warning: old_path does not exist: {old_path}")
        
        return result


# Wrapper class to match VMEvalKit's interface pattern
class MorphicWrapper(ModelWrapper):
    """
    Wrapper for MorphicService to match VMEvalKit's standard interface.
    """
    
    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        **kwargs
    ):
        """Initialize Morphic wrapper."""
        self.model = model
        self._output_dir = Path(output_dir).resolve()  # Always use absolute path
        self._output_dir.mkdir(exist_ok=True, parents=True)
        self.kwargs = kwargs
        
        # Create MorphicService instance
        self.morphic_service = MorphicService(
            model_id=model, output_dir=str(self._output_dir), **kwargs
        )
    
    @property
    def output_dir(self) -> Path:
        """Get the current output directory."""
        return self._output_dir
    
    @output_dir.setter
    def output_dir(self, value: Union[str, Path]):
        """Set the output directory and update the service's output_dir too."""
        self._output_dir = Path(value).resolve()
        self._output_dir.mkdir(exist_ok=True, parents=True)
        # Also update the service's output_dir
        self.morphic_service.output_dir = self._output_dir
    
    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate video using Morphic Frames-to-Video (matches VMEvalKit interface).
        
        Args:
            image_path: Path to input image (first frame)
            text_prompt: Text prompt for video generation
            duration: Video duration in seconds
            output_filename: Optional output filename
            **kwargs: Additional parameters, including question_data with final_image_path
            
        Returns:
            Dictionary with generation results
        """
        # Extract final_image_path from question_data (check only once at the beginning)
        question_data = kwargs.get('question_data', {})
        final_image_path = question_data.get('final_image_path')
        
        if not final_image_path:
            raise ValueError("Morphic requires final_image_path in question_data")
        
        # Remove question_data from kwargs before passing to service
        service_kwargs = {k: v for k, v in kwargs.items() if k != 'question_data'}
        
        return self.morphic_service.generate(
            image_path=image_path,
            text_prompt=text_prompt,
            duration=duration,
            output_filename=output_filename,
            final_image_path=final_image_path,
            **service_kwargs
        )
