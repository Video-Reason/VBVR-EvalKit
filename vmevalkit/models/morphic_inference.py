"""
Morphic Frames-to-Video integration for VMEvalKit.

This wrapper bridges the open-source Morphic/Wan2.2 interpolation pipeline
that lives in the `submodules/frames-to-video` git submodule and exposes it
through the standard `ModelWrapper` interface used by the inference runner.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

from .base import ModelWrapper


class MorphicConfigurationError(RuntimeError):
    """Raised when the Morphic environment is not configured correctly."""


@dataclass
class MorphicConfig:
    """Runtime configuration for the Morphic Frames-to-Video wrapper."""

    repo_dir: Path
    python_bin: Path
    ckpt_dir: Path
    lora_path: Optional[Path] = None
    task: str = "i2v-A14B"
    size: str = "1280*720"
    frame_num: int = 81
    extra_cli_args: List[str] = field(default_factory=list)

    @classmethod
    def from_env(
        cls,
        repo_dir: Optional[Union[str, Path]] = None,
        python_bin: Optional[str] = None,
        ckpt_dir: Optional[Union[str, Path]] = None,
        lora_path: Optional[Union[str, Path]] = None,
        **overrides: Any,
    ) -> "MorphicConfig":
        """
        Build a configuration using environment variables with sensible defaults.
        """
        project_root = Path(__file__).resolve().parents[2]
        default_repo_dir = project_root / "submodules" / "frames-to-video"

        repo_path = Path(repo_dir) if repo_dir else default_repo_dir
        if not (repo_path / "generate.py").exists():
            raise MorphicConfigurationError(
                "Morphic submodule not found. Expected to locate "
                f"'generate.py' under {repo_path}. "
                "Ensure `git submodule update --init --recursive` has been run."
            )

        python_path = Path(python_bin or os.getenv("MORPHIC_PYTHON_BIN", sys.executable))
        if not python_path.exists():
            raise MorphicConfigurationError(
                f"Configured Python interpreter does not exist: {python_path}"
            )

        ckpt_path = Path(
            ckpt_dir
            or os.getenv("WAN22_I2V_CKPT_DIR")
            or repo_path / "Wan2.2-I2V-A14B"
        )
        if not ckpt_path.exists():
            raise MorphicConfigurationError(
                "Wan2.2 checkpoint directory not found. "
                "Set WAN22_I2V_CKPT_DIR to point at the extracted weights."
            )

        lora_env = lora_path or os.getenv("MORPHIC_LORA_WEIGHTS")
        lora_resolved = Path(lora_env).resolve() if lora_env else None
        if lora_env and not Path(lora_env).exists():
            raise MorphicConfigurationError(
                f"Configured Morphic LoRA weights not found: {lora_env}"
            )

        cfg = cls(
            repo_dir=repo_path.resolve(),
            python_bin=python_path.resolve(),
            ckpt_dir=ckpt_path.resolve(),
            lora_path=lora_resolved,
            task=overrides.pop("task", "i2v-A14B"),
            size=overrides.pop("size", "1280*720"),
            frame_num=int(overrides.pop("frame_num", 81)),
            extra_cli_args=overrides.pop("extra_cli_args", []),
        )

        return cfg


class MorphicFramesToVideoWrapper(ModelWrapper):
    """
    VMEvalKit wrapper that invokes Morphic's frames-to-video pipeline.
    """

    def __init__(
        self,
        model: str,
        output_dir: str = "./data/outputs",
        **kwargs: Any,
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        self.config = MorphicConfig.from_env(**kwargs)

    def generate(
        self,
        image_path: Union[str, Path],
        text_prompt: str,
        duration: float = 8.0,
        output_filename: Optional[str] = None,
        final_image: Optional[Union[str, Path]] = None,
        middle_images: Optional[List[str]] = None,
        middle_timestamps: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate a video using Morphic's pipeline.

        Additional kwargs are converted into CLI switches so advanced flags
        (e.g. --sample_solver) can be overridden from the runner.
        """
        start = time.time()

        source_image = Path(image_path)
        if not source_image.exists():
            raise FileNotFoundError(f"First frame image not found: {source_image}")

        if output_filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"{self.model}_{timestamp}.mp4"

        video_path = (self.output_dir / output_filename).resolve()

        cli_args = [
            str(self.config.python_bin),
            str(self.config.repo_dir / "generate.py"),
            "--task",
            self.config.task,
            "--size",
            self.config.size,
            "--frame_num",
            str(self.config.frame_num),
            "--ckpt_dir",
            str(self.config.ckpt_dir),
            "--image",
            str(source_image),
            "--prompt",
            text_prompt,
            "--save_file",
            str(video_path),
        ]

        if self.config.lora_path:
            cli_args.extend(["--high_noise_lora_weights_path", str(self.config.lora_path)])

        if final_image:
            final_path = Path(final_image)
            if final_path.exists():
                cli_args.extend(["--img_end", str(final_path)])

        if middle_images:
            cli_args.extend(["--middle_images"] + [str(Path(p)) for p in middle_images])
            if middle_timestamps:
                cli_args.extend(
                    ["--middle_images_timestamps"] + [str(ts) for ts in middle_timestamps]
                )

        # Propagate additional CLI overrides
        for key, value in kwargs.items():
            arg_key = f"--{key.replace('_', '-')}"
            if isinstance(value, bool):
                if value:
                    cli_args.append(arg_key)
            elif isinstance(value, (list, tuple)):
                cli_args.append(arg_key)
                cli_args.extend([str(v) for v in value])
            else:
                cli_args.extend([arg_key, str(value)])

        cli_args.extend(self.config.extra_cli_args)

        env = os.environ.copy()
        env.setdefault("WAN_HOME", str(self.config.repo_dir))

        process = subprocess.run(
            cli_args,
            cwd=self.config.repo_dir,
            env=env,
            capture_output=True,
            text=True,
        )

        duration_seconds = time.time() - start

        if process.returncode != 0 or not video_path.exists():
            error_message = "\n".join(
                [
                    "Morphic generation failed.",
                    f"Command: {' '.join(shlex.quote(a) for a in cli_args)}",
                    f"Return code: {process.returncode}",
                    f"Stdout: {process.stdout.strip()}",
                    f"Stderr: {process.stderr.strip()}",
                ]
            )
            return {
                "success": False,
                "video_path": None,
                "error": error_message,
                "duration_seconds": duration_seconds,
                "generation_id": "morphic",
                "model": self.model,
                "status": "failed",
                "metadata": {
                    "prompt": text_prompt,
                    "image_path": str(source_image),
                },
            }

        return {
            "success": True,
            "video_path": str(video_path),
            "error": None,
            "duration_seconds": duration_seconds,
            "generation_id": "morphic",
            "model": self.model,
            "status": "success",
            "metadata": {
                "prompt": text_prompt,
                "image_path": str(source_image),
                "stdout": process.stdout.strip(),
            },
        }


__all__ = [
    "MorphicFramesToVideoWrapper",
    "MorphicConfigurationError",
]