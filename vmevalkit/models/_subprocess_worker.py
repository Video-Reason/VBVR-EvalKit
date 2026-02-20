#!/usr/bin/env python3
"""Generic subprocess worker for VMEvalKit model inference.

This script runs inside a model-specific virtual environment.
It imports the model wrapper (which may have heavy dependencies like
torch, diffusers, omegaconf, etc.) and runs inference, returning
results as JSON on stdout.

Usage:
    /path/to/venv/bin/python _subprocess_worker.py \
        --model-name svd \
        --image-path /path/to/image.png \
        --prompt "Generate a video" \
        --output-dir /path/to/output \
        --kwargs-json '{"num_frames": 25}'
"""

import sys
import json
import argparse
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="VMEvalKit subprocess worker")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--kwargs-json", default="{}")
    args = parser.parse_args()

    # Ensure VMEvalKit root is on sys.path
    vmeval_root = str(Path(__file__).parent.parent.parent)
    if vmeval_root not in sys.path:
        sys.path.insert(0, vmeval_root)

    start_time = time.time()

    # Import catalog (light - no heavy deps)
    from vmevalkit.runner.MODEL_CATALOG import AVAILABLE_MODELS

    if args.model_name not in AVAILABLE_MODELS:
        result = {
            "success": False,
            "video_path": None,
            "error": f"Unknown model: {args.model_name}",
            "duration_seconds": time.time() - start_time,
            "generation_id": None,
            "model": args.model_name,
            "status": "failed",
            "metadata": {},
        }
        print(json.dumps(result, default=str))
        return

    config = AVAILABLE_MODELS[args.model_name]

    # Dynamically import the wrapper module (this triggers heavy deps in venv)
    import importlib

    module = importlib.import_module(config["wrapper_module"])
    wrapper_class = getattr(module, config["wrapper_class"])

    # Build init kwargs
    init_kwargs = {
        "model": config["model"],
        "output_dir": args.output_dir,
    }
    if "args" in config:
        init_kwargs.update(config["args"])

    # Instantiate wrapper and run inference
    wrapper = wrapper_class(**init_kwargs)

    kwargs = json.loads(args.kwargs_json)
    result = wrapper.generate(args.image_path, args.prompt, **kwargs)

    # Ensure result is JSON-serializable
    serializable_result = {}
    for key, value in result.items():
        try:
            json.dumps(value, default=str)
            serializable_result[key] = value
        except (TypeError, ValueError):
            serializable_result[key] = str(value)

    # Output result as JSON on stdout (last line)
    print("__VMEVAL_RESULT__" + json.dumps(serializable_result, default=str))


if __name__ == "__main__":
    main()
