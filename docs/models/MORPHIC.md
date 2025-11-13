# Morphic Frames-to-Video Integration

This guide explains how to configure the `morphic-wan2.2-i2v` model inside VMEvalKit. The implementation bridges our `InferenceRunner` with the Morphic Frames-to-Video pipeline (Wan 2.2 + Morphic LoRA) that lives in the `submodules/frames-to-video` git submodule.

## 1. Prerequisites

- **Hardware**: At least 1 × A100 80GB (or equivalent) GPU is required for the Wan 2.2 I2V pipeline. Multi-GPU (torchrun) is also supported if you already run Wan locally.
- **Git submodule**: Make sure the submodule is checked out locally:
  ```bash
  git submodule update --init --recursive submodules/frames-to-video
  ```
- **Python environment**: We recommend creating a dedicated environment that matches the requirements shipped with Morphic. You can either follow `submodules/frames-to-video/INSTALL.md` manually or run the helper script:
  ```bash
  cd submodules/frames-to-video
  bash setup_env.sh  # optional convenience script from Morphic repo
  ```
  You can use that environment directly (activate it before running VMEvalKit) or set `MORPHIC_PYTHON_BIN=/path/to/python` so the wrapper launches the correct interpreter.

## 2. Download Required Weights

1. **Wan 2.2 I2V Base Weights**
   ```bash
   huggingface-cli download Wan-AI/Wan2.2-I2V-A14B --local-dir /path/to/Wan2.2-I2V-A14B
   ```
   Set `WAN22_I2V_CKPT_DIR` to the directory you downloaded above. If this variable is omitted the wrapper falls back to `submodules/frames-to-video/Wan2.2-I2V-A14B`.

2. **Morphic Frames-to-Video LoRA Weights**
   ```bash
   huggingface-cli download morphic/Wan2.2-frames-to-video --local-dir /path/to/morphic-frames-lora-weights
   ```
   Set `MORPHIC_LORA_WEIGHTS` to this directory if you want to apply Morphic’s LoRA automatically.

You may place both directories anywhere, as long as the environment variables point to their absolute paths.

## 3. Environment Variables

Add the following variables to your `.env` or shell profile:
```bash
export WAN22_I2V_CKPT_DIR="/absolute/path/to/Wan2.2-I2V-A14B"
export MORPHIC_LORA_WEIGHTS="/absolute/path/to/morphic-frames-lora-weights"
# Optional: use a dedicated Python interpreter
export MORPHIC_PYTHON_BIN="/path/to/morphic/env/bin/python"
```
The wrapper validates these paths before launching the generator and will raise actionable errors if anything is missing.

## 4. Running a Sanity Check

We provide a small helper script at `examples/run_morphic_demo.py`:
```bash
python examples/run_morphic_demo.py \
  --image data/questions/maze_task/maze_0000/first_frame.png \
  --prompt "Navigate the green dot through the maze corridors to reach the red flag" \
  --output-dir data/outputs/morphic-demo
```
Optional flags:
- `--final-image`: path to a reference frame passed through `--img_end`
- `--middle-images`: list of intermediate frames (multiple allowed)
- `--middle-timestamps`: timestamps for intermediate frames (0-1 range)

The script generates a single sample and prints the resulting file path. The output directory is fully compatible with the rest of the VMEvalKit tooling.

## 5. Using Inside Experiments

Once the environment is configured you can reference the model as `morphic-wan2.2-i2v` anywhere our orchestration expects a model name, e.g.:
```bash
python examples/experiment_2025-10-14.py --all-tasks --only-model morphic-wan2.2-i2v
```
Any additional keyword arguments you pass via `runner.run(..., **kwargs)` will be translated into CLI flags for `generate.py`. For example, to change the sampler:
```python
runner.run(
    model_name="morphic-wan2.2-i2v",
    image_path="...",
    text_prompt="...",
    question_data=question,
    sample_solver="dpm++"
)
```

## 6. Troubleshooting Checklist

- `Morphic submodule not found`: run the git submodule command and ensure the path is correct.
- `WAN22_I2V_CKPT_DIR not found`: verify the weights path exists and the environment variable resolves to an absolute directory.
- `CUDA out of memory`: reduce `frame_num` or run across multiple GPUs via `torchrun` (set your own distributed launch wrapper and pass `extra_cli_args=["--ulysses_size", "8"]`).
- `subprocess failed`: inspect the stdout/stderr printed in the error response – we capture Morphic’s logs in the wrapper’s return dictionary for debugging.

With these steps, the Morphic Frames-to-Video pipeline behaves just like any other VMEvalKit model: structured outputs land in `data/outputs`, evaluations consume them automatically, and the web dashboard can display the generated videos.


