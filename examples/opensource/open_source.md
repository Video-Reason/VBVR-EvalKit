## Installation


```

uv pip install transformers diffusers
```


## HunyuanVideo-I2V

Use conda to install the HunyuanVideo-I2V submodule first, then install the vmevalkit.

```bash
TODO
```



## LTX-Video


You need to install LTX-Video submodule first, then install the vmevalkit.


```bash
cd submodules/LTX-Video

uv venv
source .venv/bin/activate
uv pip install -e .\[inference\]

uv pip install -e ../..

cd ../../
uv run --project submodules/LTX-Video python -c "import sys; print(sys.executable)"  # to check if the submodule is installed correctly

uv run --project submodules/LTX-Video python -c "import ltx_inference; print(ltx_inference.__file__)"

python inference.py --prompt "PROMPT" --conditioning_media_paths IMAGE_PATH --conditioning_start_frames 0 --height HEIGHT --width WIDTH --num_frames NUM_FRAMES --seed SEED --pipeline_config configs/ltxv-13b-0.9.8-distilled.yaml

```