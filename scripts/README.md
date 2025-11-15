# VMEvalKit HuggingFace Scripts

Utility scripts for uploading and downloading VMEvalKit datasets to/from HuggingFace Hub.

## Prerequisites

```bash
# Install HuggingFace dependencies
pip install huggingface_hub datasets pyarrow python-dotenv

# Set your HuggingFace token
export HF_TOKEN=hf_your_token_here
# Or add to .env file
echo "HF_TOKEN=hf_your_token_here" >> .env
```

Get your token from: https://huggingface.co/settings/tokens

## Scripts Overview

### 1. `upload_to_hf.py` - Upload datasets to HuggingFace

Upload questions, outputs, or evaluations to HuggingFace Hub using folder mirroring.

**Usage:**
```bash
# Upload questions dataset (public)
python scripts/upload_to_hf.py \
    --repo-id your-username/vmeval-questions \
    --type questions

# Upload questions dataset (private)
python scripts/upload_to_hf.py \
    --repo-id your-username/vmeval-questions \
    --type questions \
    --private

# Upload model outputs
python scripts/upload_to_hf.py \
    --repo-id your-username/vmeval-outputs \
    --type outputs \
    --experiment pilot_experiment \
    --private

# Upload evaluations
python scripts/upload_to_hf.py \
    --repo-id your-username/vmeval-evals \
    --type evaluations \
    --experiment pilot_experiment \
    --private
```

### 2. `download_from_hf.py` - Download datasets from HuggingFace

Download datasets back to your local directory structure.

**Usage:**
```bash
# Download questions dataset
python scripts/download_from_hf.py \
    --repo-id your-username/vmeval-questions \
    --type questions

# Download model outputs
python scripts/download_from_hf.py \
    --repo-id your-username/vmeval-outputs \
    --type outputs \
    --experiment pilot_experiment

# Download evaluations
python scripts/download_from_hf.py \
    --repo-id your-username/vmeval-evals \
    --type evaluations \
    --experiment pilot_experiment
```

### 3. `convert_to_hf_dataset.py` - Convert to structured HuggingFace Dataset

Convert VMEvalKit questions to HuggingFace's native Dataset format with Arrow/Parquet storage.

**Usage:**
```bash
# Convert and upload as structured dataset
python scripts/convert_to_hf_dataset.py \
    --repo-id your-username/vmeval-dataset \
    --private
```

This creates domain splits (full, chess, maze, raven, rotation, sudoku) and enables:
- Streaming without full download
- Built-in filtering and processing
- Automatic data type handling (images as PIL)

## Quick Start Example

```bash
# 1. Upload your questions dataset
python scripts/upload_to_hf.py \
    --repo-id demo-user/vmeval-questions \
    --type questions \
    --private

# 2. Share the repo with collaborators on HuggingFace

# 3. Collaborator downloads it
python scripts/download_from_hf.py \
    --repo-id demo-user/vmeval-questions \
    --type questions

# 4. Now they can use it with VMEvalKit
python examples/generate_videos.py --model luma-ray-2 --task chess
```

## Repository Structure

After upload, your HuggingFace repos will look like:

### Questions Repo (`vmeval-questions`)
```
├── README.md
├── chess_task/
│   └── chess_0000/
│       ├── first_frame.png
│       ├── final_frame.png
│       ├── prompt.txt
│       └── question_metadata.json
├── maze_task/
├── raven_task/
├── rotation_task/
├── sudoku_task/
└── vmeval_dataset.json
```

### Outputs Repo (`vmeval-outputs-pilot`)
```
├── README.md
└── {model_name}/
    └── {domain}_task/
        └── {task_id}/
            └── {run_id}/
                ├── video/
                │   └── model_output.mp4
                ├── question/
                └── metadata.json
```

## Tips

1. **Size Management**: Outputs can be large (>1GB). Consider:
   - Creating separate repos per model
   - Using `--private` for outputs
   - Splitting by domain or experiment

2. **Version Control**: Tag important versions:
   ```bash
   # On HuggingFace web interface, create tags/releases
   # Or use Git commands in the cloned repo
   ```

3. **Access Control**: 
   - Public repos: Anyone can download
   - Private repos: Only you and collaborators
   - Require authentication token for private repos

4. **Updates**: To update a dataset, just re-run the upload script
   - Files are versioned automatically by HuggingFace

## Troubleshooting

### "No module named 'huggingface_hub'"
```bash
pip install huggingface_hub datasets pyarrow
```

### "Invalid token" or authentication errors
```bash
# Check your token
huggingface-cli whoami

# Re-login
huggingface-cli login
```

### Large file upload timeout
For very large files (>5GB), the upload might timeout. Consider:
- Splitting into smaller repos
- Using HuggingFace's git-lfs directly
- Compressing videos before upload

### Download stuck or slow
- Check internet connection
- Try resuming (re-run the same command)
- HuggingFace supports resume for interrupted downloads

## Further Reading

See the full guide: [docs/Convert_HuggingFace.md](../docs/Convert_HuggingFace.md)

---

*VMEvalKit Team*

