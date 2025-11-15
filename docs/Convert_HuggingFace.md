# Converting VMEvalKit Datasets to HuggingFace Format

This guide explains how to convert and upload VMEvalKit datasets to HuggingFace Hub for sharing and collaboration.

## Overview

VMEvalKit uses a hierarchical file structure with images, text files, and JSON metadata. HuggingFace supports two main approaches for hosting such datasets:

1. **Folder Mirroring** (Recommended) - Upload the exact folder structure
2. **Structured Dataset** (Advanced) - Convert to HuggingFace's Arrow/Parquet format

## Approach 1: Folder Mirroring (Recommended)

### Why This Approach?

**Pros:**
- ✅ Zero data transformation required
- ✅ Preserves exact file structure VMEvalKit expects
- ✅ Works immediately after download with existing code
- ✅ Simple one-command upload and download
- ✅ Handles any file type (images, text, JSON, videos, etc.)
- ✅ Easy to add new files or update existing ones

**Cons:**
- ❌ Doesn't leverage HuggingFace's dataset streaming/filtering features
- ❌ Must download entire dataset to use (no lazy loading)

### Installation

```bash
pip install huggingface_hub python-dotenv
```

### Setup Authentication

```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN=hf_your_token_here

# Or add to .env file
echo "HF_TOKEN=hf_your_token_here" >> .env
```

### Upload Script

Create `scripts/upload_to_hf.py`:

```python
#!/usr/bin/env python3
"""Upload VMEvalKit datasets to HuggingFace Hub."""

import os
from pathlib import Path
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def upload_questions_dataset(repo_id: str, private: bool = False):
    """Upload questions dataset to HuggingFace."""
    
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    # Create repository
    print(f"📦 Creating repository: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True
    )
    
    # Upload questions folder
    print(f"⬆️  Uploading questions dataset...")
    api.upload_folder(
        folder_path="data/questions",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload VMEval questions dataset",
        ignore_patterns=[".DS_Store", "__pycache__", "*.pyc"]
    )
    
    print(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")


def upload_outputs_dataset(repo_id: str, experiment: str = "pilot_experiment", private: bool = True):
    """Upload model outputs (videos) to HuggingFace."""
    
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    output_path = Path("data/outputs") / experiment
    if not output_path.exists():
        print(f"❌ Output directory not found: {output_path}")
        return
    
    # Create repository
    print(f"📦 Creating repository: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True
    )
    
    # Upload outputs folder
    print(f"⬆️  Uploading outputs from {experiment}...")
    api.upload_folder(
        folder_path=str(output_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {experiment} outputs",
        ignore_patterns=[".DS_Store", "__pycache__", "*.pyc"]
    )
    
    print(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")


def upload_evaluations_dataset(repo_id: str, experiment: str = "pilot_experiment", private: bool = True):
    """Upload evaluation results to HuggingFace."""
    
    api = HfApi(token=os.getenv("HF_TOKEN"))
    
    eval_path = Path("data/evaluations") / experiment
    if not eval_path.exists():
        print(f"❌ Evaluations directory not found: {eval_path}")
        return
    
    # Create repository
    print(f"📦 Creating repository: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True
    )
    
    # Upload evaluations folder
    print(f"⬆️  Uploading evaluations from {experiment}...")
    api.upload_folder(
        folder_path=str(eval_path),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {experiment} evaluations",
        ignore_patterns=[".DS_Store", "__pycache__", "*.pyc"]
    )
    
    print(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload VMEvalKit datasets to HuggingFace")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID (e.g., username/dataset-name)")
    parser.add_argument("--type", choices=["questions", "outputs", "evaluations"], required=True)
    parser.add_argument("--experiment", default="pilot_experiment", help="Experiment name for outputs/evaluations")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    
    args = parser.parse_args()
    
    if args.type == "questions":
        upload_questions_dataset(args.repo_id, args.private)
    elif args.type == "outputs":
        upload_outputs_dataset(args.repo_id, args.experiment, args.private)
    elif args.type == "evaluations":
        upload_evaluations_dataset(args.repo_id, args.experiment, args.private)
```

### Usage

```bash
# Upload questions dataset
python scripts/upload_to_hf.py \
    --repo-id your-username/vmeval-questions \
    --type questions \
    --private

# Upload model outputs (large files - keep private)
python scripts/upload_to_hf.py \
    --repo-id your-username/vmeval-outputs-pilot \
    --type outputs \
    --experiment pilot_experiment \
    --private

# Upload evaluations
python scripts/upload_to_hf.py \
    --repo-id your-username/vmeval-evaluations-pilot \
    --type evaluations \
    --experiment pilot_experiment \
    --private
```

### Download Script

Create `scripts/download_from_hf.py`:

```python
#!/usr/bin/env python3
"""Download VMEvalKit datasets from HuggingFace Hub."""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def download_questions_dataset(repo_id: str):
    """Download questions dataset from HuggingFace."""
    
    target_dir = Path("data/questions")
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⬇️  Downloading questions dataset from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        token=os.getenv("HF_TOKEN"),
    )
    
    print(f"✅ Downloaded to: {target_dir}")


def download_outputs_dataset(repo_id: str, experiment: str = "pilot_experiment"):
    """Download model outputs from HuggingFace."""
    
    target_dir = Path("data/outputs") / experiment
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⬇️  Downloading outputs from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        token=os.getenv("HF_TOKEN"),
    )
    
    print(f"✅ Downloaded to: {target_dir}")


def download_evaluations_dataset(repo_id: str, experiment: str = "pilot_experiment"):
    """Download evaluations from HuggingFace."""
    
    target_dir = Path("data/evaluations") / experiment
    target_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"⬇️  Downloading evaluations from {repo_id}...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(target_dir),
        token=os.getenv("HF_TOKEN"),
    )
    
    print(f"✅ Downloaded to: {target_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download VMEvalKit datasets from HuggingFace")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    parser.add_argument("--type", choices=["questions", "outputs", "evaluations"], required=True)
    parser.add_argument("--experiment", default="pilot_experiment", help="Experiment name for outputs/evaluations")
    
    args = parser.parse_args()
    
    if args.type == "questions":
        download_questions_dataset(args.repo_id)
    elif args.type == "outputs":
        download_outputs_dataset(args.repo_id, args.experiment)
    elif args.type == "evaluations":
        download_evaluations_dataset(args.repo_id, args.experiment)
```

### Usage

```bash
# Download questions dataset
python scripts/download_from_hf.py \
    --repo-id your-username/vmeval-questions \
    --type questions

# Download outputs
python scripts/download_from_hf.py \
    --repo-id your-username/vmeval-outputs-pilot \
    --type outputs \
    --experiment pilot_experiment

# Download evaluations
python scripts/download_from_hf.py \
    --repo-id your-username/vmeval-evaluations-pilot \
    --type evaluations \
    --experiment pilot_experiment
```

### Create Great Dataset Cards

After uploading, create informative README files for each dataset:

**Questions Dataset Card** (`README.md` on HuggingFace):

```markdown
# VMEval Questions Dataset

Reasoning tasks for evaluating video generation models across 5 cognitive domains.

## Dataset Structure

```
questions/
├── vmeval_dataset.json           # Master manifest
├── chess_task/                   # Strategic reasoning
│   └── chess_0000/
│       ├── first_frame.png       # Initial state
│       ├── final_frame.png       # Target state
│       ├── prompt.txt            # Instructions
│       └── question_metadata.json
├── maze_task/                    # Spatial reasoning
├── raven_task/                   # Abstract reasoning
├── rotation_task/                # 3D visualization
└── sudoku_task/                  # Logical reasoning
```

## Cognitive Domains

- **Chess** (15 tasks): Strategic thinking and tactical pattern recognition
- **Maze** (15 tasks): Spatial reasoning and navigation planning
- **Raven** (15 tasks): Abstract reasoning and pattern completion
- **Rotation** (15 tasks): 3D mental rotation and spatial visualization
- **Sudoku** (15 tasks): Logical reasoning and constraint satisfaction

**Total: 75 reasoning task pairs**

## Usage with VMEvalKit

```python
from huggingface_hub import snapshot_download

# Download dataset
snapshot_download(
    repo_id="your-username/vmeval-questions",
    repo_type="dataset",
    local_dir="data/questions",
)

# Use with VMEvalKit
from vmevalkit.runner.inference import InferenceRunner

runner = InferenceRunner(output_dir="data/outputs")
result = runner.run(
    model_name="luma-ray-2",
    image_path="data/questions/chess_task/chess_0000/first_frame.png",
    text_prompt="White to move and checkmate in 1...",
)
```

## Dataset Statistics

- **Total Size**: ~6 MB
- **Image Format**: PNG, 1280×720 (padded for video generation)
- **Tasks per Domain**: 15
- **Metadata**: JSON format with task-specific information

## Citation

```bibtex
@misc{vmeval2025,
  title={VMEvalKit: Video Model Evaluation Toolkit},
  author={Your Name},
  year={2025},
  url={https://github.com/your-org/VMEvalKit}
}
```

## License

MIT License
```

---

## Approach 2: Structured Dataset (Advanced)

### Why This Approach?

**Pros:**
- ✅ Leverages HuggingFace's dataset features (filtering, streaming, Arrow format)
- ✅ Can load/preview directly in HuggingFace UI
- ✅ Lazy loading - don't need to download everything
- ✅ Better for ML pipelines and training workflows
- ✅ Automatic data type handling (images as PIL, etc.)

**Cons:**
- ❌ Requires data transformation code
- ❌ Need to reconstruct file structure if using with VMEvalKit
- ❌ More complex to set up initially
- ❌ Updates require regenerating the entire dataset

### Conversion Script

Create `scripts/convert_to_hf_dataset.py`:

```python
#!/usr/bin/env python3
"""Convert VMEvalKit questions to HuggingFace Dataset format."""

import json
from pathlib import Path
from datasets import Dataset, Features, Value, Image, DatasetDict
from dotenv import load_dotenv

load_dotenv()


def convert_questions_to_dataset() -> Dataset:
    """Convert VMEvalKit questions structure to HuggingFace Dataset."""
    
    # Load master manifest
    manifest_path = Path("data/questions/vmeval_dataset.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    print(f"📊 Loading {len(manifest['pairs'])} task pairs...")
    
    # Prepare data rows
    data_rows = []
    for pair in manifest["pairs"]:
        task_id = pair["id"]
        domain = pair["domain"]
        
        # Build paths
        base_path = Path("data/questions") / f"{domain}_task" / task_id
        first_frame_path = base_path / "first_frame.png"
        final_frame_path = base_path / "final_frame.png"
        
        # Verify files exist
        if not first_frame_path.exists():
            print(f"⚠️  Missing: {first_frame_path}")
            continue
        if not final_frame_path.exists():
            print(f"⚠️  Missing: {final_frame_path}")
            continue
        
        row = {
            "id": task_id,
            "domain": domain,
            "difficulty": pair.get("difficulty", "unknown"),
            "task_category": pair.get("task_category", "unknown"),
            "prompt": pair["prompt"],
            "first_frame": str(first_frame_path),
            "final_frame": str(final_frame_path),
            "metadata": json.dumps(pair.get("metadata", {})),
            "created_at": pair.get("created_at", ""),
        }
        data_rows.append(row)
    
    print(f"✅ Prepared {len(data_rows)} valid task pairs")
    
    # Define schema with Image types
    features = Features({
        "id": Value("string"),
        "domain": Value("string"),
        "difficulty": Value("string"),
        "task_category": Value("string"),
        "prompt": Value("string"),
        "first_frame": Image(),  # Automatically encodes image
        "final_frame": Image(),  # Automatically encodes image
        "metadata": Value("string"),  # JSON as string
        "created_at": Value("string"),
    })
    
    # Create dataset
    dataset = Dataset.from_dict(
        {key: [row[key] for row in data_rows] for key in data_rows[0].keys()},
        features=features
    )
    
    return dataset


def create_domain_splits(dataset: Dataset) -> DatasetDict:
    """Create dataset splits by cognitive domain."""
    
    splits = {
        "full": dataset,
        "chess": dataset.filter(lambda x: x["domain"] == "chess"),
        "maze": dataset.filter(lambda x: x["domain"] == "maze"),
        "raven": dataset.filter(lambda x: x["domain"] == "raven"),
        "rotation": dataset.filter(lambda x: x["domain"] == "rotation"),
        "sudoku": dataset.filter(lambda x: x["domain"] == "sudoku"),
    }
    
    return DatasetDict(splits)


def upload_to_hub(repo_id: str, private: bool = False):
    """Convert and upload dataset to HuggingFace Hub."""
    
    print("🔄 Converting VMEvalKit dataset to HuggingFace format...")
    dataset = convert_questions_to_dataset()
    
    print("📦 Creating domain splits...")
    dataset_dict = create_domain_splits(dataset)
    
    # Print statistics
    print("\n📊 Dataset Statistics:")
    for split_name, split_data in dataset_dict.items():
        print(f"  {split_name}: {len(split_data)} tasks")
    
    print(f"\n⬆️  Uploading to {repo_id}...")
    dataset_dict.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message="Upload VMEval questions dataset"
    )
    
    print(f"✅ Successfully uploaded to: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert and upload VMEvalKit dataset to HuggingFace")
    parser.add_argument("--repo-id", required=True, help="HuggingFace repo ID")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    
    args = parser.parse_args()
    
    upload_to_hub(args.repo_id, args.private)
```

### Usage

```bash
# Convert and upload
python scripts/convert_to_hf_dataset.py \
    --repo-id your-username/vmeval-questions-dataset \
    --private
```

### Using the Structured Dataset

```python
from datasets import load_dataset

# Load all domains
ds = load_dataset("your-username/vmeval-questions-dataset", split="full")

# Load specific domain
chess_ds = load_dataset("your-username/vmeval-questions-dataset", split="chess")

# Stream without downloading everything
ds = load_dataset("your-username/vmeval-questions-dataset", split="full", streaming=True)

# Access data
for item in chess_ds:
    print(f"ID: {item['id']}")
    print(f"Prompt: {item['prompt']}")
    
    # Images are PIL.Image objects
    item['first_frame'].show()
    item['final_frame'].show()
    
    # Metadata is JSON string
    metadata = json.loads(item['metadata'])
    print(metadata)

# Filter and process
hard_tasks = ds.filter(lambda x: x['difficulty'] == 'hard')
```

---

## Comparison Table

| Feature | Folder Mirroring | Structured Dataset |
|---------|------------------|-------------------|
| Setup Complexity | Simple | Moderate |
| VMEvalKit Compatibility | Perfect | Requires adaptation |
| HuggingFace Features | Limited | Full |
| Upload Time | Fast | Slower (encoding) |
| Download Required | Full | Partial (streaming) |
| Update Process | Simple | Regenerate all |
| File Preservation | Exact | Transformed |
| Preview on HF | File browser | Data viewer |

---

## Best Practices

### 1. Repository Naming

Use clear, descriptive names:
- `username/vmeval-questions` - Questions dataset
- `username/vmeval-outputs-{experiment}` - Model outputs
- `username/vmeval-eval-{experiment}` - Evaluation results

### 2. Privacy Settings

- **Questions**: Usually public (share with community)
- **Outputs**: Usually private (large files, proprietary models)
- **Evaluations**: Depends on publication status

### 3. Size Considerations

- Questions: ~6 MB ✅ Suitable for HuggingFace
- Outputs: ~650 MB - 10 GB ⚠️ Consider Git LFS or separate repos per model
- Consider splitting large outputs by model or domain

### 4. Version Control

Tag releases in HuggingFace:

```python
from huggingface_hub import HfApi

api = HfApi(token=os.getenv("HF_TOKEN"))

# Create a tag/release
api.create_commit(
    repo_id="username/vmeval-questions",
    repo_type="dataset",
    operations=[],  # No file changes
    commit_message="Release v1.0",
    revision="main",
    create_pr=False,
)
```

### 5. Dataset Cards

Always include:
- Clear description of what the dataset contains
- Usage examples with code
- Dataset statistics
- License information
- Citation information
- Link back to VMEvalKit repository

---

## Recommended Workflow

For VMEvalKit, we recommend:

### Questions Dataset
**Use Folder Mirroring** - Simple, preserves structure, easy to use

### Model Outputs
**Use Folder Mirroring with separate repos** - One repo per experiment or per model to manage size

### Evaluations
**Use Folder Mirroring** - Small files, want exact structure

### Leaderboard/Results
**Use Structured Dataset** - Create a separate aggregated results dataset for easy querying

---

## Installation Requirements

Add to `requirements.txt`:

```txt
# HuggingFace integration (optional)
huggingface_hub>=0.24.0
datasets>=2.18.0
pyarrow>=14.0.2
```

---

## Troubleshooting

### Large File Upload Issues

If uploads fail due to size:

```python
# Use upload_large_folder for better handling
api.upload_large_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path="data/outputs",
    # This creates multiple commits and is resumable
)
```

### Network Timeouts

Set longer timeout:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="username/dataset",
    repo_type="dataset",
    local_dir="data/questions",
    etag_timeout=30,  # Increase timeout
)
```

### Authentication Issues

Verify token:

```bash
huggingface-cli whoami
```

---

## Further Reading

- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)
- [Datasets Library Documentation](https://huggingface.co/docs/datasets/index)
- [HuggingFace Hub Python Client](https://huggingface.co/docs/huggingface_hub/index)

---

*Last updated: November 2024*
*VMEvalKit Team*

