# VMEvalKit Data Management

This module provides comprehensive data management for VMEvalKit datasets, including version control, S3 synchronization, dataset organization, and web-based visualization.

## Overview

VMEvalKit's data management system handles:
- 📁 **Dataset Organization** - Structured storage of questions, inference results, and evaluations
- ☁️ **S3 Synchronization** - Automated backup and sharing via AWS S3
- 🔖 **Version Tracking** - Built-in versioning for reproducibility
- 🎥 **Web Dashboard** - Interactive visualization of results
- 🤗 **HuggingFace Integration** - Alternative dataset hosting option
- 🔄 **Experiment Management** - Organized experiment tracking and results
- ✅ **Data Validation** - Integrity checking and dataset verification

## Dataset Structure

VMEvalKit uses a hierarchical structure for organizing all data:

```
data/
├── questions/                       # Task datasets
│   ├── vmeval_dataset.json         # Master dataset manifest
│   ├── chess_task/                 # Chess puzzles (mate-in-1 scenarios)
│   │   └── chess_0000/
│   │       ├── first_frame.png     # Initial chess position
│   │       ├── final_frame.png     # Solution position
│   │       ├── prompt.txt          # Move instructions
│   │       └── question_metadata.json  # Task metadata
│   ├── maze_task/                  # Maze solving challenges
│   ├── raven_task/                 # Raven's progressive matrices
│   ├── rotation_task/              # 3D mental rotation
│   └── sudoku_task/                # Sudoku puzzles
│
├── outputs/                         # Model inference results
│   └── pilot_experiment/           # Experiment name
│       └── <model_name>/           # e.g., openai-sora-2, luma-ray-2
│           └── <domain>_task/      # e.g., chess_task
│               └── <task_id>/      # e.g., chess_0000
│                   └── <run_id>/   # Timestamped run folder
│                       ├── video/
│                       │   └── model_output.mp4
│                       ├── question/
│                       │   ├── prompt.txt
│                       │   └── first_frame.png
│                       └── metadata.json
│
├── evaluations/                     # Evaluation results
│   └── pilot_experiment/
│       └── <model_name>/
│           └── <domain>_task/
│               └── <task_id>/
│                   ├── human-eval.json      # Human evaluation scores
│                   └── GPT4OEvaluator.json  # GPT-4O evaluation scores
│
└── data_logging/                    # Version tracking
    ├── version_log.json            # Version history
    └── versions/                   # Version snapshots
```

## S3 Synchronization

### Quick Start

Upload your dataset to S3:

```bash
# Basic upload (uses timestamp: YYYYMMDDHHMM)
python data/s3_sync.py

# Upload and log version
python data/s3_sync.py --log

# Upload with specific date
python data/s3_sync.py --date 20250115

# Future: Download from S3 (to be implemented)
# python data/s3_sync.py --download --date 20250115
```

### Configuration

Set up AWS credentials in `.env`:

```bash
# Required AWS credentials
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret

# Optional configuration
S3_BUCKET=vmevalkit              # Default bucket name
AWS_DEFAULT_REGION=us-east-2     # Default region
AWS_REGION=us-east-2             # Alternative region setting
```

### S3 Structure

Data is organized by timestamp on S3:
```
s3://vmevalkit/
├── 202501151030/                  # YYYYMMDDHHMM format
│   └── data/
│       ├── questions/
│       ├── outputs/
│       └── evaluations/
├── 202501141500/
│   └── data/
└── latest/                        # Symbolic link to most recent
```

### Python API

```python
from data.s3_sync import sync_to_s3

# Upload to S3 with automatic timestamp
s3_uri = sync_to_s3()
print(f"Data uploaded to: {s3_uri}")

# Upload with custom date and specific directory
s3_uri = sync_to_s3(
    data_dir=Path("data/outputs"),
    date_prefix="202501151030"
)

# Progress monitoring (uploads show status every 100 files)
# ↳ 100 files...
# ↳ 200 files...
# ✅ Uploaded 1300 files (180.5 MB)
```

### Advanced Features (Recommended Implementations)

```python
# Download from S3 (to be implemented)
def download_from_s3(date_prefix=None, target_dir=None):
    """Download dataset from S3 to local directory."""
    # Implementation needed
    pass

# Incremental sync (to be implemented)
def incremental_sync(since_date=None):
    """Only upload changed files since last sync."""
    # Implementation needed
    pass

# Compression before upload (to be implemented)
def sync_with_compression(compression='gzip'):
    """Compress data before uploading to save bandwidth."""
    # Implementation needed
    pass
```

## Version Tracking

VMEvalKit includes built-in version tracking for datasets.

### Logging Versions

```bash
# View version history
python data/data_logging/version_tracker.py summary

# Get latest version
python data/data_logging/version_tracker.py latest

# Example output:
# 📊 Dataset Versions
# ========================================
# v1.0 (20250114) → s3://vmevalkit/202501141500/data
#   180.5MB, 1300 files
# v1.1 (20250115) → s3://vmevalkit/202501151030/data
#   195.2MB, 1450 files
```

### Python API

```python
from data.data_logging import log_version, get_latest, print_summary
from data.data_logging.version_tracker import load_log, save_log

# Log a new version with metadata
log_version(
    version="1.2",
    s3_uri="s3://vmevalkit/202501151030/data",
    stats={
        "size_mb": 195.2,
        "files": 1450,
        "change": "Added 50 new chess puzzles",
        "domains": {
            "chess": 65,
            "maze": 50,
            "sudoku": 50,
            "rotation": 50,
            "raven": 50
        }
    }
)

# Get latest version info
latest = get_latest()
if latest:
    print(f"Latest: v{latest['version']} at {latest['s3_uri']}")
    print(f"Size: {latest.get('size_mb', 0):.1f}MB")
    print(f"Files: {latest.get('files', 0)}")

# Print comprehensive version summary
print_summary()

# Advanced: Force overwrite conflicting version
log = load_log()
log['versions'] = [v for v in log['versions'] if v['version'] != '1.0']
save_log(log)
```

### Version Log Format

Versions are stored in `data/data_logging/version_log.json`:

```json
{
  "versions": [
    {
      "version": "1.0",
      "date": "20250115",
      "s3_uri": "s3://vmevalkit/202501151030/data",
      "size_mb": 180.5,
      "files": 1300,
      "timestamp": "2025-01-15T10:30:00.123456",
      "stats": {
        "change": "Initial dataset",
        "domains": {
          "chess": 15,
          "maze": 15,
          "sudoku": 15,
          "rotation": 15,
          "raven": 15
        }
      }
    }
  ]
}
```

## Dataset Creation

### Generate Questions Dataset

Create a new dataset with specified tasks:

```bash
# Generate standard dataset (15 questions per domain, 75 total)
python -m vmevalkit.runner.create_dataset --pairs-per-domain 15

# Custom configuration with specific domains
python -m vmevalkit.runner.create_dataset \
    --pairs-per-domain 20 \
    --random-seed 42

# Read existing dataset without generation
python -m vmevalkit.runner.create_dataset --read-only
```

### Domain Registry System

VMEvalKit uses a flexible domain registry for easy task extension:

```python
# In vmevalkit/runner/create_dataset.py
DOMAIN_REGISTRY = {
    'chess': {
        'emoji': '♟️',
        'name': 'Chess',
        'description': 'Strategic thinking and tactical pattern recognition',
        'module': 'vmevalkit.tasks.chess_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    'maze': {
        'emoji': '🌀',
        'name': 'Maze',
        'description': 'Spatial reasoning and navigation planning',
        'module': 'vmevalkit.tasks.maze_task',
        'create_function': 'create_dataset',
        'process_dataset': lambda dataset, num_samples: dataset['pairs']
    },
    # Add new domains here...
}
```

### Adding New Task Domains

To add a new reasoning domain:

1. Create task module in `vmevalkit/tasks/your_task/`
2. Implement `create_dataset()` function
3. Register in `DOMAIN_REGISTRY`
4. Generate dataset with new domain

Example:
```python
# vmevalkit/tasks/physics_task/__init__.py
def create_dataset(num_samples=15):
    """Generate physics reasoning tasks."""
    pairs = []
    for i in range(num_samples):
        # Generate physics simulation pairs
        pass
    return {"pairs": pairs}

# Add to DOMAIN_REGISTRY
'physics': {
    'emoji': '⚛️',
    'name': 'Physics',
    'description': 'Physical reasoning and simulation',
    'module': 'vmevalkit.tasks.physics_task',
    'create_function': 'create_dataset',
    'process_dataset': lambda dataset, num_samples: dataset['pairs']
}
```

### Dataset Manifest

The master manifest (`vmeval_dataset.json`) tracks all questions:

```json
{
  "name": "vmeval_dataset",
  "description": "VMEvalKit video reasoning evaluation dataset (75 task pairs)",
  "created_at": "2025-01-15T10:30:00.123456Z",
  "total_pairs": 75,
  "generation_info": {
    "timestamp": "2025-01-15T10:30:00.123456Z",
    "random_seed": 42,
    "pairs_per_domain": 15,
    "target_pairs": 75,
    "actual_pairs": 75,
    "allocation": {
      "chess": 15,
      "maze": 15,
      "sudoku": 15,
      "rotation": 15,
      "raven": 15
    },
    "domains": {
      "chess": {
        "count": 15,
        "description": "Strategic thinking and tactical pattern recognition"
      },
      "maze": {
        "count": 15,
        "description": "Spatial reasoning and navigation planning"
      }
    }
  },
  "pairs": [
    {
      "id": "chess_0000",
      "domain": "chess",
      "difficulty": "medium",
      "task_category": "mate_in_1",
      "first_image_path": "chess_task/chess_0000/first_frame.png",
      "final_image_path": "chess_task/chess_0000/final_frame.png",
      "prompt": "White to move and checkmate in 1. Move the rook from a1 to a8.",
      "created_at": "2025-01-15T10:30:00.123456Z",
      "metadata": {
        "fen_initial": "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",
        "fen_final": "R5k1/5ppp/8/8/8/8/8/7K w - - 1 1",
        "solution": "Ra8#"
      }
    }
  ]
}
```

## Web Dashboard

VMEvalKit includes an interactive web dashboard for data visualization:

### Starting the Dashboard

```bash
# Navigate to web directory
cd web

# Start the dashboard
python app.py

# Or use the start script
./start.sh

# Access at: http://localhost:5000
```

### Dashboard Features

- **Hierarchical Navigation**: Models → Domains → Tasks
- **Video Playback**: View generated videos with lazy loading
- **Image Comparison**: Side-by-side initial/final frames
- **Model Performance**: Success rates and statistics
- **Quick Navigation**: Jump buttons for each model
- **API Access**: REST endpoints for programmatic access
- **Deduplication**: Automatically shows most recent runs

### API Endpoints

```bash
# Get all results
GET http://localhost:5000/api/results

# Filter by model
GET http://localhost:5000/api/results?model=luma-ray-2

# Filter by domain  
GET http://localhost:5000/api/results?domain=chess

# Filter by task
GET http://localhost:5000/api/results?task_id=maze_0001

# Get specific media
GET http://localhost:5000/media/video/{model}/{domain}/{task_id}/{run_id}/{filename}
```

### Dashboard Configuration

```python
# web/app.py configuration
app.config['OUTPUT_DIR'] = Path('../data/outputs/pilot_experiment')
app.config['QUESTIONS_DIR'] = Path('../data/questions')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
```

## Data Flow

### Complete Pipeline

```
1. Question Creation
   Task Generator → data/questions/ → vmeval_dataset.json → S3 Backup

2. Inference Pipeline  
   Questions → Model API → data/outputs/{experiment}/ → S3 Sync

3. Evaluation Pipeline
   Outputs → Human/GPT-4O Eval → data/evaluations/ → S3 Sync

4. Visualization
   All Data → Web Dashboard → Interactive Analysis
```

### Experiment Management

VMEvalKit organizes results by experiment:

```python
# Define experiment name during inference
experiment_name = "pilot_experiment"  # Or "ablation_study", "final_run", etc.

# Results are organized under:
data/outputs/{experiment_name}/
data/evaluations/{experiment_name}/

# This allows parallel experiments without conflicts
```

## Data Validation & Integrity

### Dataset Validation

```python
from vmevalkit.runner.create_dataset import read_dataset_from_folders

# Read and validate existing dataset
dataset = read_dataset_from_folders()

# Check dataset integrity
def validate_dataset(dataset):
    """Validate dataset structure and files."""
    issues = []
    
    for pair in dataset['pairs']:
        # Check required fields
        if not pair.get('id'):
            issues.append(f"Missing ID in pair")
        
        # Check file existence
        base = Path("data/questions")
        first_img = base / pair.get('first_image_path', '')
        if not first_img.exists():
            issues.append(f"Missing image: {first_img}")
    
    return issues

issues = validate_dataset(dataset)
if issues:
    print(f"Found {len(issues)} issues:")
    for issue in issues[:10]:
        print(f"  - {issue}")
```

### Pre-experiment Checklist

```python
def pre_experiment_check():
    """Validate data before running experiments."""
    checks = {
        'dataset_exists': Path("data/questions/vmeval_dataset.json").exists(),
        'images_complete': check_all_images(),
        'prompts_valid': validate_prompts(),
        's3_configured': check_s3_credentials(),
        'disk_space': check_disk_space() > 10_000  # MB
    }
    
    for check, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check}")
    
    return all(checks.values())
```

## Best Practices

### 1. Regular Backups

```bash
# Daily backup with version logging
python data/s3_sync.py --log

# Automated backup script
#!/bin/bash
cd /path/to/VMEvalKit
source venv/bin/activate
python data/s3_sync.py --log
echo "Backup completed: $(date)" >> backup.log
```

### 2. Version Before Major Changes

```python
from data.data_logging import log_version

# Before adding new tasks
log_version("1.1-pre", s3_uri, {"change": "Before adding 50 chess puzzles"})

# After adding new tasks
log_version("1.1", s3_uri, {"change": "Added 50 new chess puzzles"})
```

### 3. Data Integrity

- Keep `vmeval_dataset.json` in sync with actual files
- Validate dataset structure before experiments
- Use version tracking for reproducibility
- Verify file counts match expectations

### 4. Storage Management

```python
# Archive old experiments
def archive_old_experiments(days_old=30):
    """Move old experiments to S3 and remove locally."""
    cutoff = datetime.now() - timedelta(days=days_old)
    
    for exp_dir in Path("data/outputs").iterdir():
        if exp_dir.stat().st_mtime < cutoff.timestamp():
            # Upload to S3
            s3_uri = sync_to_s3(exp_dir, f"archive/{exp_dir.name}")
            # Remove locally
            shutil.rmtree(exp_dir)
            print(f"Archived {exp_dir.name} to {s3_uri}")

# Clean up temporary files
def cleanup_temp_files():
    """Remove temporary and cache files."""
    patterns = ['*.tmp', '*.cache', '__pycache__', '.DS_Store']
    for pattern in patterns:
        for file in Path("data").rglob(pattern):
            file.unlink() if file.is_file() else shutil.rmtree(file)
```

### 5. Performance Optimization

```python
# Batch operations for large datasets
def batch_upload(batch_size=100):
    """Upload files in batches for better performance."""
    files = list(Path("data").rglob("*"))
    
    for i in range(0, len(files), batch_size):
        batch = files[i:i+batch_size]
        # Process batch
        upload_batch(batch)
        
# Use multiprocessing for parallel operations
from multiprocessing import Pool

def parallel_validate(dataset, num_workers=4):
    """Validate dataset using multiple workers."""
    with Pool(num_workers) as pool:
        results = pool.map(validate_single_item, dataset['pairs'])
    return results
```

## HuggingFace Integration

Mirror VMEvalKit datasets to the HuggingFace Hub and restore them back locally in the exact same folder structure. This complements the S3-based flow and ensures formats remain consistent with VMEvalKit's expectations.

### What this enables
- Upload any dataset folder (e.g., `data/questions/`) to a HuggingFace dataset repo, preserving directory layout and filenames.
- Download a complete snapshot from HuggingFace back to a local folder with identical structure.
- Non-interactive CLI and Python API, suitable for automation and CI.

### Install and auth
Dependencies are included in `requirements.txt`:
- `huggingface_hub`
- `datasets`
- `pyarrow`

Authenticate using an environment variable:
```bash
export HF_TOKEN=hf_xxx_your_token_here
```

### CLI usage
The `data/hf_sync.py` utility provides three commands: `upload`, `download`, and `ls`.

#### Upload a local folder
```bash
python data/hf_sync.py upload \
  --path data/questions \
  --repo-id your-username/vmevalkit-questions \
  --private \
  --commit-message "Upload questions set" \
  --revision main
```

#### Download to a local folder
```bash
python data/hf_sync.py download \
  --repo-id your-username/vmevalkit-questions \
  --target data/questions \
  --revision main
```

#### List files in a repo
```bash
python data/hf_sync.py ls \
  --repo-id your-username/vmevalkit-questions \
  --revision main
```

Notes:
- Repos are created automatically if they do not exist (`--private` respected).
- Structure is mirrored exactly, so VMEvalKit can use the data immediately.
- Use separate repos for large artifacts if desired (e.g., `...-outputs`, `...-evaluations`).

### Python API
```python
from pathlib import Path
from data.hf_sync import hf_upload, hf_download, hf_list_files

# Upload folder
hf_upload(
    local_path=Path("data/questions"),
    repo_id="your-username/vmevalkit-questions",
    private=True,
    commit_message="Upload questions",
)

# Download snapshot
out_dir = hf_download(
    repo_id="your-username/vmevalkit-questions",
    target_dir=Path("data/questions"),
)

# List files
files = hf_list_files(repo_id="your-username/vmevalkit-questions")
print(len(files), "files on hub")
```

### Format consistency
The sync preserves VMEvalKit's layout:
- `data/questions/{domain}_task/{task_id}/` containing `first_frame.png`, `final_frame.png`, `prompt.txt`, and `question_metadata.json`
- `data/outputs/{experiment}/...` and `data/evaluations/{experiment}/...` if you choose to sync those as well

This guarantees seamless use with:
- Runners (`vmevalkit.runner.InferenceRunner`)
- Web dashboard (`web/app.py`)
- Scoring pipelines (`examples/score_videos.py`)

### Optional: Tabular datasets on the Hub
If you prefer a tabular dataset (for metadata-centric workflows), you can still convert `vmeval_dataset.json` to a HuggingFace `datasets` object and push it. For complete reproducibility of files, the folder mirroring approach above is recommended.

```python
import json
from datasets import Dataset, load_dataset
import pandas as pd

# Convert VMEvalKit dataset to HuggingFace format
def to_huggingface_dataset():
    # Load VMEvalKit dataset
    with open("data/questions/vmeval_dataset.json") as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data['pairs'])
    
    # Create HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    
    # Add metadata
    dataset.info.description = data['description']
    dataset.info.version = data.get('version', '1.0')
    
    return dataset

# Upload to HuggingFace
dataset = to_huggingface_dataset()
dataset.push_to_hub(
    "your-username/vmevalkit-questions",
    private=False,
    commit_message="Update VMEvalKit dataset"
)

# Download from HuggingFace
dataset = load_dataset("your-username/vmevalkit-questions")

# Use in inference
for item in dataset:
    process_task(item)
```

### Best practices
- Use separate repos for different data categories if they are very large.
- Keep your `HF_TOKEN` in a secure secret store for CI.
- Prefer `download` when collaborating to ensure a consistent layout across machines.

## Troubleshooting

### Common Issues

1. **S3 Upload Fails**
   ```python
   # Check credentials
   import boto3
   try:
       s3 = boto3.client('s3')
       s3.list_buckets()
       print("✅ AWS credentials valid")
   except Exception as e:
       print(f"❌ AWS error: {e}")
   
   # Verify permissions
   s3.head_bucket(Bucket='vmevalkit')
   ```

2. **Version Conflict**
   ```python
   # Force overwrite version
   from data.data_logging.version_tracker import load_log, save_log
   
   log = load_log()
   # Remove conflicting version
   log['versions'] = [v for v in log['versions'] if v['version'] != '1.0']
   # Add new version
   log['versions'].append({...})
   save_log(log)
   ```

3. **Large Dataset Upload**
   ```python
   # Use multipart upload for files > 100MB
   from boto3.s3.transfer import TransferConfig
   
   config = TransferConfig(
       multipart_threshold=1024 * 25,  # 25MB
       max_concurrency=10,
       multipart_chunksize=1024 * 25,
       use_threads=True
   )
   
   s3.upload_file(file, bucket, key, Config=config)
   ```

4. **Dashboard Not Loading**
   ```bash
   # Check port availability
   lsof -i :5000
   
   # Use different port
   python app.py --port 8080
   
   # Check data directory
   ls -la ../data/outputs/pilot_experiment/
   ```

5. **Memory Issues with Large Datasets**
   ```python
   # Stream large files instead of loading all
   def stream_large_dataset(filepath):
       with open(filepath) as f:
           for line in f:
               yield json.loads(line)
   
   # Process in chunks
   for chunk in pd.read_json(filepath, lines=True, chunksize=1000):
       process_chunk(chunk)
   ```

## CLI Commands Summary

| Command | Description |
|---------|-------------|
| `python data/s3_sync.py` | Upload data to S3 |
| `python data/s3_sync.py --log` | Upload and log version |
| `python data/s3_sync.py --date YYYYMMDDHHMM` | Upload with specific timestamp |
| `python data/data_logging/version_tracker.py summary` | View version history |
| `python data/data_logging/version_tracker.py latest` | Get latest version |
| `python -m vmevalkit.runner.create_dataset --pairs-per-domain N` | Generate dataset |
| `python -m vmevalkit.runner.create_dataset --read-only` | Read existing dataset |
| `python web/app.py` | Start web dashboard |
| `./web/start.sh` | Start dashboard with script |
| `python data/hf_sync.py upload --path data/questions --repo-id <repo>` | Upload to HF |
| `python data/hf_sync.py download --repo-id <repo> --target data/questions` | Download from HF |
| `python data/hf_sync.py ls --repo-id <repo>` | List HF repo files |

## Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `AWS_ACCESS_KEY_ID` | AWS access key | For S3 | - |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key | For S3 | - |
| `S3_BUCKET` | S3 bucket name | For S3 | vmevalkit |
| `AWS_DEFAULT_REGION` | AWS region | For S3 | us-east-2 |
| `AWS_REGION` | Alternative region setting | For S3 | us-east-2 |
| `SECRET_KEY` | Flask secret key | For web | auto-generated |
| `HF_TOKEN` | HuggingFace access token | For HF | - |

## Data Lifecycle Management

### Retention Policies

```python
# Define retention policies
RETENTION_POLICIES = {
    'questions': None,  # Keep forever
    'outputs': 90,     # Keep for 90 days
    'evaluations': 180, # Keep for 180 days
    'temp': 7          # Keep for 7 days
}

def apply_retention_policies():
    """Apply data retention policies."""
    for data_type, days in RETENTION_POLICIES.items():
        if days:
            cleanup_old_data(f"data/{data_type}", days)
```

### Data Migration

```python
# Migrate data between storage systems
def migrate_to_new_structure():
    """Migrate from old to new data structure."""
    old_path = Path("data/old_structure")
    new_path = Path("data/questions")
    
    for old_file in old_path.rglob("*.json"):
        # Transform to new format
        new_data = transform_format(old_file)
        # Save in new location
        save_new_format(new_data, new_path)
```

## Related Documentation

- [INFERENCE.md](INFERENCE.md) - How inference results are stored and managed
- [EVALUATION.md](EVALUATION.md) - How evaluation results are organized and analyzed
- [ADDING_TASKS.md](ADDING_TASKS.md) - Creating new task datasets and domains
- [WEB_DASHBOARD.md](WEB_DASHBOARD.md) - Using the web dashboard for visualization

## Future Enhancements

### Planned Features

1. **S3 Download/Restore**: Implement bidirectional sync with S3
2. **Incremental Sync**: Only upload changed files
3. **Compression**: Automatic compression before upload
4. **Data Validation UI**: Web interface for dataset validation
5. **Automated Backups**: Cron-based backup scheduling
6. **Dataset Versioning UI**: Web interface for version management
7. **Multi-cloud Support**: Support for GCS, Azure Blob Storage
8. **Data Lineage Tracking**: Track dataset transformations
9. **Automated Cleanup**: Smart cleanup based on usage patterns
10. **Real-time Sync**: Watch for changes and auto-sync

### Contributing

To contribute data management improvements:

1. Check existing issues on GitHub
2. Propose new features via issue discussion
3. Implement with tests and documentation
4. Submit pull request with clear description

---

*Last updated: November 2024*
*VMEvalKit Team*