# Dataset Version Logging

Simple S3 version tracking for VMEvalKit datasets.

## Structure
```
data_logging/
├── version_log.json      # Version history
└── versions/             # Detailed logs (auto-generated)
```

## Usage

### Upload to S3
```bash
# Upload with today's date
python data/s3_sync.py

# Upload and log version
python data/s3_sync.py --log
```

### View versions
```bash
python data/data_logging/version_tracker.py summary
```

### From Python
```python
from data.data_logging import log_version, get_latest

# Log a version
log_version("1.0", "s3://vmevalkit/20251015/data", {"size_mb": 180, "files": 1300})

# Get latest
latest = get_latest()
```

## S3 Structure
Data is stored at: `s3://vmevalkit/YYYYMMDD/data`