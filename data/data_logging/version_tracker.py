#!/usr/bin/env python3
"""Simple S3 dataset version tracker."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Paths
DATA_LOGGING_DIR = Path(__file__).parent
VERSION_LOG_PATH = DATA_LOGGING_DIR / "version_log.json"
VERSIONS_DIR = DATA_LOGGING_DIR / "versions"


def load_log() -> Dict:
    """Load version log."""
    # Ensure directories exist
    DATA_LOGGING_DIR.mkdir(exist_ok=True)
    VERSIONS_DIR.mkdir(exist_ok=True)
    
    if VERSION_LOG_PATH.exists():
        try:
            with open(VERSION_LOG_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Warning: Could not read version log ({e}), creating new one")
    return {'versions': []}


def save_log(log: Dict) -> None:
    """Save version log."""
    try:
        with open(VERSION_LOG_PATH, 'w') as f:
            json.dump(log, f, indent=2)
    except IOError as e:
        print(f"❌ Failed to save version log: {e}")
        raise


def log_version(version: str, s3_uri: str, stats: Dict) -> None:
    """Log a new dataset version."""
    log = load_log()
    
    # Check if already exists
    for v in log['versions']:
        if v['version'] == version:
            print(f"⚠️  Version {version} already exists - skipping")
            return
    
    # Add version
    log['versions'].append({
        'version': version,
        'date': datetime.now().strftime('%Y%m%d'),
        's3_uri': s3_uri,
        'size_mb': stats.get('size_mb', 0),
        'files': stats.get('files', 0),
        'timestamp': datetime.now().isoformat()
    })
    
    save_log(log)
    print(f"✅ Logged v{version} → {s3_uri}")


def get_latest() -> Optional[Dict]:
    """Get latest version."""
    log = load_log()
    return log['versions'][-1] if log['versions'] else None


def print_summary() -> None:
    """Print version summary."""
    log = load_log()
    if not log['versions']:
        print("📝 No versions logged yet")
        return
    
    print("\n📊 Dataset Versions")
    print("=" * 50)
    for v in log['versions']:
        print(f"📦 v{v['version']} ({v['date']}) → {v['s3_uri']}")
        print(f"   💾 {v.get('size_mb', 0):.1f}MB, {v.get('files', 0)} files")
    print("=" * 50)


def main() -> None:
    """CLI entry point."""
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        print_summary()
    elif len(sys.argv) > 1 and sys.argv[1] == 'latest':
        latest = get_latest()
        if latest:
            print(f"Latest: v{latest['version']} → {latest['s3_uri']}")
    else:
        print_summary()


if __name__ == "__main__":
    main()
