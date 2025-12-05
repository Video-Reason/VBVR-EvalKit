#!/usr/bin/env python3
"""S3 sync for VMEvalKit data - upload, download, and list datasets."""

import os
import datetime
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import subprocess
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


def get_bucket_name() -> str:
    """Get S3 bucket name from environment."""
    return os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET", "vmevalkit")


def get_username() -> Optional[str]:
    """Get S3 username prefix from environment."""
    return os.getenv("AWS_S3_USERNAME") or os.getenv("S3_USERNAME")


def upload_to_s3(local_path: str, s3_prefix: str = None, date_prefix: str = None) -> str:
    """
    Upload a file or directory to S3.
    
    Args:
        local_path: Local file or directory path to upload
        s3_prefix: S3 prefix/folder (if not provided, uses username/date_prefix/data format)
        date_prefix: Date prefix in YYYYMMDDHHMM format (for backward compatibility)
        
    Returns:
        S3 URI of uploaded data
    """
    if local_path is None:
        raise ValueError("local_path cannot be None")
    
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Path does not exist: {local_path}")
    
    bucket = get_bucket_name()
    username = get_username()
    
    # Maintain original YYYYMMDDHHMM/data/ format for backward compatibility
    # Add username prefix if configured (only when using default prefix)
    if s3_prefix is None:
        date_folder = date_prefix or datetime.datetime.now().strftime("%Y%m%d%H%M")
        s3_prefix = f"{date_folder}/data"
        if username:
            s3_prefix = f"{username}/{s3_prefix}"
    
    # Calculate file count and total size before upload
    file_count = 0
    total_size = 0
    
    if local_path.is_file():
        file_count = 1
        total_size = local_path.stat().st_size
    else:
        for root, _, files in os.walk(local_path):
            for filename in files:
                file_path = Path(root) / filename
                file_count += 1
                total_size += file_path.stat().st_size
    
    s3_uri = f"s3://{bucket}/{s3_prefix}"
    
    print(f"📤 Uploading to {s3_uri}/")
    
    # Use AWS CLI to upload
    if local_path.is_file():
        # Single file: use cp
        s3_dest = f"{s3_uri}/{local_path.name}"
        cmd = ["aws", "s3", "cp", str(local_path), s3_dest]
        subprocess.run(cmd, check=True)
        print(f"  ✓ {local_path.name}")
    else:
        # Directory: use sync
        s3_dest = f"{s3_uri}/"
        cmd = ["aws", "s3", "sync", str(local_path), s3_dest]
        subprocess.run(cmd, check=True)
    
    size_mb = total_size / (1024 * 1024)
    
    print(f"✅ Uploaded {file_count} file(s) ({size_mb:.1f} MB)")
    print(f"📍 Location: {s3_uri}")
    
    return s3_uri


def download_from_s3(s3_uri: str, local_path: str = None) -> str:
    """
    Download data from S3.
    
    Args:
        s3_uri: S3 URI (s3://bucket/prefix) or just the prefix
        local_path: Local destination path (default: ./downloads/{prefix})
        
    Returns:
        Local path where data was downloaded
    """
    # Parse S3 URI
    if s3_uri.startswith("s3://"):
        full_s3_uri = s3_uri
    else:
        bucket = get_bucket_name()
        full_s3_uri = f"s3://{bucket}/{s3_uri}"
    
    # Set default local path
    if local_path is None:
        if s3_uri.startswith("s3://"):
            parts = s3_uri[5:].split("/", 1)
            s3_prefix = parts[1] if len(parts) > 1 else ""
        else:
            s3_prefix = s3_uri
        prefix_name = s3_prefix.rstrip("/").split("/")[-1] or "data"
        local_path = Path(__file__).parent / "downloads" / prefix_name
    else:
        local_path = Path(local_path)
    
    Path(local_path).mkdir(parents=True, exist_ok=True)
    
    # AWS CLI parallel download
    cmd = [
        "aws", "s3", "sync",
        full_s3_uri,
        str(local_path)
    ]
    
    print(f"📥 Downloading from {full_s3_uri}")
    subprocess.run(cmd, check=True)
    
    return str(local_path)


def list_s3_datasets(bucket: str = None, prefix: str = "") -> List[Dict]:
    """
    List available datasets in S3.
    
    Args:
        bucket: S3 bucket name (default: from environment)
        prefix: Filter by prefix (optional)
        
    Returns:
        List of dataset information dicts
    """
    bucket = bucket or get_bucket_name()
    
    # Build S3 URI
    s3_uri = f"s3://{bucket}"
    if prefix:
        s3_uri = f"{s3_uri}/{prefix.rstrip('/')}"
    
    print(f"📋 Listing datasets in {s3_uri}")
    print()
    
    # Use AWS CLI to list files recursively
    cmd = ["aws", "s3", "ls", s3_uri, "--recursive"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    folder_stats = {}
    
    # Parse output: format is "YYYY-MM-DD HH:MM:SS      SIZE path/to/file"
    # Note: aws s3 ls output does NOT include s3://bucket/ prefix, only relative path
    for line in result.stdout.strip().split('\n'):
        if not line.strip():
            continue
        
        # Parse line: "2024-01-01 12:00:00     1234567 path/to/file"
        parts = line.split(None, 3)  # Split into max 4 parts: date, time, size, path
        if len(parts) < 4:
            continue
        
        # Extract relative path (may contain spaces, so use parts[3])
        relative_path = parts[3]
        
        # Apply prefix filter if specified
        if prefix:
            prefix_stripped = prefix.rstrip('/')
            if not relative_path.startswith(prefix_stripped):
                continue
            # Remove prefix from path for dataset name extraction
            relative_path = relative_path[len(prefix_stripped):].lstrip('/')
        
        # Skip if it's just the prefix (directory marker) or empty
        if not relative_path or relative_path.endswith('/'):
            continue
        
        # Extract top-level folder (dataset name)
        path_parts = relative_path.split('/')
        if len(path_parts) > 1:
            dataset_name = path_parts[0]
            
            # Parse file size
            file_size = int(parts[2])
            
            # Parse date and time
            date_str = parts[0]
            time_str = parts[1]
            last_modified = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
            
            if dataset_name not in folder_stats:
                folder_stats[dataset_name] = {
                    'name': dataset_name,
                    'file_count': 0,
                    'total_size': 0,
                    'last_modified': last_modified
                }
            
            folder_stats[dataset_name]['file_count'] += 1
            folder_stats[dataset_name]['total_size'] += file_size
            
            # Keep the most recent modification time
            if last_modified > folder_stats[dataset_name]['last_modified']:
                folder_stats[dataset_name]['last_modified'] = last_modified
    
    # Convert to sorted list
    datasets = sorted(folder_stats.values(), key=lambda x: x['last_modified'], reverse=True)
    
    # Print results
    if not datasets:
        print("No datasets found.")
        return []
    
    print(f"{'Dataset':<30} {'Files':<10} {'Size':<15} {'Last Modified':<20}")
    print("-" * 80)
    
    for ds in datasets:
        size_mb = ds['total_size'] / (1024 * 1024)
        size_str = f"{size_mb:.1f} MB" if size_mb < 1024 else f"{size_mb/1024:.1f} GB"
        modified_str = ds['last_modified'].strftime("%Y-%m-%d %H:%M")
        
        print(f"{ds['name']:<30} {ds['file_count']:<10} {size_str:<15} {modified_str:<20}")
        ds['s3_uri'] = f"s3://{bucket}/{ds['name']}"
    
    print()
    print(f"Total datasets: {len(datasets)}")
    
    return datasets


def sync_to_s3(data_dir: Path = None, date_prefix: str = None) -> str:
    """
    Sync data folder to S3 (legacy function, uses upload_to_s3).
    
    Args:
        data_dir: Path to data directory (default: ./data)
        date_prefix: Date folder (default: today's date YYYYMMDDHHMM)
        
    Returns:
        S3 URI of uploaded data
    """
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent
    
    date_folder = date_prefix or datetime.datetime.now().strftime("%Y%m%d%H%M")
    username = get_username()
    
    if username:
        s3_prefix = f"{username}/{date_folder}/data"
    else:
        s3_prefix = f"{date_folder}/data"
    
    return upload_to_s3(str(data_dir), s3_prefix=s3_prefix)


def main():
    parser = argparse.ArgumentParser(
        description="S3 sync for VMEvalKit - upload, download, and list datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Add backward-compatible flags at top level (for original usage)
    parser.add_argument('--date', help='Date folder (YYYYMMDDHHMM) - for backward compatibility')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    upload_parser = subparsers.add_parser('upload', help='Upload files/directory to S3')
    upload_parser.add_argument('path', help='Local file or directory to upload')
    upload_parser.add_argument('--prefix', help='S3 prefix/folder (default: YYYYMMDDHHMM/data)')
    upload_parser.add_argument('--bucket', help='S3 bucket (default: from environment)')
    upload_parser.add_argument('--date', help='Date prefix (YYYYMMDDHHMM)')
    
    download_parser = subparsers.add_parser('download', help='Download data from S3')
    download_parser.add_argument('s3_uri', help='S3 URI (s3://bucket/prefix) or just prefix')
    download_parser.add_argument('--output', help='Local destination path (default: ./downloads/{prefix})')
    
    list_parser = subparsers.add_parser('list', help='List available S3 datasets')
    list_parser.add_argument('--prefix', default='', help='Filter by prefix')
    list_parser.add_argument('--bucket', help='S3 bucket (default: from environment)')
    
    # Sync command (for explicit sync usage)
    sync_parser = subparsers.add_parser('sync', help='Sync data folder to S3')
    sync_parser.add_argument('--date', help='Date folder (YYYYMMDDHHMM)')
    
    args = parser.parse_args()
    
    # BACKWARD COMPATIBILITY: Default to sync if no command provided (original behavior)
    # This preserves the original design where `python data/s3_sync.py` just uploads
    if args.command is None:
        if args.date:
            # Original usage: python data/s3_sync.py --date 202411181530
            print("📦 Running sync (original usage)")
        else:
            # Default: python data/s3_sync.py
            print("📦 Running default sync (use --help to see new commands)")
        args.command = 'sync'

    if args.command == 'upload':
        date_prefix = getattr(args, 'date', None)
        s3_uri = upload_to_s3(args.path, s3_prefix=args.prefix, bucket=args.bucket, date_prefix=date_prefix)
        print(f"Uploaded to {s3_uri}")
        return 0
    elif args.command == 'download':
        local_path = download_from_s3(args.s3_uri, local_path=args.output)
        print(f"Downloaded to {local_path}")
        return 0
    elif args.command == 'list':
        list_s3_datasets(bucket=args.bucket, prefix=args.prefix)
        return 0
    elif args.command == 'sync':
        s3_uri = sync_to_s3(date_prefix=args.date)
        return 0
    else:
        print(f"Unknown command: {args.command}")
        return 1


if __name__ == "__main__":
    main()