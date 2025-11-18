#!/usr/bin/env python3
"""S3 sync for VMEvalKit data - upload, download, and list datasets."""

import os
import sys
import datetime
from pathlib import Path
from typing import List, Dict, Optional
import boto3
from botocore.exceptions import ClientError

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


def get_s3_client():
    """Create and return an S3 client with credentials from environment."""
    return boto3.client("s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-2")
    )


def get_bucket_name() -> str:
    """Get S3 bucket name from environment."""
    return os.getenv("AWS_S3_BUCKET") or os.getenv("S3_BUCKET", "vmevalkit")


def upload_to_s3(local_path: str, s3_prefix: str = None, bucket: str = None, date_prefix: str = None) -> str:
    """
    Upload a file or directory to S3.
    
    Args:
        local_path: Local file or directory path to upload
        s3_prefix: S3 prefix/folder (if not provided, uses date_prefix/data format)
        bucket: S3 bucket name (default: from environment)
        date_prefix: Date prefix in YYYYMMDDHHMM format (for backward compatibility)
        
    Returns:
        S3 URI of uploaded data
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Path does not exist: {local_path}")
    
    bucket = bucket or get_bucket_name()
    
    # Maintain original YYYYMMDDHHMM/data/ format for backward compatibility
    if s3_prefix is None:
        date_folder = date_prefix or datetime.datetime.now().strftime("%Y%m%d%H%M")
        s3_prefix = f"{date_folder}/data"
    
    s3 = get_s3_client()
    
    file_count = 0
    total_size = 0
    
    print(f"📤 Uploading to s3://{bucket}/{s3_prefix}/")
    
    # Upload single file or directory
    if local_path.is_file():
        s3_key = f"{s3_prefix}/{local_path.name}"
        s3.upload_file(str(local_path), bucket, s3_key)
        file_count = 1
        total_size = local_path.stat().st_size
        print(f"  ✓ {local_path.name}")
    else:
        for root, _, files in os.walk(local_path):
            for filename in files:
                file_path = Path(root) / filename
                rel_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{rel_path.as_posix()}"
                
                s3.upload_file(str(file_path), bucket, s3_key)
                file_count += 1
                total_size += file_path.stat().st_size
                
                if file_count % 50 == 0:
                    print(f"  ↳ {file_count} files...")
    
    s3_uri = f"s3://{bucket}/{s3_prefix}"
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
        parts = s3_uri[5:].split("/", 1)
        bucket = parts[0]
        s3_prefix = parts[1] if len(parts) > 1 else ""
    else:
        bucket = get_bucket_name()
        s3_prefix = s3_uri
    
    # Set default local path
    if local_path is None:
        prefix_name = s3_prefix.rstrip("/").split("/")[-1] or "data"
        local_path = Path(__file__).parent / "downloads" / prefix_name
    else:
        local_path = Path(local_path)
    
    local_path.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()
    
    print(f"📥 Downloading from s3://{bucket}/{s3_prefix}/")
    print(f"📂 Destination: {local_path}")
    
    file_count = 0
    total_size = 0
    
    # List and download all objects with the prefix
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            s3_key = obj['Key']
            
            # Skip if it's just the prefix (directory marker)
            if s3_key == s3_prefix or s3_key.endswith('/'):
                continue
            
            # Determine local file path
            rel_path = s3_key[len(s3_prefix):].lstrip('/')
            local_file = local_path / rel_path
            local_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            s3.download_file(bucket, s3_key, str(local_file))
            file_count += 1
            total_size += obj['Size']
            
            if file_count % 50 == 0:
                print(f"  ↳ {file_count} files...")
    
    size_mb = total_size / (1024 * 1024)
    print(f"✅ Downloaded {file_count} file(s) ({size_mb:.1f} MB)")
    
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
    s3 = get_s3_client()
    
    print(f"📋 Listing datasets in s3://{bucket}/{prefix}")
    print()
    
    # Use delimiter to get "folders" at top level
    paginator = s3.get_paginator('list_objects_v2')
    datasets = []
    folder_stats = {}
    
    # Get all objects
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue
            
        for obj in page['Contents']:
            s3_key = obj['Key']
            
            # Extract top-level folder (dataset name)
            parts = s3_key.split('/')
            if len(parts) > 1:
                dataset_name = parts[0]
                
                if dataset_name not in folder_stats:
                    folder_stats[dataset_name] = {
                        'name': dataset_name,
                        'file_count': 0,
                        'total_size': 0,
                        'last_modified': obj['LastModified']
                    }
                
                folder_stats[dataset_name]['file_count'] += 1
                folder_stats[dataset_name]['total_size'] += obj['Size']
                
                # Keep the most recent modification time
                if obj['LastModified'] > folder_stats[dataset_name]['last_modified']:
                    folder_stats[dataset_name]['last_modified'] = obj['LastModified']
    
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
    s3_prefix = f"{date_folder}/data"
    
    return upload_to_s3(str(data_dir), s3_prefix=s3_prefix)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="S3 sync for VMEvalKit - upload, download, and list datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ORIGINAL USAGE (Backward Compatible):
  # Default: Upload data/ folder with auto-generated timestamp (YYYYMMDDHHMM/data/)
  python data/s3_sync.py
  
  # Upload with specific date prefix
  python data/s3_sync.py --date 202411181530
  
  # Upload and log version
  python data/s3_sync.py --log

NEW COMMANDS:

  LIST - See what datasets are available in S3:
    # List all datasets (shows YYYYMMDDHHMM folders)
    python data/s3_sync.py list
    
    # List with specific prefix
    python data/s3_sync.py list --prefix 202411
    
  UPLOAD - Upload specific files or directories:
    # Upload with date prefix (maintains YYYYMMDDHHMM/data/ format)
    python data/s3_sync.py upload ./data/outputs
    
    # Upload with custom S3 prefix
    python data/s3_sync.py upload ./data/outputs --prefix my-experiment
    
    # Upload specific file
    python data/s3_sync.py upload ./results.json --prefix 202411181530/data
    
  DOWNLOAD - Download datasets from S3:
    # Download using date prefix (YYYYMMDDHHMM/data/ format)
    python data/s3_sync.py download 202411181234/data
    
    # Download with full S3 URI
    python data/s3_sync.py download s3://vmevalkit/202411181234/data
    
    # Download to specific local directory
    python data/s3_sync.py download 202411181234/data --output ./restored-data

S3 STRUCTURE (Original Design):
  s3://vmevalkit/
  ├── 202411181234/data/    # YYYYMMDDHHMM/data/ format
  │   ├── questions/
  │   ├── outputs/
  │   └── evaluations/
  └── 202411181530/data/
      └── ...

CREDENTIALS:
  Loaded from .env file:
    - AWS_ACCESS_KEY_ID
    - AWS_SECRET_ACCESS_KEY
    - AWS_REGION (defaults to us-east-2)
    - AWS_S3_BUCKET (defaults to 'vmevalkit')
        """
    )
    
    # Add backward-compatible flags at top level (for original usage)
    parser.add_argument('--date', help='Date folder (YYYYMMDDHHMM) - for backward compatibility')
    parser.add_argument('--log', action='store_true', help='Log version after upload - for backward compatibility')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Upload command
    upload_parser = subparsers.add_parser('upload', help='Upload files/directory to S3')
    upload_parser.add_argument('path', help='Local file or directory to upload')
    upload_parser.add_argument('--prefix', help='S3 prefix/folder (default: YYYYMMDDHHMM/data)')
    upload_parser.add_argument('--bucket', help='S3 bucket (default: from environment)')
    upload_parser.add_argument('--date', help='Date prefix (YYYYMMDDHHMM)')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download data from S3')
    download_parser.add_argument('s3_uri', help='S3 URI (s3://bucket/prefix) or just prefix')
    download_parser.add_argument('--output', help='Local destination path (default: ./downloads/{prefix})')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available S3 datasets')
    list_parser.add_argument('--prefix', default='', help='Filter by prefix')
    list_parser.add_argument('--bucket', help='S3 bucket (default: from environment)')
    
    # Sync command (for explicit sync usage)
    sync_parser = subparsers.add_parser('sync', help='Sync data folder to S3')
    sync_parser.add_argument('--date', help='Date folder (YYYYMMDDHHMM)')
    sync_parser.add_argument('--log', action='store_true', help='Log version after upload')
    
    args = parser.parse_args()
    
    # BACKWARD COMPATIBILITY: Default to sync if no command provided (original behavior)
    # This preserves the original design where `python data/s3_sync.py` just uploads
    if args.command is None:
        if args.date or args.log:
            # Original usage: python data/s3_sync.py --date 202411181530 --log
            print("📦 Running sync (original usage)")
        else:
            # Default: python data/s3_sync.py
            print("📦 Running default sync (use --help to see new commands)")
        
        # Use sync command with the provided or default date/log values
        args.command = 'sync'
        # args.date and args.log are already set from top-level parser
    
    # Execute command with error handling
    try:
        if args.command == 'upload':
            date_prefix = getattr(args, 'date', None)
            s3_uri = upload_to_s3(args.path, s3_prefix=args.prefix, bucket=args.bucket, date_prefix=date_prefix)
            return 0
            
        elif args.command == 'download':
            local_path = download_from_s3(args.s3_uri, local_path=args.output)
            return 0
            
        elif args.command == 'list':
            list_s3_datasets(bucket=args.bucket, prefix=args.prefix)
            return 0
            
        elif args.command == 'sync':
            s3_uri = sync_to_s3(date_prefix=args.date)
            
            # Log version if requested (original feature)
            if args.log:
                from data_logging.version_tracker import log_version
                version = input("Version number (e.g. 1.0): ")
                # Extract stats from sync result
                log_version(version, s3_uri, {'change': 'Data sync'})
                print(f"✅ Version {version} logged")
            
            return 0
            
        else:
            print(f"Unknown command: {args.command}")
            return 1
            
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        return 1
    except ClientError as e:
        print(f"❌ AWS S3 error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())