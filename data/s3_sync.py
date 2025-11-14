#!/usr/bin/env python3
"""Simple S3 sync for VMEvalKit data."""

import os
import sys
import datetime
from pathlib import Path
import boto3

# Load environment variables from .env file
def load_env_file():
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Remove quotes if present
                    value = value.strip('"\'')
                    os.environ[key] = value

# Load .env file at module import
load_env_file()


def download_from_s3(date_prefix: str, data_dir: Path = None, subfolder: str = None) -> str:
    """
    Download data from S3.
    
    Args:
        date_prefix: Date folder to download (YYYYMMDDHHMM)
        data_dir: Local data directory (default: ./data)
        subfolder: Specific subfolder to download (e.g., 'evaluations', 'outputs')
        
    Returns:
        Local path of downloaded data
    """
    # Defaults
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent
    
    # S3 setup - validate required environment variables
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET environment variable is required")
    
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not aws_access_key or not aws_secret_key:
        raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are required")
    
    # Create S3 client
    s3 = boto3.client("s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=os.getenv("AWS_REGION", "us-east-2")
    )
    # Test connection
    s3.head_bucket(Bucket=bucket)
    
    # Set up S3 prefix
    s3_prefix = f"{date_prefix}/data"
    if subfolder:
        s3_prefix = f"{s3_prefix}/{subfolder}"
    
    print(f"📥 Downloading from s3://{bucket}/{s3_prefix}/")
    
    # Create local directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download files
    file_count = 0
    total_size = 0
    
    # List all objects with the prefix
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)
    
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                s3_key = obj['Key']
                
                # Skip directories (keys ending with '/')
                if s3_key.endswith('/'):
                    continue
                
                # Create local path
                rel_path = s3_key.replace(f"{date_prefix}/data/", "")
                local_path = data_dir / rel_path
                local_path.parent.mkdir(parents=True, exist_ok=True)
                
                s3.download_file(bucket, s3_key, str(local_path))
                file_count += 1
                total_size += obj['Size']
                
                if file_count % 50 == 0:
                    print(f"  ↳ {file_count} files...")
    
    size_mb = total_size / (1024 * 1024)
    
    print(f"✅ Downloaded {file_count} files ({size_mb:.1f} MB)")
    print(f"📍 Location: {data_dir}")
    
    return str(data_dir)


def sync_to_s3(data_dir: Path = None, date_prefix: str = None) -> str:
    """
    Sync data folder to S3.
    
    Args:
        data_dir: Path to data directory (default: ./data)
        date_prefix: Date folder (default: today's date YYYYMMDDHHMM)
        
    Returns:
        S3 URI of uploaded data
    """
    # Defaults
    if data_dir is None:
        data_dir = Path(__file__).resolve().parent
    
    # Validate data directory
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # S3 setup - validate required environment variables
    bucket = os.getenv("S3_BUCKET")
    if not bucket:
        raise ValueError("S3_BUCKET environment variable is required")
    
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not aws_access_key or not aws_secret_key:
        raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are required")
    
    date_folder = date_prefix or datetime.datetime.now().strftime("%Y%m%d%H%M")
    s3_prefix = f"{date_folder}/data"
    
    # Create S3 client
    s3 = boto3.client("s3",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=os.getenv("AWS_REGION", "us-east-2")
    )
    # Test connection
    s3.head_bucket(Bucket=bucket)
    
    # Count files
    file_count = 0
    total_size = 0
    
    print(f"📦 Syncing to s3://{bucket}/{s3_prefix}/")
    
    # Upload files
    for root, _, files in os.walk(data_dir):
        for filename in files:
            local_path = Path(root) / filename
            rel_path = local_path.relative_to(data_dir)
            s3_key = f"{s3_prefix}/{rel_path.as_posix()}"
            
            s3.upload_file(str(local_path), bucket, s3_key)
            file_count += 1
            total_size += local_path.stat().st_size
            
            if file_count % 100 == 0:
                print(f"  ↳ {file_count} files...")
    
    s3_uri = f"s3://{bucket}/{s3_prefix}"
    size_mb = total_size / (1024 * 1024)
    
    print(f"✅ Uploaded {file_count} files ({size_mb:.1f} MB)")
    print(f"📍 Location: {s3_uri}")
    
    # Log version if requested
    if '--log' in sys.argv:
        # Add current directory to path for import
        sys.path.insert(0, str(Path(__file__).parent))
        from data_logging.version_tracker import log_version
        version = input("📝 Version number (e.g. 1.0): ").strip()
        if version:
            log_version(version, s3_uri, {'size_mb': size_mb, 'files': file_count})
        else:
            print("⚠️  No version provided - skipping version logging")
    
    return s3_uri


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sync VMEvalKit data with S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload: Basic sync with auto-generated timestamp
  python data/s3_sync.py
  
  # Upload: Sync with specific date prefix
  python data/s3_sync.py --date 20250115
  
  # Upload: Sync and log version
  python data/s3_sync.py --log
  
  # Download: Get evaluation results from specific date
  python data/s3_sync.py --download --date 202510241120 --subfolder evaluations
  
  # Download: Get all data from specific date
  python data/s3_sync.py --download --date 202510241120

Required Environment Variables:
  S3_BUCKET - S3 bucket name
  AWS_ACCESS_KEY_ID - AWS access key
  AWS_SECRET_ACCESS_KEY - AWS secret key
  AWS_REGION (optional) - AWS region (default: us-east-2)
        """
    )
    parser.add_argument("--date", help="Date folder prefix (YYYYMMDDHHMM)")
    parser.add_argument("--log", action="store_true", help="Log version after upload")
    parser.add_argument("--download", action="store_true", help="Download data from S3 instead of uploading")
    parser.add_argument("--subfolder", help="Specific subfolder to download (e.g., evaluations, outputs)")
    
    args = parser.parse_args()
    
    if args.download:
        if not args.date:
            print("❌ --date is required when downloading")
            return 1
        
        print("📥 Starting S3 download...")
        download_from_s3(date_prefix=args.date, subfolder=args.subfolder)
        print("🎉 S3 download completed successfully!")
    else:
        print("📤 Starting S3 upload...")
        sync_to_s3(date_prefix=args.date)
        print("🎉 S3 upload completed successfully!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())