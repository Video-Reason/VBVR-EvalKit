#!/usr/bin/env python3
"""Simple S3 sync for VMEvalKit data."""

import os
import sys
import datetime
from pathlib import Path
import boto3
from botocore.exceptions import ClientError


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
    
    # S3 setup
    bucket = os.getenv("S3_BUCKET", "vmevalkit")
    date_folder = date_prefix or datetime.datetime.now().strftime("%Y%m%d%H%M")
    s3_prefix = f"{date_folder}/data"
    
    # Create S3 client
    s3 = boto3.client("s3",
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        region_name=os.getenv("AWS_REGION", "us-east-2")
    )
    
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
            
            try:
                s3.upload_file(str(local_path), bucket, s3_key)
                file_count += 1
                total_size += local_path.stat().st_size
                
                if file_count % 100 == 0:
                    print(f"  ↳ {file_count} files...")
            except Exception as e:
                print(f"  ⚠️ Failed: {rel_path}: {e}")
    
    s3_uri = f"s3://{bucket}/{s3_prefix}"
    size_mb = total_size / (1024 * 1024)
    
    print(f"✅ Uploaded {file_count} files ({size_mb:.1f} MB)")
    print(f"📍 Location: {s3_uri}")
    
    # Log version if requested
    if '--log' in sys.argv:
        try:
            from data_logging.version_tracker import log_version
            version = input("Version number (e.g. 1.0): ")
            log_version(version, s3_uri, {'size_mb': size_mb, 'files': file_count})
        except ImportError:
            pass
    
    return s3_uri


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync data to S3")
    parser.add_argument("--date", help="Date folder (YYYYMMDDHHMM)")
    parser.add_argument("--log", action="store_true", help="Log version after upload")
    
    args = parser.parse_args()
    
    try:
        sync_to_s3(date_prefix=args.date)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())