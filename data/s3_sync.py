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
    try:
        s3 = boto3.client("s3",
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=os.getenv("AWS_REGION", "us-east-2")
        )
        # Test connection
        s3.head_bucket(Bucket=bucket)
    except ClientError as e:
        raise ValueError(f"S3 connection failed: {e}")
    except Exception as e:
        raise ValueError(f"Failed to create S3 client: {e}")
    
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
            # Add current directory to path for import
            sys.path.insert(0, str(Path(__file__).parent))
            from data_logging.version_tracker import log_version
            version = input("📝 Version number (e.g. 1.0): ").strip()
            if version:
                log_version(version, s3_uri, {'size_mb': size_mb, 'files': file_count})
            else:
                print("⚠️  No version provided - skipping version logging")
        except ImportError as e:
            print(f"⚠️  Could not import version tracker: {e}")
        except Exception as e:
            print(f"⚠️  Version logging failed: {e}")
    
    return s3_uri


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Sync VMEvalKit data to S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sync with auto-generated timestamp
  python data/s3_sync.py
  
  # Sync with specific date prefix
  python data/s3_sync.py --date 20250115
  
  # Sync and log version
  python data/s3_sync.py --log

Required Environment Variables:
  S3_BUCKET - S3 bucket name
  AWS_ACCESS_KEY_ID - AWS access key
  AWS_SECRET_ACCESS_KEY - AWS secret key
  AWS_REGION (optional) - AWS region (default: us-east-2)
        """
    )
    parser.add_argument("--date", help="Date folder prefix (YYYYMMDDHHMM)")
    parser.add_argument("--log", action="store_true", help="Log version after upload")
    
    args = parser.parse_args()
    
    try:
        print("🚀 Starting S3 sync...")
        sync_to_s3(date_prefix=args.date)
        print("🎉 S3 sync completed successfully!")
        return 0
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("💡 Check your environment variables and settings")
        return 1
    except Exception as e:
        print(f"❌ Sync Failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())