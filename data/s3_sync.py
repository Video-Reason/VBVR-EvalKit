import os
import sys
import datetime
from pathlib import Path
from typing import Optional

import boto3
from botocore.exceptions import ClientError


def get_s3_client():
    session = boto3.session.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN"),
        region_name=os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1")),
    )
    return session.client("s3")


def get_date_prefix(date_override: Optional[str] = None) -> str:
    if date_override:
        return date_override
    # YYYYMMDD
    return datetime.datetime.now().strftime("%Y%m%d")


def sync_directory_to_s3(
    local_dir: Path,
    bucket: str,
    s3_prefix: str,
) -> None:
    s3 = get_s3_client()
    local_dir = local_dir.resolve()

    if not local_dir.exists() or not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory not found or not a directory: {local_dir}")

    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = Path(root) / filename
            rel_path = local_path.relative_to(local_dir)
            s3_key = f"{s3_prefix.rstrip('/')}/{rel_path.as_posix()}"
            try:
                s3.upload_file(str(local_path), bucket, s3_key)
            except ClientError as e:
                raise RuntimeError(f"Failed to upload {local_path} to s3://{bucket}/{s3_key}: {e}")


def write_latest_marker(bucket: str, latest_path: str) -> None:
    s3 = get_s3_client()
    marker_key = "latest_data_path.txt"
    try:
        s3.put_object(Bucket=bucket, Key=marker_key, Body=latest_path.encode("utf-8"))
    except ClientError as e:
        raise RuntimeError(f"Failed to write latest marker to s3://{bucket}/{marker_key}: {e}")


def sync_data_folder(
    data_dir: Path = Path(__file__).resolve().parent,
    bucket: str = os.getenv("S3_BUCKET", "vmevalkit"),
    date_prefix: Optional[str] = None,
) -> str:
    """
    Sync complete VMEvalKit data folder to S3 with date-based versioning.
    
    Syncs the entire data/ directory which includes:
    - questions/ - Dataset files, images, and task definitions
    - outputs/ - Model-generated videos and inference results
    - VERSION.md - Dataset version tracking
    - s3_sync.py - This sync script
    
    Creates: s3://<bucket>/<YYYYMMDD>/data/
    """
    # s3://<bucket>/<YYYYMMDD>/data
    date_folder = get_date_prefix(date_prefix)
    s3_prefix = f"{date_folder}/data"
    sync_directory_to_s3(local_dir=data_dir, bucket=bucket, s3_prefix=s3_prefix)

    latest_uri = f"s3://{bucket}/{s3_prefix}"
    write_latest_marker(bucket=bucket, latest_path=latest_uri)
    return latest_uri


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Sync complete VMEvalKit data package (questions + outputs) to S3 with date-versioned prefix.")
    parser.add_argument("--data-dir", type=str, default=str(Path(__file__).resolve().parent), help="Path to local data directory")
    parser.add_argument("--bucket", type=str, default=os.getenv("S3_BUCKET", "vmevalkit"), help="S3 bucket name")
    parser.add_argument("--date", type=str, default=None, help="Override date folder (YYYYMMDD)")
    args = parser.parse_args(argv)

    uri = sync_data_folder(data_dir=Path(args.data_dir), bucket=args.bucket, date_prefix=args.date)
    print(uri)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


